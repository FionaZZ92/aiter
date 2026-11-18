# SPDX-License-Identifier: Apache-2.0
"""CDNA3 implementation of MegaMoEV2's FP8 x FP4 scaled tile ABI.

The default uses four native FNUZ FP8 MFMAs with software FP4 unpacking
and K32 E8M0 scaling. An eight-BF16-MFMA reference path is also available.
"""
import os
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

IS_GFX942=str(get_rocm_arch()).startswith('gfx942')
USE_NATIVE_FP8=os.environ.get('AITER_MEGA_GFX942_MMA','fp8')=='fp8'
UNPACK_MODE=os.environ.get('AITER_MEGA_GFX942_UNPACK','reuse' if IS_GFX942 else 'scalar')
USE_PACKED_LUT=UNPACK_MODE in ('lut','reuse','fast','fastreuse')
REUSE_DECODED_B=UNPACK_MODE in ('reuse','fastreuse')
USE_FAST_PACKED_WEIGHTS=UNPACK_MODE in ('fast','fastreuse')
if IS_GFX942 and USE_FAST_PACKED_WEIGHTS and not USE_NATIVE_FP8:
    raise ValueError('fast packed weights require the native FP8 compute path')
GFX942_MODE_ID=(100 if USE_NATIVE_FP8 else 0)+{
    'scalar':0,'lut':1,'reuse':2,'fast':3,'fastreuse':4,
}.get(UNPACK_MODE,0)
OPERAND_MODE=os.environ.get('AITER_MEGA_GFX942_OPERANDS','legacy')
if OPERAND_MODE not in ('legacy','native_a','native_ab','early','native_ab_early','native_ab_k32'):
    raise ValueError('unknown gfx942 operand mode: '+OPERAND_MODE)
NATIVE_A=IS_GFX942 and OPERAND_MODE in ('native_a','native_ab','native_ab_early','native_ab_k32')
NATIVE_B=IS_GFX942 and OPERAND_MODE in ('native_ab','native_ab_early','native_ab_k32')
EARLY_SCALE=IS_GFX942 and OPERAND_MODE in ('early','native_ab_early')
K32_SCOPE=IS_GFX942 and OPERAND_MODE=='native_ab_k32'
if IS_GFX942 and OPERAND_MODE!='legacy' and not (USE_NATIVE_FP8 and REUSE_DECODED_B):
    raise ValueError('native/early operands require native FP8 and decoded-B reuse')
GFX942_MODE_ID+=1000*('legacy','native_a','native_ab','early','native_ab_early','native_ab_k32').index(OPERAND_MODE)


@flyc.jit
def unpack_fp4x8_lut(word):
    """Borrow the A8W4 packed v_perm decoder without changing scales.

    Magnitude LUT contains exact unscaled FNUZ encodings for
    [0,.5,1,1.5,2,3,4,6]; the sign LUT excludes negative zero.
    """
    raw=fx.Uint32(word)
    even_selector=raw&fx.Uint32(0x07070707)
    odd_selector=(raw>>fx.Uint32(4))&fx.Uint32(0x07070707)
    lo=fx.Uint32(0x44403800);hi=fx.Uint32(0x54504c48)
    if fx.const_expr(USE_FAST_PACKED_WEIGHTS):
        # setup has packed K0..3 in the low nibbles and K4..7 in the
        # high nibbles. No sign lookup or byte interleave is needed.
        first=fx.Uint32(rocdl.perm_b32(hi,lo,even_selector))|((raw<<fx.Uint32(4))&fx.Uint32(0x80808080))
        second=fx.Uint32(rocdl.perm_b32(hi,lo,odd_selector))|(raw&fx.Uint32(0x80808080))
        return Vec.from_elements([first.bitcast(fx.Int32),second.bitcast(fx.Int32)],fx.Int32)
    sign_lo=fx.Uint32(0x80808000);sign_hi=fx.Uint32(0x80808080)
    even=fx.Uint32(rocdl.perm_b32(hi,lo,even_selector))
    even=even|(fx.Uint32(rocdl.perm_b32(sign_hi,sign_lo,even_selector))&(raw<<fx.Uint32(4)))
    odd=fx.Uint32(rocdl.perm_b32(hi,lo,odd_selector))
    odd=odd|(fx.Uint32(rocdl.perm_b32(sign_hi,sign_lo,odd_selector))&raw)
    first=fx.Int32(rocdl.perm_b32(odd,even,fx.Uint32(0x05010400)))
    second=fx.Int32(rocdl.perm_b32(odd,even,fx.Uint32(0x07030602)))
    return Vec.from_elements([first,second],fx.Int32)


@flyc.jit
def fp8_ocp_bits(value):
    """Finite, saturating E4M3FN conversion with round-to-nearest-even."""
    raw=value.bitcast(fx.Int32)
    absolute=raw&fx.Int32(0x7fffffff)
    magnitude=absolute.bitcast(fx.Float32)
    rounded=absolute+fx.Int32(0x7ffff)+((absolute>>fx.Int32(20))&fx.Int32(1))
    normal=(rounded>>fx.Int32(20))-fx.Int32(120*8)
    sub=((magnitude*fx.Float32(512.0)+fx.Float32(8388608.0)).bitcast(fx.Int32)
         &fx.Int32(0x7fffff))
    exp=(absolute>>fx.Int32(23))&fx.Int32(255)
    bits=(exp<=fx.Int32(120)).select(sub,normal)
    bits=(bits>fx.Int32(126)).select(fx.Int32(126),bits)
    return bits|((raw>>fx.Int32(24))&fx.Int32(128))


@flyc.jit
def cvt_pk_fp8_f32(result_type,a,b,old,half):
    """Architecture-correct OCP packing for MegaMoE's FP8 wire format."""
    if fx.const_expr(IS_GFX942):
        packed=fp8_ocp_bits(a)|(fp8_ocp_bits(b)<<fx.Int32(8))
        shift=half*16
        output=(old&fx.Int32(~(0xffff<<shift)))|(packed<<fx.Int32(shift))
    else:
        output=rocdl.cvt_pk_fp8_f32(result_type,a,b,old,half)
    return output


def cvt_scalef32_pk_fp8_f32(result_type,old,a,b,scale,half):
    if not IS_GFX942:
        return rocdl.cvt_scalef32_pk_fp8_f32(result_type,old,a,b,scale,half)
    previous=Vec(old).bitcast(fx.Int32)[0]
    s=fx.Float32(scale)
    bits=cvt_pk_fp8_f32(T.i32,fx.Float32(a)/s,fx.Float32(b)/s,previous,half)
    return Vec.from_elements([bits],fx.Int32).bitcast(fx.Int16).ir_value()


def cvt_pk_f32_fp8(*,res,src,word_sel):
    if not IS_GFX942:
        return rocdl.cvt_pk_f32_fp8(res,src,word_sel)
    raw=fx.Int32(src)
    values=[]
    for i in range(2):
        byte=(raw>>fx.Int32((int(word_sel)*2+i)*8))&fx.Int32(255)
        exp=(byte>>fx.Int32(3))&fx.Int32(15)
        mant=byte&fx.Int32(7)
        norm=(((exp+fx.Int32(120))<<fx.Int32(23))|(mant<<fx.Int32(20))).bitcast(fx.Float32)
        v=(exp==fx.Int32(0)).select(mant.to(fx.Float32)*fx.Float32(1./512.),norm)
        values.append(((byte&fx.Int32(128))!=fx.Int32(0)).select(-v,v))
    return Vec.from_elements(values,fx.Float32).ir_value()


@flyc.jit
def scaled_mma_fp8_fp4_bf16(c, a, b, scale_a, scale_b, opsel_a, opsel_b):
    lane=fx.Int32(fx.thread_idx.x)&fx.Int32(63)
    row=lane&fx.Int32(15)
    klane=lane>>fx.Int32(4)
    av=Vec(a.load())
    bv=Vec(b.load())
    acc=Vec(c.load())
    atom=fx.make_mma_atom(rocdl.MFMA(16,16,16,fx.BFloat16))
    # FP8 A owns two K16 slabs 64 apart; FP4 B owns contiguous K32.
    # Both E8M0 lane layouts own a K32 group. A CDNA3 K16 instruction
    # instead owns four K elements/lane, so permute each operand separately.
    for step in range_constexpr(8):
        source_a=row+fx.Int32((step%4)*16)
        source=row+fx.Int32((step//2)*16)
        shuffled_a=Vec.from_elements([
            fx.Int32(rocdl.ds_bpermute(T.i32,source_a*fx.Int32(4),av[w]))
            for w in range_constexpr(8)],fx.Int32)
        shuffled_b=Vec.from_elements([
            fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),bv[w]))
            for w in range_constexpr(4)],fx.Int32)
        aw=shuffled_a[fx.Int32((step//4)*4)+klane]
        bw=shuffled_b[fx.Int32((step%2)*2)+(klane>>fx.Int32(1))]
        bw=bw>>((klane&fx.Int32(1))*fx.Int32(16))
        sa=fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),scale_a))
        sb=fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),scale_b))
        ea=(sa>>fx.Int32(opsel_a*8))&fx.Int32(255)
        eb=(sb>>fx.Int32(opsel_b*8))&fx.Int32(255)
        # E8M0 scales are powers of two. Normal test/model scales fit f32.
        scale_af=(ea<<fx.Int32(23)).bitcast(fx.Float32)
        scale_bf=(eb<<fx.Int32(23)).bitcast(fx.Float32)
        scale_af=(ea==fx.Int32(0)).select(fx.Float32(2.0**-127),scale_af)
        scale_bf=(eb==fx.Int32(0)).select(fx.Float32(2.0**-127),scale_bf)
        aa=[]
        bb=[]
        for element in range_constexpr(4):
            byte=(aw>>fx.Int32(8*element))&fx.Int32(255)
            exp=(byte>>fx.Int32(3))&fx.Int32(15)
            mant=byte&fx.Int32(7)
            normal=(((exp+fx.Int32(120))<<fx.Int32(23))
                    |(mant<<fx.Int32(20))).bitcast(fx.Float32)
            sub=mant.to(fx.Float32)*fx.Float32(1.0/512.0)
            magnitude=(exp==fx.Int32(0)).select(sub,normal)
            value=((byte&fx.Int32(128))!=fx.Int32(0)).select(-magnitude,magnitude)
            aa.append((value*scale_af).to(fx.BFloat16))
            nibble=(bw>>fx.Int32(4*element))&fx.Int32(15)
            mag=nibble&fx.Int32(7)
            e=mag>>fx.Int32(1)
            bits=((e+fx.Int32(126))<<fx.Int32(7))|((mag&fx.Int32(1))<<fx.Int32(6))
            bits=(mag==fx.Int32(0)).select(fx.Int32(0),bits)
            bits=(mag==fx.Int32(1)).select(fx.Int32(0x3f00),bits)
            bits=bits|((nibble&fx.Int32(8))<<fx.Int32(12))
            weight=bits.to(fx.Int16).bitcast(fx.BFloat16).to(fx.Float32)
            bb.append((weight*scale_bf).to(fx.BFloat16))
        af=fx.make_rmem_tensor(4,fx.BFloat16)
        bf=fx.make_rmem_tensor(4,fx.BFloat16)
        cf=fx.make_rmem_tensor(4,fx.Float32)
        af.store(Vec.from_elements(aa,fx.BFloat16))
        bf.store(Vec.from_elements(bb,fx.BFloat16))
        cf.store(acc)
        fx.gemm(atom,cf,af,bf,cf)
        acc=Vec(cf.load())
    c.store(acc)


@flyc.jit
def scaled_mma_fp8_fp4_native(c,a,b,scale_a,scale_b,opsel_a,opsel_b):
    lane=fx.Int32(fx.thread_idx.x)&fx.Int32(63)
    row=lane&fx.Int32(15)
    klane=lane>>fx.Int32(4)
    av=Vec(a.load())
    bv=Vec(b.load())
    acc=Vec(c.load())
    atom=fx.make_mma_atom(rocdl.MFMA(16,16,32,fx.Float8E4M3FNUZ))
    # The same normal FP8 byte denotes half its OCP value in FNUZ.
    # Keep those bytes, unpack FP4 to exact FNUZ values, and compensate
    # by two in each K32 partial-product scale. Mask OCP negative zero.
    for group in range_constexpr(4):
        source_a=row+((fx.Int32(group*2)+(klane>>fx.Int32(1)))&fx.Int32(3))*fx.Int32(16)
        ashuffle=Vec.from_elements([
            fx.Int32(rocdl.ds_bpermute(T.i32,source_a*fx.Int32(4),av[group//2*4+w]))
            for w in range_constexpr(4)],fx.Int32)
        apair=[]
        for h in range_constexpr(2):
            word=ashuffle[(klane&fx.Int32(1))*fx.Int32(2)+fx.Int32(h)]
            if fx.const_expr(USE_PACKED_LUT):
                magnitude=word&fx.Int32(0x7f7f7f7f)
                zero_sign=(~(magnitude+fx.Int32(0x7f7f7f7f)))&fx.Int32(0x80808080)
                cleaned=word&~zero_sign
            else:
                cleaned=fx.Int32(0)
                for i in range_constexpr(4):
                    byte=(word>>fx.Int32(i*8))&fx.Int32(255)
                    byte=(byte==fx.Int32(128)).select(fx.Int32(0),byte)
                    cleaned=cleaned|(byte<<fx.Int32(i*8))
            apair.append(cleaned)
        source=row+fx.Int32(group*16)
        bshuffle=Vec.from_elements([
            fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),bv[w]))
            for w in range_constexpr(4)],fx.Int32)
        bword=bshuffle[klane]
        if fx.const_expr(USE_PACKED_LUT):
            decoded=unpack_fp4x8_lut(bword)
            bpair=[decoded[0],decoded[1]]
        else:
            bpair=[]
            for h in range_constexpr(2):
                packed=fx.Int32(0)
                for i in range_constexpr(4):
                    n=(bword>>fx.Int32((h*4+i)*4))&fx.Int32(15)
                    mag=n&fx.Int32(7)
                    bits=(((mag>>fx.Int32(1))+fx.Int32(7))<<fx.Int32(3))|((mag&fx.Int32(1))<<fx.Int32(2))
                    bits=(mag==fx.Int32(1)).select(fx.Int32(56),bits)
                    bits=(mag==fx.Int32(0)).select(fx.Int32(0),bits|((n&fx.Int32(8))<<fx.Int32(4)))
                    packed=packed|(bits<<fx.Int32(i*8))
                bpair.append(packed)
        af=fx.make_rmem_tensor(2,fx.Int32)
        bf=fx.make_rmem_tensor(2,fx.Int32)
        cf=fx.make_rmem_tensor(4,fx.Float32)
        af.store(Vec.from_elements(apair,fx.Int32))
        bf.store(Vec.from_elements(bpair,fx.Int32))
        cf.store(Vec.filled(4,0.,fx.Float32))
        fx.gemm(atom,cf,af,bf,cf)
        partial=Vec(cf.load())
        sb=fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),scale_b))
        eb=(sb>>fx.Int32(opsel_b*8))&fx.Int32(255)
        values=[]
        for i in range_constexpr(4):
            arow=klane*fx.Int32(4)+fx.Int32(i)
            sa=fx.Int32(rocdl.ds_bpermute(T.i32,(arow+fx.Int32(group*16))*fx.Int32(4),scale_a))
            ea=(sa>>fx.Int32(opsel_a*8))&fx.Int32(255)
            multiplier=((ea+eb-fx.Int32(126))<<fx.Int32(23)).bitcast(fx.Float32)
            values.append(acc[i]+partial[i]*multiplier)
        acc=Vec.from_elements(values,fx.Float32)
    c.store(acc)


@flyc.jit
def scaled_mma_fp8_fp4(c,a,b,scale_a,scale_b,opsel_a,opsel_b):
    if fx.const_expr(USE_NATIVE_FP8):
        if fx.const_expr(REUSE_DECODED_B):
            from .gfx942_operand_reuse import prepare_b,mma_prepared_b
            prepared=prepare_b(Vec(b.load()),scale_b,opsel_b)
            mma_prepared_b(c,a,scale_a,opsel_a,prepared)
        else:
            scaled_mma_fp8_fp4_native(c,a,b,scale_a,scale_b,opsel_a,opsel_b)
    else:
        scaled_mma_fp8_fp4_bf16(c,a,b,scale_a,scale_b,opsel_a,opsel_b)
