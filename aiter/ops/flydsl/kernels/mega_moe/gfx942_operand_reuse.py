# SPDX-License-Identifier: Apache-2.0
"""A8W4-style B-major operand reuse for the gfx942 MegaMoEV2 port.

Packed B is decoded once per N/K tile and reused across M fragments.
A stays a lazy LDS read, so a full decoded A tile is not held in VGPRs.
Activation quantization, E8M0 scales and accumulation order are unchanged.
"""
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .gfx942_scaled_mma import unpack_fp4x8_lut,NATIVE_A,NATIVE_B,EARLY_SCALE


def prepare_b_group(raw,scale,opsel,group):
    """One native K32 fragment, released after its M-fragment cluster."""
    lane=fx.Int32(fx.thread_idx.x)&fx.Int32(63)
    source=(lane&fx.Int32(15))+fx.Int32(group*16)
    frag=fx.make_rmem_tensor(2,fx.Int32)
    frag.store(unpack_fp4x8_lut(Vec(raw)[group]))
    packed=fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),scale))
    exponent=(packed>>fx.Int32(opsel*8))&fx.Int32(255)
    return frag,exponent


def mma_native_group(acc,a,scale_a,opsel_a,prepared,group):
    """Keep K32 accumulation order; prepare four scale factors before MFMA."""
    lane=fx.Int32(fx.thread_idx.x)&fx.Int32(63)
    klane=lane>>fx.Int32(4)
    bfrag,bexp=prepared
    multipliers=[]
    for i in range(4):
        source=klane*fx.Int32(4)+fx.Int32(i+group*16)
        packed=fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),scale_a))
        exponent=(packed>>fx.Int32(opsel_a*8))&fx.Int32(255)
        multipliers.append(((exponent+bexp-fx.Int32(126))<<fx.Int32(23)).bitcast(fx.Float32))
    raw=Vec(a);values=[]
    for h in range(2):
        word=raw[group*2+h]
        mag=word&fx.Int32(0x7f7f7f7f)
        zero=(~(mag+fx.Int32(0x7f7f7f7f)))&fx.Int32(0x80808080)
        values.append(word&~zero)
    af=fx.make_rmem_tensor(2,fx.Int32)
    cf=fx.make_rmem_tensor(4,fx.Float32)
    af.store(Vec.from_elements(values,fx.Int32))
    cf.store(Vec.filled(4,0.,fx.Float32))
    atom=fx.make_mma_atom(rocdl.MFMA(16,16,32,fx.Float8E4M3FNUZ))
    fx.gemm(atom,cf,af,bfrag,cf)
    partial=Vec(cf.load());acc=Vec(acc)
    return Vec.from_elements([acc[i]+partial[i]*multipliers[i] for i in range(4)],fx.Float32)


def prepare_b(raw,scale,opsel):
    lane=fx.Int32(fx.thread_idx.x)&fx.Int32(63)
    column=lane&fx.Int32(15)
    klane=lane>>fx.Int32(4)
    raw=Vec(raw)
    fragments=[];exponents=[]
    for group in range(4):
        source=column+fx.Int32(group*16)
        words=Vec.from_elements([
            fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),raw[i]))
            for i in range(4)],fx.Int32) if not NATIVE_B else raw
        value=unpack_fp4x8_lut(words[group] if NATIVE_B else words[klane])
        frag=fx.make_rmem_tensor(2,fx.Int32)
        frag.store(value)
        fragments.append(frag)
        packed=fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),scale))
        exponents.append((packed>>fx.Int32(opsel*8))&fx.Int32(255))
    return fragments,exponents


def mma_prepared_b(c,a,scale_a,opsel_a,prepared):
    lane=fx.Int32(fx.thread_idx.x)&fx.Int32(63)
    row=lane&fx.Int32(15)
    klane=lane>>fx.Int32(4)
    raw=Vec(a.load());acc=Vec(c.load())
    bfrags,bexp=prepared
    atom=fx.make_mma_atom(rocdl.MFMA(16,16,32,fx.Float8E4M3FNUZ))
    early=[]
    if fx.const_expr(EARLY_SCALE):
        for group in range(4):
            factors=[]
            for i in range(4):
                source_scale=klane*fx.Int32(4)+fx.Int32(i+group*16)
                packed=fx.Int32(rocdl.ds_bpermute(T.i32,source_scale*fx.Int32(4),scale_a))
                exponent=(packed>>fx.Int32(opsel_a*8))&fx.Int32(255)
                factors.append(((exponent+bexp[group]-fx.Int32(126))<<fx.Int32(23)).bitcast(fx.Float32))
            early.append(factors)
    for group in range(4):
        source=row+((fx.Int32(group*2)+(klane>>fx.Int32(1)))&fx.Int32(3))*fx.Int32(16)
        words=Vec.from_elements([
            fx.Int32(rocdl.ds_bpermute(T.i32,source*fx.Int32(4),raw[group//2*4+i]))
            for i in range(4)],fx.Int32) if not NATIVE_A else raw
        values=[]
        for h in range(2):
            word=words[group*2+h] if NATIVE_A else words[(klane&fx.Int32(1))*fx.Int32(2)+fx.Int32(h)]
            mag=word&fx.Int32(0x7f7f7f7f)
            zero=(~(mag+fx.Int32(0x7f7f7f7f)))&fx.Int32(0x80808080)
            values.append(word&~zero)
        af=fx.make_rmem_tensor(2,fx.Int32)
        af.store(Vec.from_elements(values,fx.Int32))
        cf=fx.make_rmem_tensor(4,fx.Float32)
        cf.store(Vec.filled(4,0.,fx.Float32))
        fx.gemm(atom,cf,af,bfrags[group],cf)
        partial=Vec(cf.load());out=[]
        for i in range(4):
            if fx.const_expr(EARLY_SCALE):
                multiplier=early[group][i]
            else:
                source_scale=klane*fx.Int32(4)+fx.Int32(i+group*16)
                packed=fx.Int32(rocdl.ds_bpermute(T.i32,source_scale*fx.Int32(4),scale_a))
                exponent=(packed>>fx.Int32(opsel_a*8))&fx.Int32(255)
                multiplier=((exponent+bexp[group]-fx.Int32(126))<<fx.Int32(23)).bitcast(fx.Float32)
            out.append(acc[i]+partial[i]*multiplier)
        acc=Vec.from_elements(out,fx.Float32)
    c.store(acc)
