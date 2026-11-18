# SPDX-License-Identifier: Apache-2.0
"""Numerically validate the packed scaled-MFMA replacement on gfx942."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('AITER_MEGA_GFX942_UNPACK','reuse')
os.environ.setdefault('AITER_MEGA_GFX942_OPERANDS','native_ab_early')
import json
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels.mega_moe.gfx942_scaled_mma import scaled_mma_fp8_fp4,NATIVE_A,NATIVE_B,OPERAND_MODE,K32_SCOPE
from aiter.ops.flydsl.kernels.mega_moe.gfx942_operand_reuse import prepare_b_group,mma_native_group


@flyc.kernel
def kernel(A:fx.Tensor,B:fx.Tensor,SA:fx.Tensor,SB:fx.Tensor,O:fx.Tensor,
           OSA:fx.Constexpr[int],OSB:fx.Constexpr[int]):
    tid=fx.Int32(fx.thread_idx.x)
    av=fx.make_view(fx.add_offset(fx.get_iter(A),tid*fx.Int32(8)),fx.make_layout(8,1))
    bv=fx.make_view(fx.add_offset(fx.get_iter(B),tid*fx.Int32(4)),fx.make_layout(4,1))
    af=fx.make_rmem_tensor(8,fx.Int32)
    bf=fx.make_rmem_tensor(4,fx.Int32)
    cf=fx.make_rmem_tensor(4,fx.Float32)
    af.store(Vec(av.load()))
    bf.store(Vec(bv.load()))
    cf.store(Vec.filled(4,0.0,fx.Float32))
    sa=fx.Int32(fx.make_view(fx.add_offset(fx.get_iter(SA),tid),fx.make_layout(1,1)).load()[0])
    sb=fx.Int32(fx.make_view(fx.add_offset(fx.get_iter(SB),tid),fx.make_layout(1,1)).load()[0])
    if fx.const_expr(K32_SCOPE):
        for group in range(4):
            prepared=prepare_b_group(Vec(bf.load()),sb,OSB,group)
            cf.store(mma_native_group(Vec(cf.load()),Vec(af.load()),sa,OSA,prepared,group))
    else:
        scaled_mma_fp8_fp4(cf,af,bf,sa,sb,OSA,OSB)
    out=fx.make_view(fx.add_offset(fx.get_iter(O),tid*fx.Int32(4)),fx.make_layout(4,1))
    out.store(Vec(cf.load()))


@flyc.jit
def launch(A:fx.Tensor,B:fx.Tensor,SA:fx.Tensor,SB:fx.Tensor,O:fx.Tensor,
           OSA:fx.Constexpr[int],OSB:fx.Constexpr[int],stream:fx.Stream=fx.Stream(None)):
    kernel(A,B,SA,SB,O,OSA,OSB).launch(grid=(1,),block=(64,),stream=stream)


def main():
    torch.cuda.set_device(0)
    gen=torch.Generator().manual_seed(4726)
    a=torch.randn((16,128),generator=gen).to(torch.float8_e4m3fn)
    n=torch.randint(0,16,(16,128),generator=gen,dtype=torch.uint8)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
    b=lut[n.long()]
    ea=torch.randint(123,131,(16,4,4),generator=gen)
    eb=torch.randint(123,131,(16,4,4),generator=gen)
    # Operand lanes: row/column=lane%16, K32 group=lane//16.
    ap=a.view(torch.uint8).reshape(16,2,4,16).permute(2,0,1,3).contiguous().view(torch.int32).reshape(64,8).cuda()
    bp=(n[:,0::2]|(n[:,1::2]<<4)).reshape(16,4,16).permute(1,0,2).contiguous().view(torch.int32).reshape(64,4).cuda()
    if NATIVE_A:
        ap=a.view(torch.uint8).reshape(16,4,4,8).permute(2,0,1,3).contiguous().view(torch.int32).reshape(64,8).cuda()
    if NATIVE_B:
        bp=(n[:,0::2]|(n[:,1::2]<<4)).reshape(16,4,4,4).permute(2,0,1,3).contiguous().view(torch.int32).reshape(64,4).cuda()
    if os.environ.get('AITER_MEGA_GFX942_UNPACK','scalar') in ('fast','fastreuse'):
        from aiter.ops.flydsl.kernels.mega_moe.gfx942_weight_prep import prepare_fp4_fast
        bp=prepare_fp4_fast(bp.view(torch.uint8)).view(torch.int32)
    sa=sum(ea[:,:,j]<<(8*j) for j in range(4)).permute(1,0).contiguous().to(torch.int32).cuda()
    sb=sum(eb[:,:,j]<<(8*j) for j in range(4)).permute(1,0).contiguous().to(torch.int32).cuda()
    out=torch.empty((64,4),device='cuda',dtype=torch.float32)
    rows=[]
    for osa,osb in [(0,0),(1,2),(2,1),(3,3)]:
        launch(ap,bp,sa,sb,out,osa,osb)
        torch.cuda.synchronize()
        actual=out.cpu().reshape(4,16,4).permute(0,2,1).reshape(16,16)
        aa=a.float()*torch.pow(2.,ea[:,:,osa].float()-127).repeat_interleave(32,dim=1)
        bb=b*torch.pow(2.,eb[:,:,osb].float()-127).repeat_interleave(32,dim=1)
        ref=aa@bb.T
        err=float((actual-ref).norm()/ref.norm())
        maximum=float((actual-ref).abs().max())
        rows.append(dict(opsel_a=osa,opsel_b=osb,relative_l2=err,max_abs=maximum))
        assert torch.allclose(actual,ref,rtol=1e-5,atol=1e-4),(osa,osb,err,maximum)

    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
