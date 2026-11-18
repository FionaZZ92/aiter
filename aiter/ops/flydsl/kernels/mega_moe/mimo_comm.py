# SPDX-License-Identifier: MIT
"""Small-M communication helpers, without changing the base AITER kernels."""

from ..communication_ops_utils import *  # noqa: F403
from ..communication_ops_utils import load_i64_acquire
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T


def hw_xcc_id():
    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)",
            "=s",
            has_side_effects=True,
        ).result
    )


def invalidate_l1():
    llvm.InlineAsmOp(
        None, [], "buffer_inv sc0\ns_waitcnt vmcnt(0)", "", has_side_effects=True
    )


def store_i64_global(addr_i64, val):
    ptr = fx.inttoptr(
        fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
        fx.Int64(addr_i64),
    )
    fx.ptr_store(fx.Int64(val), ptr)


@flyc.jit
def spin_until_shifted_eq_i64(addr_i64, val, *, shift):
    cur = fx.Int64(load_i64_acquire(addr_i64))
    while (cur >> fx.Int64(shift)) != fx.Int64(val):
        cur = fx.Int64(load_i64_acquire(addr_i64))
    return cur
