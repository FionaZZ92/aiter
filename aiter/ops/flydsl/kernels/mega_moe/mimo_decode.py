# SPDX-License-Identifier: Apache-2.0
"""gfx942 MegaMoEV2 decode eligibility, independent of the gfx950 main path."""


def small_m_eligible(
    *,
    enabled,
    arch,
    num_cu,
    world_size,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    swiglu_limit,
):
    return bool(
        enabled
        and arch == "gfx942"
        and num_cu == 304
        and world_size == 8
        and model_dim == 4096
        and inter_dim == 2048
        and experts == 256
        and topk == 8
        and quant == "a8w4"
        and swiglu_limit == 0.0
    )
