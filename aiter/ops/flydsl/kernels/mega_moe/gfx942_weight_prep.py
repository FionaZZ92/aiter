# SPDX-License-Identifier: Apache-2.0
"""A8W4 fast-decoder nibble layout for static MegaMoEV2 weights.

Each output byte packs (K[j], K[j+4]); negative-zero nibbles become zero.
No FP8-expanded weight allocation or scale modification is performed.
This setup conversion must happen before graph capture.
"""
import torch

def prepare_fp4_fast(weight,chunk_words=2**20):
    raw=weight.contiguous().view(torch.uint8)
    if raw.numel()%4:raise ValueError('packed weight must contain whole dwords')
    source=raw.view(torch.int32).flatten()
    output=torch.empty_like(source)
    for start in range(0,source.numel(),chunk_words):
        words=source[start:start+chunk_words].to(torch.int64)&0xffffffff
        result=torch.zeros_like(words)
        for j in range(4):
            low=(words>>(4*j))&15
            high=(words>>(4*(j+4)))&15
            low=torch.where(low==8,0,low)
            high=torch.where(high==8,0,high)
            result|=(low|(high<<4))<<(8*j)
        output[start:start+chunk_words]=result.to(torch.int32)
    prepared=output.view(torch.uint8).reshape(raw.shape).view(weight.dtype)
    prepared.mega_fast_packed=True
    return prepared
