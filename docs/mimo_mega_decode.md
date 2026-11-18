# MiMo MegaMoEV2 decode review

Base SGLang: `2f6aebf22998e5a3193d7c72ffb9a0b69e4f8eaa`.
Base AITER: `108940c2b0aa6bc767b456f6f4c103046e651aef`.
Both local branches are `mimo_mega_decode`, with push remotes pointing at FionaZZ92. The paired SGLang/AITER changes use branch `mimo_mega_decode`.

The change adds opt-in `SGLANG_MIMO_MEGA_DECODE=1`. Eligible MiMo MXFP4 BF16 decode/idle batches bypass MORI dispatch and combine, using the fused MegaMoEV2 small-M path. Host global mode/bounds keep EP ranks on the same transport. Prefill, target verification, unsupported shapes, EPLB, two-batch overlap and multi-host layouts keep the baseline. Source capacity is512, geometry H4096/I2048/E256/topk8, EP8 on304-CU gfx942.

Native weight/scale layouts are prepared before fd2 rewrites the fallback weights. One runtime per EP group owns communication storage; layer-specific weight tables and all three kernel configurations are warmed before capture. This retains an additional native MXFP4 weight representation while enabled. The feature remains disabled by default because large-M performance is still worse than the baseline.

AITER adds the gfx942 small-M path and isolated communication helper extensions on FlyDSL0.3.2. The existing gfx950 main path, global dispatch helpers, dependency pin and register-P attention kernels are not replaced.

Source protocol provenance: ROCm/aiter PR7699 at `a602776099ba5cc2bf8b2cf1278d7291161aa6dd`, with the validated local gfx942 operand adaptations and FlyDSL0.3.2 compatibility helpers.

## New-base EP8 ABBA data

| M/rank | MORI EPv1 + A8W4 ms | MegaMoEV2 decode ms | baseline/current | latency reduction |
|---:|---:|---:|---:|---:|
| 12 | 0.6286 | 0.4042 | 1.555x | +35.7% |
| 24 | 0.6546 | 0.4381 | 1.494x | +33.1% |
| 48 | 0.6613 | 0.4177 | 1.583x | +36.8% |
| 80 | 0.5370 | 0.4999 | 1.074x | +6.9% |
| 96 | 0.6223 | 0.4999 | 1.245x | +19.7% |
| 192 | 0.6276 | 0.8566 | 0.733x | -36.5% |
| 256 | 0.8829 | 1.0660 | 0.828x | -20.7% |
| 304 | 1.1333 | 1.2585 | 0.901x | -11.0% |
| 418 | 1.1135 | 1.5262 | 0.730x | -37.1% |
| 512 | 1.3260 | 1.7781 | 0.746x | -34.1% |

Geometric mean over10 equally weighted M: 1.039x. Values below1 or negative latency reduction are regressions, not optimizations. Each implementation has18 pooled samples/M; each sample is25 replays of a graph containing8 full MoE calls, using max rank time. Inputs, routing and raw weight fingerprints match all ranks/runs. P10–P90 and raw samples are in `mimo_mega_decode_ep8.csv` (raw samples are retained in the local review artifact).

Both paths pass independent reference with max relative L2 0.04694153 (unchanged0.08 threshold). Different activation formats/quantization mean baseline and Mega outputs are not bitwise identical; max pair L2 is 0.06621487. Mega output is bitwise identical across its two runs. Prefill/verify routing, rank0 idle with7 decode peers, cross-layer binding and old graph replay after rebinding all pass.

Router outputs are fixed fixtures. This is a MoE operator test that exercises SGLang's actual post-loading adapter and MiMo entrypoint, not a whole-model/checkpoint serving throughput result. An earlier baseline setup had17–18% external VRAM and was quarantined under `abba_preliminary`; the final table uses qualified replacement runs only. GPU memory qualification is not proof that every brief external GPU operation was absent.

## Enable and reproduce

```bash
SGLANG_MIMO_MEGA_DECODE=1 python -m sglang.launch_server <normal MiMo MXFP4 EP8 MORI/AITER arguments>
python benchmark/kernels/bench_mimo_mega_decode_ep8.py --aiter-root <AITER checkout> --output-dir <results> --sizes 12,24,48,80,96,192,256,304,418,512
python benchmark/kernels/bench_mimo_mori_a8w4_ep8.py --aiter-root <AITER checkout> --output-dir <baseline> --sizes 12,24,48,80,96,192,256,304,418,512
```

The benchmark scripts must be launched through `torch.distributed.run --standalone --nproc_per_node=8`; defaults are8 calls/graph,25 replays and9 rounds. With the feature disabled, no native representation or shared Mega runtime is prepared.
