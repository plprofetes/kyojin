# Switch sweep: qwen35-35b-a3b-4bpw, draft policy off

2026-11-04 23:19, commit `d0a0a45f018a` (5 dirty paths), prompts ~4096 / ~16384 tokens, greedy, 256 new tokens, fresh server per run, 2 rounds, default build `EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD`. Baseline = the server's own SERVE_ENV. Baseline replies identical across runs.

| depth | baseline prefill tok/s | baseline decode tok/s | noise (half-range) prefill / decode |
|---|---|---|---|
| 4096 | 2950.05 | 38.519999999999996 | ±6.4% / ±15.7% |
| 16384 | 2671.45 | 35.91 | ±0.7% / ±9.7% |

Arms: % change of the median vs baseline (+ = faster); **bold** = outside the baseline noise and > 1 %; `same` = reply identical to the baseline.

| arm | override | runs | prefill 4096 | decode 4096 | same 4096 | prefill 16384 | decode 16384 | same 16384 |
|---|---|---|---|---|---|---|---|---|
| dec_moe_off | `EXL3_DEC_MOE=0` | 2/2 | **+6.7** | **-61.4** | **no** | +0.9 | **-59.4** | **no** |
| moe_cfg_0 | `EXL3_MOE_CFG=0` | 2/2 | +3.2 | -11.6 | yes | +0.6 | -8.3 | yes |
| moe_minblk_a_64 | `EXL3_DEC_MOE_MIN_BLOCKS_A=64` | 2/2 | -0.7 | +3.9 | yes | -0.1 | -0.2 | yes |
| moe_cfg_1 | `EXL3_MOE_CFG=1` | 2/2 | +0.3 | +1.3 | yes | +0.1 | +0.9 | yes |
| dec_moe_shared_fold_off | `EXL3_DEC_SHARED_FOLD=0` | 2/2 | -0.4 | +1.0 | yes | -0.1 | +0.9 | yes |
| moe_kbs_a_8 | `EXL3_DEC_MOE_KBS_A=8` | 2/2 | -0.4 | -0.2 | **no** | -0.3 | +0.9 | yes |
| moe_ktw_a_4 | `EXL3_DEC_MOE_KTW_A=4` | 2/2 | -1.3 | +0.1 | yes | -0.1 | -0.4 | yes |
| moe_ktw_a_16 | `EXL3_DEC_MOE_KTW_A=16` | 2/2 | -0.5 | -0.3 | yes | -0.0 | -0.4 | yes |
| moe_kbs_a_4 | `EXL3_DEC_MOE_KBS_A=4` | 2/2 | -0.8 | +0.4 | **no** | +0.1 | +0.2 | **no** |
| moe_minblk_a_16 | `EXL3_DEC_MOE_MIN_BLOCKS_A=16` | 2/2 | +0.4 | -0.3 | yes | +0.0 | +0.1 | yes |

Raw: `results.jsonl` (reply texts included), `server/*.log`, `builds/*.log`, `sweep.log`.

## Reading (added 2026-11-04, by hand)

- **The run was not quiet.** The baseline's own half-range is ±15.7 % (4K) and ±9.7 % (16K) on decode, against ±1 % in the 27B sweep. Any arm inside that band is not distinguishable from the baseline. Hands-off, the same model decodes about 55 tok/s with the default settings ([S6-09 Evidence log](../../windows/sprints/s6-scale-and-speed/09-qwen3-30b-a3b-moe-decode.md)); the baseline medians here are 38.5 and 35.9.
- **Only one arm is a real effect:** `EXL3_DEC_MOE=0` (fused MoE decode off) costs about 60 % of decode at both depths, which matches the side-by-side A/B (about 16.5 tok/s against 47-55). Its replies differ from the baseline (`same: no`): the fallback kernels are not bit-equal to the fused ones, as expected.
- **No tuning win found.** `EXL3_MOE_CFG=0` (-11.6 / -8.3 %) is inside the noise at both depths, and every other arm is within +-4 %. The shipped defaults of the `EXL3_DEC_MOE_*` knobs are fine for this shape (hidden 2048, expert width 512, 256 experts, top 8).
- **`EXL3_DEC_MOE_KBS_A=4` and `=8` change the reply** at one or both depths: the k-split changes the accumulation order. Even if one were faster, it would need the plain == speculative check first. None was faster.
- **Prefill figures differ from `tools/bench.py` through `localhost` (2950 here against about 1005) and that is resolved:** `cached` is 0 on every baseline request, so the prefix cache is not involved (the sweep script did also divide cached tokens in; fixed in `tools/sweep.py`, no effect on this run). The difference is a constant ~2 s that `bench.py` paid per request on Windows, because `localhost` resolves to `::1` first and Windows takes about 2 s to refuse the connection before the client falls back to IPv4. Through `127.0.0.1`, `bench.py` reads 2862 tok/s prefill for a 2472-token prompt, in line with this sweep and with the server's own timer (3100-3200 tok/s at ~3K tokens). The sweep's prefill column is correct.
