# Qwen3.8-27B (EXL3 4.0 bpw) decode on gfx1151: closing the gap to upstream (2026-11-02)

Native Windows, Ryzen AI Max+ 395 (gfx1151), driver 32.0.32015.7029, 200 W. Kyojin at commit `bf627bd` (branch `claude/modest-rubin-p69b5g`). Method: the operator's `bench_http.py` (warm-up pass, exact-token prompts, 128 new tokens, 3 reps, one fresh server per arm, `--draft-policy off`); raw data in `logs/bench27b-final`. Upstream ExLlamaV3 and llama.cpp rows are from the operator's session of 2026-10-31 on the same machine (`wsl setup/docs/engine-comparison-20261031`, benchmarks.md section 17), **not re-run** here; Kyojin's prefill today is about 10 % above its 10-31 figures for reasons not identified, so treat cross-session ratios as +-3 %.

## Decode (tok/s, plain, no speculation)

| depth | Kyojin default | Kyojin + kbs table + fused gate/up | gain | upstream | llama.cpp (UD-Q4_K_M) | gap to upstream |
|---|---|---|---|---|---|---|
| 2K | 11.52 | **12.27** | +6.5 % | 13.7 | 12.8 | -15.9 % -> **-10.4 %** |
| 8K | 11.47 | **12.07** | +5.2 % | 13.3 | 12.5 | -13.8 % -> -9.3 % |
| 16K | 11.20 | **11.75** | +5.0 % | 12.9 | 12.2 | -13.2 % -> -8.9 % |
| 32K | 10.62 | **11.08** | +4.4 % | 12.2 | 11.4 | -13.0 % -> -9.2 % |

Short prompts (battery, mean of 2): prose 11.92 -> 12.46, chat 11.78 -> 12.29, code 11.79 -> 12.40.

Prefill is unchanged by the switches (default / final / upstream / llama.cpp, tok/s): 2K 477 / 482 / 354 / 200, 8K 528 / 528 / 381 / 207, 16K 511 / 512 / 375 / 199, 32K 473 / 473 / 354 / 177.

About a third of the decode gap to upstream is closed. Kyojin still leads upstream on prefill by 30-40 % and llama.cpp by 2.4-2.7 x; with speculation (not re-measured here) it was ahead of upstream's plain decode in the 10-31 comparison.

## What produces the gain (all off by default, env switches)

| Switch | Effect on decode | Check |
|---|---|---|
| `EXL3_DEC_KBS_TAB=1`, `EXL3_DEC_KBS_GU=25`, `EXL3_DEC_KBS_DN=8` | +3.4 / +2.1 / +3.2 / +3.1 % at 2K / 8K / 16K / 32K | plain == R-row bitwise on real weights, R = 1..32; table off bitwise equal to the old default |
| `EXL3_DEC_GU=1` (gate+up in one launch) | +3.0 % at 2K, +1.8 % at 16K on top of the table | bitwise equal to two launches (48 of 48 on real weights) |

Settings that were measured and **not** worth it: `EXL3_DEC_PROD=2` (-1.0 / -1.4 % in the shipped configuration, although -2 % kernel time alone), `EXL3_DEC_PF=1`, `EXL3_DEC_SB` 2 / 8, global `EXL3_DEC_KBS` / `EXL3_DEC_MIN_BLOCKS` (the table is the per-shape version), `EXL3_CUDA_GRAPH=1`, `EXL3_LAG1=1` (no effect on ROCm). The 24 `SERVE_ENV` switches reverted one by one: noise ([switch sweep](sweeps/qwen38-27b-4bpw-off.md)).

## Caveats

- **Plain != speculative is not exact in the default configuration either**: on one 4K prompt the reply depends on the draft depth (depth 1, 3 differ from plain, 7 equals plain) at a near-tie token. The table moves draft composition and so shows it on that prompt; on 8 other 2K prompts all four combinations (default / table x plain / mix) are identical. Culprit not located (not the dense MLP GEMVs). See the journal entry of 2026-11-02.
- The table covers only the 27B MLP shapes (mul1, 4 bpw): K 5120 <-> N 17408. Other models and bit widths use the old rule.
- Decisions that need the operator: enable the switches by default (they are off), and whether to chase the remaining 9-10 % (host-bound attention, inter-block glue, sampler path: about 2.8 + 2.2 + 2.4 ms/token in the operator's 2026-11-01 profiling).
