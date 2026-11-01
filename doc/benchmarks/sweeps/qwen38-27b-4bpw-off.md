# Switch sweep: qwen38-27b-4bpw, draft policy off

2026-11-01 14:43, commit `577eac30e5b0` (8 dirty paths), prompts ~4096 / ~16384 tokens, greedy, 256 new tokens, fresh server per run, 2 rounds, default build `EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD`. Baseline = the server's own SERVE_ENV. Baseline replies identical across runs.

| depth | baseline prefill tok/s | baseline decode tok/s | noise (half-range) prefill / decode |
|---|---|---|---|
| 4096 | 541.15 | 11.91 | ±0.2% / ±1.1% |
| 16384 | 517.5 | 11.42 | ±0.3% / ±0.8% |

Arms: % change of the median vs baseline (+ = faster); **bold** = outside the baseline noise and > 1 %; `same` = reply identical to the baseline.

| arm | override | runs | prefill 4096 | decode 4096 | same 4096 | prefill 16384 | decode 16384 | same 16384 |
|---|---|---|---|---|---|---|---|---|
| dec_off | `EXL3_DEC=0` | 2/2 | +0.4 | **-5.0** | **no** | +0.0 | **-4.6** | yes |
| build_dec_sb8 | build `EXL3_DEC_SB=8` | 2/2 | -0.8 | **-4.5** | yes | -0.7 | **-4.0** | yes |
| build_dec_prod2 | build `EXL3_DEC_PROD=2` | 2/2 | -0.0 | **+3.2** | yes | +0.0 | **+4.1** | yes |
| dec_min_blocks_192 | `EXL3_DEC_MIN_BLOCKS=192` | 2/2 | +0.1 | **+3.1** | **no** | +0.0 | **+3.5** | yes |
| dec_kbs_10 | `EXL3_DEC_KBS=10` | 2/2 | -0.1 | **+3.4** | **no** | +0.1 | **+3.4** | yes |
| dec_kbs_8 | `EXL3_DEC_KBS=8` | 2/2 | -0.1 | **+3.1** | **no** | -0.1 | **+2.8** | yes |
| dec_kbs_4 | `EXL3_DEC_KBS=4` | 2/2 | -0.5 | **-2.4** | **no** | +0.2 | -0.5 | yes |
| revert_EXL3_NC2_ASYNC_STASH | `EXL3_NC2_ASYNC_STASH=0` | 2/2 | +0.0 | **-1.7** | yes | -0.3 | -0.2 | yes |
| revert_EXL3_ROLL_CKPT | `EXL3_ROLL_CKPT=0` | 2/2 | -0.2 | **-1.7** | yes | +0.1 | -0.4 | yes |
| revert_EXL3_TOPK_PAR | `EXL3_TOPK_PAR=0` | 2/2 | -0.3 | **-1.7** | yes | +0.1 | -0.5 | yes |
| revert_EXL3_PF_SKIP | `EXL3_PF_SKIP=0` | 2/2 | +0.1 | **-1.4** | yes | +0.2 | -0.1 | yes |
| revert_EXL3_NC2_ATTN_F16 | `EXL3_NC2_ATTN_F16=0` | 2/2 | +0.1 | -0.8 | yes | +0.1 | **-1.3** | yes |
| revert_EXL3_PF_TAIL_MERGE | `EXL3_PF_TAIL_MERGE=0` | 2/2 | -0.1 | -0.8 | yes | -0.1 | **-1.2** | yes |
| revert_EXL3_TAIL_REPLY | `EXL3_TAIL_REPLY=0` | 2/2 | **+1.3** | **-1.2** | yes | +0.2 | -0.2 | yes |
| dec_min_blocks_16 | `EXL3_DEC_MIN_BLOCKS=16` | 2/2 | -0.0 | **-1.1** | yes | +0.3 | -0.0 | yes |
| revert_EXL3_GR_GS | `EXL3_GR_GS=0` | 2/2 | -0.4 | -1.1 | yes | -0.2 | -0.8 | yes |
| revert_EXL3_HOST_LEAN | `EXL3_HOST_LEAN=0` | 2/2 | -0.4 | -0.7 | yes | +0.1 | **-1.1** | yes |
| dec_qkv_unfused | `EXL3_DEC_QKV=0` | 2/2 | -0.1 | -1.0 | yes | +0.0 | -0.7 | yes |
| revert_EXL3_HOST_CUTS | `EXL3_HOST_CUTS=0` | 2/2 | -0.3 | -1.0 | yes | +0.0 | **-1.0** | yes |
| revert_EXL3_NC2_ABLIT_F16 | `EXL3_NC2_ABLIT_F16=0` | 2/2 | -0.0 | -1.0 | yes | +0.1 | +0.4 | yes |
| build_dec_sb2 | build `EXL3_DEC_SB=2` | 2/2 | +0.2 | +0.8 | yes | +0.2 | +1.0 | yes |
| build_dec_pf1 | build `EXL3_DEC_PF=1` | 2/2 | -0.1 | +0.3 | yes | +0.1 | +0.8 | yes |
| revert_EXL3_HIP_PREFILL_MIN_ROWS | `EXL3_HIP_PREFILL_MIN_ROWS=17` | 2/2 | +0.1 | -0.8 | yes | +0.1 | +0.0 | yes |
| revert_EXL3_PF_NO_TAIL | `EXL3_PF_NO_TAIL=0` | 2/2 | **+1.6** | +0.1 | yes | **-1.3** | -0.7 | yes |
| revert_EXL3_GR_HIP | `EXL3_GR_HIP=0` | 2/2 | -0.1 | -0.6 | yes | +0.1 | -0.7 | yes |
| build_dec_prod2_pf1 | build `EXL3_DEC_PROD=2 EXL3_DEC_PF=1` | 2/2 | -0.1 | -0.6 | yes | +0.2 | -0.3 | yes |
| revert_EXL3_GDN_HF | `EXL3_GDN_HF=0` | 2/2 | **-1.5** | +0.6 | yes | **-2.0** | +0.3 | yes |
| revert_EXL3_PF_GR_FUSE | `EXL3_PF_GR_FUSE=0` | 2/2 | -0.3 | -0.6 | yes | +0.1 | -0.5 | yes |
| revert_EXL3_GDN_FUSE | `EXL3_GDN_FUSE=0` | 2/2 | **+1.4** | -0.2 | yes | **-1.1** | -0.4 | yes |
| dec_min_blocks_64 | `EXL3_DEC_MIN_BLOCKS=64` | 2/2 | -0.0 | +0.4 | yes | +0.2 | -0.1 | yes |
| revert_EXL3_BRANCH_CKPT | `EXL3_BRANCH_CKPT=0` | 2/2 | -0.4 | +0.1 | yes | +0.1 | -0.4 | yes |
| revert_EXL3_TAIL_MAXGAP | `EXL3_TAIL_MAXGAP=1024` | 2/2 | +0.0 | +0.3 | yes | -0.0 | +0.3 | yes |
| revert_EXL3_DQ_HIP | `EXL3_DQ_HIP=0` | 2/2 | -0.1 | +0.3 | yes | +0.0 | -0.0 | yes |
| revert_EXL3_TAIL_DENSE | `EXL3_TAIL_DENSE=0` | 2/2 | **+1.8** | +0.3 | yes | +0.3 | +0.0 | yes |
| revert_EXL3_STATE_REBASE | `EXL3_STATE_REBASE=0` | 2/2 | +0.1 | -0.2 | yes | +0.1 | -0.2 | yes |
| revert_EXL3_REPLY_FIRST | `EXL3_REPLY_FIRST=0` | 2/2 | +0.1 | -0.2 | yes | -0.3 | -0.0 | yes |
| revert_EXL3_PLE_HIP | `EXL3_PLE_HIP=0` | 2/2 | -0.0 | +0.1 | yes | +0.2 | +0.1 | yes |

Raw: `results.jsonl` (reply texts included), `server/*.log`, `builds/*.log`, `sweep.log`.
