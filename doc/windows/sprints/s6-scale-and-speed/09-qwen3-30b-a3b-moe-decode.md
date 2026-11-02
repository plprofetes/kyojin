# Qwen3-30B-A3B decode: 9.8 tok/s here against about 73 (upstream) and 86 (llama.cpp) on the same machine

- Status: todo
- Depends: S4-05
- Risk: MEDIUM: the cause is only inferred from code (the fused MoE decode kernel is not eligible for this model); the fix is either a kernel extension (expert width 768 is not a multiple of 512) or a port of upstream's ROCm MoE decode kernel, and either must keep plain decode and speculative verify bit-identical.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-09 · priority order 9 in the sprint

**Marking:** **[V]** measured on the target machine, **[I]** inferred from code or docs, **[?]** unknown. Machine: ASUS ProArt PX13, Ryzen AI Max+ 395 (gfx1151), Windows 11.

## Summary (what is being asked)

Qwen3-30B-A3B (EXL3 4.0 bpw, 128 experts, top-8, 48 layers, hidden 2048, `moe_intermediate_size` 768) decodes at **9.8 tok/s at 4K and 9.3 at 16K** in this fork, while upstream ExLlamaV3 on the same machine decodes it at about **73 tok/s** and llama.cpp (Vulkan, Q4_K_M) at about **86 tok/s**. Prefill is competitive (about 1330 tok/s at 4K, 995 at 16K here against 1431 at 2K and 1013 at 16K for upstream). The goal is to find why decode is about 7.5 times slower than upstream and to close most of the gap. This is the largest speed difference between this fork and upstream that has been measured so far (the 27B dense gap is about 12 %).

The operator's decision (2026-11-01): the 30B model is treated on its own, not as a side check of the 27B work.

## Evidence

| Engine, 30B-A3B 4 bpw | Decode tok/s | Prefill tok/s | Source |
|---|---|---|---|
| Upstream ExLlamaV3, KV fp16 | 72.9 median (min 53.7, max 75.1), tg128 | 1431 (2K), 1239 (8K), 1013 (16K), 755 (32K), 490 (64K) | `C:\exl3-probe\logs\bench-30b-kvfp16.json` **[V, earlier operator run; date and commit not recorded here]** |
| llama.cpp b9939 Vulkan, Q4_K_M, KV q8_0 | 86.0 (tg128), 85.0 (tg32) | 1067 (2K), 725 (8K), 522 (16K) | `C:\exl3-probe\logs\llama-anchor.log`, `wsl setup\docs\engine-comparison-20261031\anchor30b.log` **[V]** |
| This fork (`577eac3` plus the 2026-11-01 working tree, `PROD=2` build installed) | **9.83 (4K), 9.26 (16K)**, 256 new tokens | 1330 (4K), 995 (16K) | `tools/sweep.py` harness, journal 2026-11-01 **[V, one run, server env as shipped]** |

- Active parameters about 3.3 B at 4 bpw, so about 1.65 GB read per token: about 8 ms at 200 GB/s, a ceiling near 120 tok/s **[I]**. Upstream (about 14 ms/token) is near it, this fork (about 100 ms/token, about 2 ms per layer) is 12 times above it. A gap that size points to the wrong path or host overhead, not to arithmetic.
- `block_sparse_mlp.py:907-925` (`support_dec_moe`) requires `hidden_size % 512 == 0` and `intermediate_size_padded % 512 == 0` for the fused batch-1 decode (`exl3_dec_moe`, two launches for the whole routed sum). The model has 2048 and 768, so 768 fails **[I, from code; the path actually taken was not profiled]**. Whether the Qwen3.8-Flash-Next-class packs (the MoE target of S6-04) have the same property is **[?]**: a comment in `block_sparse_mlp.py` mentions widths of 640-768 for them.
- This fork's snapshot (2026-10-02) predates upstream's ROCm kernels (2026-10-03 onward); upstream's MoE decode is `exl3_rocm_moe_decode` in `rocm/quant/exl3_gemv_multirow_rdna.cuh` **[I, see S6-08 evaluation]**.
- The server environment (`SERVE_ENV` in `tools/qwen/serve.py`) was tuned for the dense Qwen3.8 27B. For this model it sets `EXL3_MOE_FUSED=1`, `EXL3_MOE_CFG=2`, `EXL3_MOE_R8=1`, `EXL3_MOE_VALU=1`, `EXL3_MOE_WMMA_MIN_ROWS=9` and others that were never checked on this model **[V, effective env printed in the server log]**.
- The pack in `C:\exl3-probe\models\Qwen3-30B-A3B-exl3-4.0bpw` had no `chat_template.jinja`, which `serve.py` requires. One was extracted unchanged from its `tokenizer_config.json` on 2026-11-01 (delete it to undo).

## Where to look

- `exllamav3/modules/block_sparse_mlp.py`: `support_dec_moe` (~907), the grouped HIP path (`EXL3_HIP_GROUPED_MOE*`, ~2326), the batch-1 decode branch (~2164-2240), the union (verify) branch (~2261).
- `exllamav3/exllamav3_ext/quant/exl3_dec.cu`: `exl3_dec_moe`, `exl3_dec_moe_shared`, `exl3_dec_moe_union` (~2830-3300), `STRIP = 512`.
- Switches that select a MoE path or kernel: [env_vars.md](../../../env_vars.md), [knobs.md](../../../knobs.md) (`EXL3_DEC_MOE`, `EXL3_HIP_GROUPED_MOE`, `EXL3_HIP_GROUPED_MOE_MULTIROW`, `EXL3_HIP_GROUPED_MOE_PREFILL`, `EXL3_HIP_PREFILL_MIN_ROWS`, `EXL3_MOE_FUSED`, `EXL3_MOE_CFG`, `EXL3_MOE_WMMA*`, `EXL3_BLOCK_GRAPH_DEC_MOE`, `EXL3_DEC_MOE_UNION*`, tuning `EXL3_DEC_MOE_KTW_*`, `_MIN_BLOCKS_*`, `_KBS_B`).
- Upstream for comparison: `C:\exl3-probe\src\exllamav3` (the operator's probe install).

## Steps

Each step is cheap and can stop the task. No new scripts without the operator's consent (`scripts/windows/CLAUDE.md`); `tools/sweep.py` and the operator's `bench_http.py` (`wsl setup\docs\engine-comparison-20261031`) already exist.

0. **Same-session baseline.** Kyojin, upstream (probe install) and llama.cpp (`wsl setup\windows-tests\llama-b9939-bin-win-vulkan-x64`) on the same prompts and the same method (`bench_http.py`, depths 2K, 8K, 16K, 32K), back to back, so drift is visible. Record versions and power state. Stop if upstream's 73 tok/s does not reproduce within 10 %.
1. **Path sweep, no code.** `tools/sweep.py run --model C:\exl3-probe\models\Qwen3-30B-A3B-exl3-4.0bpw` (no `--skip MOE`), plus an arms file for the path switches listed above and for the server's `SERVE_ENV` reverts. Greedy, 4K and 16K, 3-4 rounds. Report to `doc/benchmarks/sweeps/qwen3-30b-a3b-4bpw-off.md`. A switch that moves decode by more than 2x is the lead.
2. **Profile one decode token.** Which MoE path runs (`support_dec_moe` false, grouped HIP, or something slower), kernels and launches per layer, host gaps, `.cpu()` syncs. Compare time per layer with the bandwidth floor.
3. **Decide from the profile:**
   - host overhead or a bad default: fix it, no kernel work;
   - fused kernel not eligible: extend `exl3_dec_moe` (and `_shared`, `_union`) to expert widths that are multiples of 128 or 256 (the dense kernel already handles a partial last 512-column strip);
   - otherwise port upstream's MoE decode kernel as an isolated subtree, as S6-08 describes for the dense multirow GEMV (a 16-file, 5.5K-line include closure; est. 2-3 days).
4. **Gates for any kernel change:** bit-identity of plain decode and speculative verify (per row), the 27B dense path bitwise unchanged with the new code off or on, `PROD=2` default-vs-flag comparison on this model (this is the MoE half of the `EXL3_DEC_PROD=2` gate left open in S6-08: once the model is eligible for the fused kernel, it is the first MoE pack on disk that exercises it).

## Done when

- Plain decode on the 30B-A3B at 4K **>= 50 tok/s** (proposed first gate, about 70 % of upstream's measured 73) and the cause of the 9.8 tok/s written down with evidence. Stretch: >= 70.
- Prefill no worse than today (>= 1300 tok/s at 4K, >= 980 at 16K).
- Plain == speculative bitwise still holds; the 27B dense results are unchanged.
- The same-session table (step 0) is in the benchmark docs.

## If it fails

- The path sweep and profile show no cheap fix and the port is larger than estimated: stop, write the estimate, and ask the operator before starting a port.
- A change breaks plain == speculative: revert it, record the shape and rows.
- Upstream's number does not reproduce: record what the probe install actually does today and re-base the target on it.

## Evidence log

(append dated entries here)

## Links

- [S6-08: single-row decode projection](08-decode-gemv-upstream-exl3-gemm.md), [plan-08](plan-08-decode-gemv.md) (dense work, the `PROD=2` and `kbs` findings, the open MoE gate)
- [S6-04: Flash-Next 85 GiB](04-flash-next-85gib.md) (the MoE target model class)
- [Switch sweeps](../../../benchmarks/sweeps/README.md), [journal](../../journal.md)
