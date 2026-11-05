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
- `block_sparse_mlp.py:907-925` (`support_dec_moe`) requires `hidden_size % 512 == 0` and `intermediate_size_padded % 512 == 0` for the fused batch-1 decode (`exl3_dec_moe`, two launches for the whole routed sum). The model has 2048 and 768, so 768 fails **[I, from code; the path actually taken was not profiled]**. **Superseded 2026-11-04: the width is not the likely cause, the pack's codebook is; see the Evidence log.** Whether the Qwen3.8-Flash-Next-class packs (the MoE target of S6-04) have the same property is **[?]**: a comment in `block_sparse_mlp.py` mentions widths of 640-768 for them.
- This fork's snapshot (2026-10-02) predates upstream's ROCm kernels (2026-10-03 onward); upstream's MoE decode is `exl3_rocm_moe_decode` in `rocm/quant/exl3_gemv_multirow_rdna.cuh` **[I, see S6-08 evaluation]**.
- The server environment (`SERVE_ENV` in `tools/qwen/serve.py`) was tuned for the dense Qwen3.8 27B. For this model it sets `EXL3_MOE_FUSED=1`, `EXL3_MOE_CFG=2`, `EXL3_MOE_R8=1`, `EXL3_MOE_VALU=1`, `EXL3_MOE_WMMA_MIN_ROWS=9` and others that were never checked on this model **[V, effective env printed in the server log]**.
- The pack in `C:\exl3-probe\models\Qwen3-30B-A3B-exl3-4.0bpw` had no `chat_template.jinja`, which `serve.py` requires. One was extracted unchanged from its `tokenizer_config.json` on 2026-11-01 (delete it to undo).

## Where to look

- `exllamav3/modules/block_sparse_mlp.py`: `support_dec_moe` (~907), the grouped HIP path (`EXL3_HIP_GROUPED_MOE*`, ~2326), the batch-1 decode branch (~2164-2240), the union (verify) branch (~2261).
- `exllamav3/exllamav3_ext/quant/exl3_dec.cu`: `exl3_dec_moe`, `exl3_dec_moe_shared`, `exl3_dec_moe_union` (~2830-3300), `STRIP = 512`.
- Switches that select a MoE path or kernel: [env_vars.md](../../../env_vars.md), [knobs.md](../../../knobs.md) (`EXL3_DEC_MOE`, `EXL3_HIP_GROUPED_MOE`, `EXL3_HIP_GROUPED_MOE_MULTIROW`, `EXL3_HIP_GROUPED_MOE_PREFILL`, `EXL3_HIP_PREFILL_MIN_ROWS`, `EXL3_MOE_FUSED`, `EXL3_MOE_CFG`, `EXL3_MOE_WMMA*`, `EXL3_BLOCK_GRAPH_DEC_MOE`, `EXL3_DEC_MOE_UNION*`, tuning `EXL3_DEC_MOE_KTW_*`, `_MIN_BLOCKS_*`, `_KBS_B`).
- Upstream for comparison: `C:\exl3-probe\src\exllamav3` (the operator's probe install).

## Steps

**Added 2026-11-04 (do these before step 0, they decide whether steps 1-4 are needed):**

A. **Done.** Qwen3.5-35B-A3B (`mcg` codebook) through the same tree: fused MoE decode doubles to triples decode speed (Evidence log). The tree and Windows are not the problem.
B. **Confirm the codebook of the 30B pack.** Its `config.json` has no `codebook` key (quantiser 0.0.1), which means `3inst` **[I]**. Load log or `quantization_config` of the local pack; then check what the layers report (`multi.mul1`, `multi.mcg` in `block_sparse_mlp.py`).
C. **Requantise the 30B with `-cb mul1`** (`convert.py`; the original bf16 weights are about 60 GB), then run the same A/B as in the log. This is the direct test. Unknown **[?]**: whether `convert.py` runs on this Windows build (the quantiser kernels are built for ROCm, the converter was never run here), how long it takes, and whether it fits (Windows sees 31.6 GB of the 128 GB RAM).
D. **Optional second check on another architecture:** `turboderp/Laguna-XS-2.1-exl3` `4.00bpw` (`mul1`, hidden 2048, experts 512 wide, 256 experts, top 8).
E. **Only if C is not wanted:** a fast decode path for `3inst` MoE packs on ROCm. That is kernel work (upstream has one in its ROCm layer), sized like S6-08 Option A' (days).

Steps 0-4 below are the earlier plan and stay valid for a model that has a fast codebook and is still slow.

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

### 2026-11-04: Qwen3.5-35B-A3B on the same tree: fused MoE decode makes it 3x faster; the 30B's cause is probably its codebook, not its width

**Setup [V].** Tree `claude/modest-rubin-p69b5g` at `d0a0a45`, the installed default build, `serve` as in the install guide (plain decode, `--no-vision`, ctx 8192). Pack `turboderp/Qwen3.5-35B-A3B-exl3` branch `4.00bpw` (19.3 GB): hidden 2048, `moe_intermediate_size` 512, 256 experts, top 8, `codebook: mcg`, quantiser 0.0.22. `tools/bench.py` (a later copy with the UTF-8 decode fix, see the journal), 256 new tokens, greedy, prefill about 3.1K tokens, 3 repeats per cell (median). Two servers loaded side by side so the settings could alternate without restarts: A on port 8000 (default), B on port 8001 with `EXL3_DEC_MOE=0`. The operator left the machine alone only in part of the runs.

| Round | A default: prose / chat / code | B `EXL3_DEC_MOE=0`: prose / chat / code |
|---|---|---|
| 1 | 47.0 / 47.5 / 46.5 | 16.4 / 16.4 / 16.5 |
| 2 | 30.6 / 55.4 / 40.3 | 16.5 / 16.5 / 16.6 |
| 3 | 55.1 / 55.0 / 55.0 | 14.3 / 16.5 / 16.4 |

Decode tok/s. Round 4 was cut short by the operator. Median over the nine values: **A 47.5, B 16.5 (2.9x)**; best A 55.4 (3.3x), worst A 30.6 (1.8x); B never above 16.6. Prefill: A 1006-1016, B 988-995 tok/s as `tools/bench.py` printed it through `localhost`, **which carried a constant ~2 s connection delay (corrected 2026-11-05, below); the server's own timer gave 3000-3200 tok/s for the same requests, and `bench.py` through `127.0.0.1` gave 2862**. A and B differ by less than 2 % either way. Earlier, not interleaved, default-only runs of the same server: 53.7 / 53.7 / 52.6, then 30.1 / 29.7 / 29.9, then 47.1 / 46.9 / 46.7 (prose / chat / code).

**Variation.** A is noisy (30-55), B is steady. The operator reports that A depends on whether the machine is used (clicking, browsing) during a run; left alone it holds about 55 **[operator observation]**. The mechanism (the desktop sharing the GPU and memory with latency-sensitive fused kernels) is a guess **[?]**.

**Dispatch, from the code [V by reading `block_sparse_mlp.py`; nothing in it depends on the OS]:**

| Path | Needs |
|---|---|
| Fused batch-1 decode `exl3_dec_moe` | hidden and padded expert width multiples of 512; uniform `mul1` or `mcg`; K 2-6; no mixed-bitrate layer |
| Grouped decode `exl3_moe_gfx12_k3` | multiples of 128; top-k 6, 8 or 10; K in 1-6, one K per layer; **`mul1` and not `mcg`** |
| Anything else (`3inst`) | generic fallback; the fused kernel for `3inst` packs (`EXL3_MOE_FUSED_3INST`) is CUDA only |

The code comment on the grouped gate says the original gate was "2560 / 640-768 / top-10 ... the Qwen3.8-Flash-Next pack only", so 768 is inside the grouped path's range.

**Pack codebooks (`quantization_config` on the 4 bpw branches) [V]:** Qwen3-30B-A3B quantiser 0.0.1, no `codebook` key; Qwen3-Next-80B-A3B 0.0.6, no key; Qwen3.5-35B-A3B `mcg`; Laguna-XS-2.1 `mul1`; Qwen3.8-27B and Flash-Next `mul1`. A pack without the key predates the codebook option and is `3inst` **[I]**; the repo itself calls Qwen3-30B-A3B the `3INST` example (`block_sparse_mlp.py`, `_FUSED_3INST`).

**Conclusion, with its limits.**
- **[V]** On this Windows tree, MoE decode through the fused path reaches 47-55 tok/s on a 3 B-active model, inside the README's Linux range for Flash-Next (47-73). Without it the same model decodes at 16.5.
- **[I]** The 30B's 9.8 tok/s is the same kind of fallback speed, so the likely cause is that its `3inst` pack has no fast MoE path in this fork. The 768 width is probably not the cause (the grouped path accepts it) but a `mul1` copy is needed to say so.
- **Not proven:** the two models differ in layers, expert count and architecture, and the 9.8 was one run at 4K depth against these short-prompt runs. The decisive test is step C.
- **Not measured:** Laguna-XS-2.1, a `mul1`/`mcg` copy of the 30B, the same-session baseline of step 0, any Linux run.
- **Knob sweep on the 35B (2026-11-04):** ten MoE arms, report [qwen35-35b-a3b-4bpw-off.md](../../../benchmarks/sweeps/qwen35-35b-a3b-4bpw-off.md). The machine was not quiet (baseline noise +-16 % / +-10 % on decode). Only `EXL3_DEC_MOE=0` is a real effect (-60 %); no tuning arm beat the noise, so the defaults stay. Not worth repeating before a model with a fast codebook shows a gap.
- **Correction of what "new" means:** Qwen3-30B-A3B is the original Qwen3 generation (quantiser 0.0.1), older than Qwen3.5 / 3.6 / 3.8, which all run here. Its slowness is a property of the pack, not of newer models failing on Windows.
- **Prefill discrepancy resolved (2026-11-05): a Windows `localhost` delay, not the model.** The sweep reported 2950 / 2671 tok/s (4K / 16K, `cached` = 0 on every request), `tools/bench.py` through `localhost` reported 860-1017. The server's `qserve: request` lines for the bench requests gave `prefill_s` about 1.0 s for 3096 tokens (3100-3200 tok/s); the request log (`KYOJIN_REQLOG_DIR`) showed 0.872 s for a 2472-token request, of which 0.824 s prefill. The rest was on the client side: `localhost` resolves to `::1` first, the server listens on IPv4 only, and Windows takes about 2 s to refuse a connection to a closed local port. Same prompt, `bench.py --base http://127.0.0.1:8000/v1`: **prefill 2862 tok/s, decode 54.5 tok/s** (16-token outputs), against 860 and 30.5 through `localhost` **[V]**. Prefill of this 35B on the machine is therefore about 2900-3200 tok/s, not 1000. The 30B figures in this task (1330 / 995 tok/s) come from the sweep harness, which does not go through `localhost`, and are unaffected.
- **Side finding:** `tools/bench.py` printed a UnicodeDecodeError traceback on this machine (cp1250 helper output); fixed on the `pr/windows-core` branch, harmless to the numbers.

## Links

- [S6-08: single-row decode projection](08-decode-gemv-upstream-exl3-gemm.md), [plan-08](plan-08-decode-gemv.md) (dense work, the `PROD=2` and `kbs` findings, the open MoE gate)
- [S6-04: Flash-Next 85 GiB](04-flash-next-85gib.md) (the MoE target model class)
- [Switch sweeps](../../../benchmarks/sweeps/README.md), [journal](../../journal.md)
