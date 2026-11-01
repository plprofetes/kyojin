# Single-row decode projection: adopt upstream's autotuned `exl3_gemm` path (or retune `exl3_dec`) on gfx1151

- Status: todo
- Depends: S4-05
- Risk: HIGH: the kernel set is excluded from the ROCm build (`build_config.py`), Kyojin's copies of those files have diverged from upstream, upstream's RDNA inner kernel lives in a `rocm/` directory that this tree does not have, the RDNA launch path uses a device barrier (co-residency deadlock risk, TDR), and plain == speculative bit-identity must survive. Expected gain is modest (about 2 ms of ~79 ms per token).

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-08 · priority order 8 in the sprint

**Marking:** **[V]** measured on the target machine, **[I]** inferred from code or docs, **[?]** unknown. Measurements: ASUS ProArt PX13, Ryzen AI Max+ 395 (gfx1151), Windows 11, 200 W charger (as stated by the operator; wattage is not readable from software), Adrenalin driver 32.0.32015.7029 (2026-10-18), 2026-10-31/11-01.

## Summary (what is being asked)

On Qwen3.8-27B (EXL3 4.00bpw) **plain decode is about 7 ms/token slower in this fork than in upstream ExLlamaV3 built for the same GPU** (12.7 vs 14.0 tok/s [V]). A large, identifiable part of that is the dense **gate/up projection (5120 -> 17408, 4-bit): 217.0 us here vs 201.6 us upstream, same weights [V]**; it is 128 calls/token, about +2 ms/token (about 2.6% of decode). The faster upstream kernel is the **autotuned `exl3_gemm`**, which this fork's ROCm build excludes. Either build that path for single-row decode and use it where it wins, or retune `exl3_dec` to the same speed. **Scope is single-row decode only; do not change the 2-144-row dispatch** (this fork is already faster than upstream there, see the non-regression table).

## Evidence

### E1. The decode gap exists and is not the server or `SERVE_ENV` [V]
Same `exl3_bench.py`-style generator loop run against each source tree, `C:\models\qwen38-27b-4bpw` (`turboderp/Qwen3.8-27B-exl3` @ 4.00bpw), greedy, depth ~0:

| setup | decode tok/s |
|---|---|
| upstream tree (torch 2.13.0+rocm10.0.0, HIP 7.15.26333) | 14.0-14.1 |
| this tree, library defaults | 12.6 |
| this tree + `SERVE_ENV` | 12.7 (and 484 vs 209 tok/s prefill at 2K: `SERVE_ENV` is the prefill lead) |
| this fork's server / upstream via a benchmark server / llama.cpp b9939 | 12.0-12.2 / 13.7 / 12.8 |

Pipelined replay of the real decode (event-free): `model.forward` 66.9 ms (upstream) vs 71.8 ms (here), +4.9 ms; blocks +2.55 ms, inter-block glue +2.2 ms.

### E2. The gate/up projection kernel [V]
Real gate/up weights (layers' `gate_proj`/`up_proj`, 5120 -> 17408, K = 4), 24 different layers round-robin so weights come from DRAM, 300 calls pipelined, one sync, median of 3 reps, microseconds per call (GB/s of weight bytes, 44.6 MB per matrix):

| path | upstream tree | this tree |
|---|---|---|
| **production** (`Linear.forward`) | **201.6 (221.3)** | **217.0 (205.6)** |
| `inner.dec_gemv` (`exl3_dec`) | n/a | 221.8 (201.1) |
| `inner.hip_gemv` / `ext.exl3_gemv` direct | 213.8 (208.7) | 232.3 / 230.7 (192.0 / 193.4) |

Outputs agree to fp16 rounding (this tree's kernels differ by <= 0.002 on |y| up to 3.3; upstream production vs its direct `exl3_gemv`: bit-identical). Practical bandwidth ceiling measured with torch-only probes: about 238 GB/s (streaming sum), so upstream is at about 93% of it on this shape and this fork at about 86%.

Whole projection path, every linear replayed per shape class and weighted by call count (ms/token): 72.5 (upstream) vs 75.0 (here), +2.5 ms; gate/up group 28.50 vs 30.85, `down_proj` (17408 -> 5120, x64) 14.78 vs 15.15, lm_head (5120 -> 248320, K = 6, x1) 4.20 vs 4.23 (225 vs 227 GB/s, equal), others within +-3% except 5120 -> 1024 (78.5 vs 49.4 us, here faster). This fork's per-call host cost is lower (about 41 vs 75 us).

### E3. It is not the driver or the runtime stack [V]
One driver for both stacks (so no relative driver effect). Torch-only probes (no ExLlamaV3 extension) on the two torch/HIP stacks (upstream: torch 2.13.0 / HIP 7.15.26333; this fork: torch 2.15.0a0+rocm10.1.0a20260822 / HIP 7.16.26332): device copy 211.9 vs 213.4 GB/s, fp16 `mv` (hipBLAS) 227.2 vs 229.4, streaming sum 238.2 vs 235.6. Within this fork's own stack its three kernels span 217-232 us, as large as the gap to upstream, so kernel choice explains it. **[?] Not separated:** compiler version (hipcc 7.15 vs 7.16) versus source for the `exl3_gemv` pair; they are different source files.

### E4. Why upstream is faster (source) [V for what the code does, [I] for the speed attribution]
Upstream `BC_LinearEXL3::run_gr` sends a single row to **`exl3_gemm_gr`** (the EXL3 GEMM kernel, with its autotuner and RDNA launch rules), not to the plain GEMV. In this fork the same sources exist on disk but `quant/exl3_gemm.cu`, `quant/coop_autotune.cu`, `quant/exl3_kernel_map.cu` and all of `comp_units/` are excluded from the ROCm build, so `LinearEXL3.forward` uses `dec_gemv` (`exl3_dec`) for one row.

### E5. Where `exl3_gemm` is and is not a win (do not regress this) [V]
Time to first token for a random prompt of N tokens, fresh each time, median of 5, first pass discarded (ms; plain loop, `max_new_tokens = 1`):

| rows | upstream | this fork | rows | upstream | this fork |
|---|---|---|---|---|---|
| 4 | 146 | 176 | 144 | 1947 | 911 |
| 8 | 182 | 205 | 145 | 1932 | 909 |
| 16 | 316 | 214 | 192 | 878 | 871 |
| 17 | 312 | 213 | 256 | 979 | 897 |
| 24 | 547 | 746 | 512 | 1911 | 1404 |
| 32 | 548 | 762 | 1024 | 2940 | 2595 |
| 64 | 958 | 674 | 2048 | 5129 | 4336 |
| 128 | 1739 | 715 | | | |

Upstream runs `exl3_gemm` for 1-144 rows and `reconstruct_hgemm` above 144 (`AUTO_RECONSTRUCT_THRESHOLD = 144`); this fork runs `dec_gemv` for 1 row, `hip_gemv` up to 16 rows, and `reconstruct_hgemm` otherwise. For 24-144 rows upstream's `exl3_gemm` scales roughly linearly (1.7-1.9 s at 128-144 rows) and loses clearly to this fork's path. This fork is slower than upstream at 4-8 rows (by 23-30 ms) and at 24-32 rows (by about 200 ms; see Follow-ups); those two bands are outside this task's scope. Above 144 rows both use `reconstruct_hgemm`, so this kernel is **not** a long-prefill kernel.

## Where to look

**Upstream** (`https://github.com/turboderp-org/exllamav3`, MIT). The tree measured is `a3bfbfd0` (2026-10-19) plus two Windows patches, local clone `C:\exl3-probe\src\exllamav3` branch `stage1-windows-rocm` (patches in the operator's `wsl setup` repo, `docs/exl3-stage0/patches/`). Upstream's own ROCm support landed in `da7983bc` ("ROCm preparatory work", 2026-10-03) and `0ca49f29` ("ROCm: Extension plumbing, compatibility macros, WMMA wrappers, Triton compile, emulated int8 wmma, etc.", 2026-10-04), **after** this fork's snapshot (root commit `c181df3`, 2026-10-02; its `libtorch/*.cpp` match upstream versions of 2026-07-21 to 2026-09-10).
- `exllamav3/exllamav3_ext/libtorch/linear.cpp:34-50`: `BC_LinearEXL3::run_gr`; line 38 `exl3_gemm_gr(...)` when `x.numel() == x.size(-1)`; identical in this fork (0 changed lines).
- `exllamav3/exllamav3_ext/quant/exl3_gemm.cu`: dispatch and autotune; RDNA arms around lines 296-360 (per-shape LDS budget `smem_max`, autotune key `^= 0x52444e4157313031` at 315, occupancy clamp `EXL3_RDNA_OCC_MAX` at 336-355; "a plain launch past true co-residency deadlocks the device barrier").
- `quant/exl3_gemm_kernel.cuh` (device barrier, `EXL3_GRID_SYNC`), `quant/coop_autotune.cu` (`CoopKernelAutotuner`), `quant/exl3_kernel_map.cu`, `quant/comp_units/exl3_comp_unit_4_*.cu` (K = 4; the lm_head needs K = 6 but is already equal).
- `exllamav3/exllamav3_ext/rocm/` (does not exist in this fork): `rocm/quant/exl3_gemm_inner_rdna.cuh` (the RDNA inner kernel), `rocm/compat.h`, `rocm/ptx_rocm.cuh`, WMMA wrappers.
- `exllamav3/util/cuda_flags.py`, `HIP_EXCLUDED_SOURCES`: the only sources upstream leaves out on HIP are `hgemm_f16acc.cu`, `exl3_gemv.cu` (replaced by `rocm/quant/exl3_gemv_rdna.cu`) and two M-tiled MoE instance patterns; `libtorch/`, `exl3_gemm.cu`, `coop_autotune.cu`, `exl3_kernel_map.cu` and `comp_units/` are built.

**This fork.**
- `exllamav3/exllamav3_ext/build_config.py:7-29`: `ROCM_EXCLUDE_DIRS` (includes `comp_units`), `ROCM_ALLOW_PREFIXES` (only the int8 GEMV and `quantize_tiles` instances), `ROCM_EXCLUDE_FILES` (includes `quant/exl3_gemm.cu`, `quant/coop_autotune.cu`, `quant/exl3_kernel_map.cu`, `quant/exl3_moe*.cu`, `libtorch/*.cpp`).
- `exllamav3/modules/quant/exl3.py`: `AUTO_RECONSTRUCT_THRESHOLD` (line 25), `forward` (469-510; `rows == 1` -> `dec_gemv` at 470), `dec_gemv` (511-516, 257 calls/token each with one `empty` and two `view`s), `hip_gemv` (564-578), `EXL3_DEC` flag (~235).
- `exllamav3/ext.py:174-215`: `BC_*` constructors return `None` and about 20 fused C++ ops fall back to PyTorch on HIP.
- Divergence of this fork's copies from upstream `a3bfbfd0` (changed lines, both directions): `quant/exl3_gemm.cu` 256 (892 vs 720 lines), `quant/exl3_kernel_map.cu` 196, `quant/exl3_gemm_inner.cuh` 358, `quant/exl3_gemm_kernel.cuh` 48, `quant/coop_autotune.cu` 34, `quant/exl3_gemm.cuh` 4. These files are present on disk but not compiled for ROCm.

**Reproduction.** Raw data, scripts and write-ups are in the operator's `wsl setup` repo (commits `69df0db`, `33c6bf3`, `4ea8063`): `kyojin_instrumentation_profiling.md` (steps 13-18: method, what is unreliable), `kyojin_performance_ideas.md`, and `docs/engine-comparison-20261031/profiling/` (`kgemv.py` = E2/E3 microbenchmark, `klinear.py` = per-shape sweep, `kbw.py` = stack control, `kreplay.py`/`kreplay2.py` = event-free replay, `ksmall.py` = E5, `kbench.py` = generator loop on either tree). **Measurement warnings:** `torch.cuda.Event.elapsed_time` under-counts chains of small kernels on this stack and `torch.profiler` has no GPU timeline; use wall-clock replay with one final `synchronize`. First use of each prompt-length class is slow (tuning); discard the first pass.

## Proposal

- **Option A (second step, larger; see Effort):** build `quant/exl3_gemm.cu`, `quant/coop_autotune.cu`, `quant/exl3_kernel_map.cu` and the `comp_units` for K = 4 (K = 6 optional) for gfx1151, together with whatever of upstream's `rocm/` layer the RDNA inner kernel needs; route **`rows == 1`** projections through the `exl3_gemm_gr`-equivalent behind a switch (default off until A/B), keep `dec_gemv_r` and every other row count as is.
- **Option B (first step, cheapest; evidence below):** retune `exl3_dec` per shape (a table or a start-up autotune) for 5120 -> 17408, K = 4 (tile and split-k sweep, lane count, the `EXL3_DEC_PF` prefetch depth) until the microbenchmark matches 201.6 us, without new sources.
- **Option C (measurement only):** compile one source file (for example `exl3_gemv.cu`) with both toolchains to separate compiler version from source.

## Option B evidence (2026-11-01) [V]

`exl3_dec.cu` picks the dense k-tile width in `pick_ktw` as the largest that still leaves `min_blocks` (default 32) blocks (`WPB` = 8 waves per block, `KTW_MAX` = 32); the comment above `kbs_of` says it was measured at **MiMo shapes** (`tools/decode/sweep.py`). The knobs `EXL3_DEC_MIN_BLOCKS`, `EXL3_DEC_KTW`, `EXL3_DEC_KBS` are read with `getenv` on every call, so they can be swept without a rebuild. Gate/up (5120 -> 17408, K = 4) on the real weights, interleaved fresh processes, n = 4, production path, microseconds per call:

| setting | median | range |
|---|---|---|
| this fork, default | 218.0 | 216.7-221.2 |
| this fork, `EXL3_DEC_MIN_BLOCKS=192` | 209.3 | 208.8-209.9 |
| upstream (autotuned `exl3_gemm`) | 201.9 | 201.4-203.2 |

One variable closes 54% of the gap. It is **global**, though: per-shape sweep with 192 (us/call, default -> 192): 5120x17408 239.8 -> 234.4 (-2.2%), 5120x10240 144.7 -> 157.0 (+8.5%), 5120x12288 167.5 -> 177.8 (+6.1%), 6144x5120 98.5 -> 102.7 (+4.3%), 5120x6144 99.2 -> 102.4 (+3.2%), 5120x1024 49.1 -> 55.8 (+13.5%); weighted sum 74.99 -> 75.32 ms/token, end-to-end decode within noise. So the fix is a **per-shape** choice, which is also what upstream's autotuner does. Keep the R-row kernels consistent: they call the same `pick_ktw`/`kbs_of` so batch-1 and verify keep the same accumulation order (plain == speculative). Noise: the same configuration varied 216.7-227.0 us across fresh processes; always interleave, n >= 4.

## Effort (static analysis, 2026-11-01) [I]

- Upstream's hand-written `rocm/` layer is 27 files / 9,252 lines. The single-row `exl3_gemm` path needs **6 of them (1,889 lines)**: `rocm/rdna_wmma.cuh` 740, `rocm/quant/exl3_gemm_inner_rdna.cuh` 660, `rocm/rdna_wmma_emu.cuh` 128, `rocm/quant/exl3_mgemv_rdna.cuh` 138, `rocm/quant/exl3_gemv_multirow_rdna.cuh` 111, `rocm/ptx_rocm.cuh` 112, plus shared `arch.cuh` (38) and `ptx_portable.cuh` (66); none exist in this tree. (Include closure counted regardless of `#if`, so an upper bound.)
- The shared closure is 35 files / 6,354 lines; 20 differ from this fork's copies (~1,850 changed lines). This fork's `quant/exl3_gemm.cu`, `quant/exl3_gemm_kernel.cuh`, `quant/exl3_kernel_map.cu` and `ptx.cuh` contain **no ROCm guards** (upstream: 7, 5, 1, 28), i.e. they are pre-ROCm CUDA versions: a port replaces them.
- `ptx.cuh` is included by 41 of this fork's sources and `util.cuh` by 92, so do not replace shared headers in place; vendor the closure as an isolated subtree (about 8.3k lines copied verbatim, 25-30 files) with a distinct entry point, and route only `rows == 1`.
- Translation units: `exl3_gemm.cu`, `coop_autotune.cu`, `exl3_kernel_map.cu` and the three K = 4 comp units; the kernel map references every K, so trim it to K = 4 or compile all 24 + 3 comp units.
- **Option A: about 3-5 working days** (vendoring and glue 1; compile/link debugging 1-2 at about 12 min per full build; validation 1 including plain == speculative, the TTFT table and a TDR soak). **Option B: about 1-2 days** (per-shape sweep over the nine shape classes, a small table or start-up autotune in `pick_ktw`, one rebuild, same validation). Do B first; do A only if B leaves a gap worth the extra days.

## Previous handoff: evaluate porting the autotuner (operator instruction, 2026-11-01; done, see Evaluation below)

The operator asked that the next session in this repo **evaluate porting upstream's autotuner at least** (written "autoevaluator"; read as upstream's `CoopKernelAutotuner`, the per-shape launch-config selection in `quant/coop_autotune.cu` and `quant/exl3_gemm.cu`). The evaluation is the deliverable: a written go/no-go in the journal and in this task's Evidence. Implement only on a go, and ask before adding any new script, tool or subcommand (`scripts/windows/CLAUDE.md`).

**Context to keep in mind: this fork's priority is MoE models** (operator, 2026-11-01), so the dense decode GEMV path is a niche here, but not an irrelevant one. `README.strix-halo.md` lines 87-95 give the byte budget of a Flash-Next-class forward as routed experts 47%, **dense EXL3 GEMVs 40%**, mixers 13%, with dense GEMVs at 55-65% of the 236 GB/s peak and grouped expert GEMVs at 131-135 GB/s **[V, quote]**. The dense tiling heuristic (`pick_ktw`) was tuned at MiMo shapes, so the shapes of other models (Qwen3.8-27B, others) are untuned. The "about 135 GB/s" is the small-kernel rate measured by `tools/strix_halo/read_bw_vs_size.py`, not a platform ceiling: that tool reproduces 104-148 GB/s at 4-64 MiB on this Windows machine, and the 236 GB/s ceiling is for big copies **[V]**. For small kernels the lever is bytes per launch (each launch pays a fixed ramp-up cost).

**What to find out** (each answer goes in the journal):
1. What upstream's autotuner does: candidate set (GEMM shape index x `num_sms` x concurrency), key (`gemm_autotune_hash`, RDNA variant XORed at `exl3_gemm.cu:315`), measurement (CUDA event pair around `repeats` launches, `coop_autotune.cu` about lines 415-440, explicitly "timing candidates with warm cache" about line 372), launch (cooperative launch with a device barrier; `EXL3_DEBUG_AUTOTUNE` exists in upstream only, this fork's copy is the older version with 34 changed lines), and where the results persist **[V from reading; persistence not confirmed]**.
2. Whether it applies to `exl3_dec`: that kernel's tiling knobs are `ktw`, `kbs` and `min_blocks` (`pick_ktw`, `kbs_of`) and it is not a cooperative GEMM kernel, so most likely the **pattern** (key, candidates, timing, cache) is reusable and the code is not **[I]**.
3. Whether its timing method is valid on this stack: event pairs around 100-300 us kernels must be validated (on this machine `Event.elapsed_time` is right for one long GPU queue but under-counts chains of small kernels), and warm-cache bias (tuning one matrix is MALL-warm, 32 MB; production cycles over layers, which is why the operator's replay round-robins over 24 layers).
4. Where per-shape results would be stored: this fork already has `exllamav3/util/tune_cache.py` (`signature`, `read_records`, `write_records`, `merge_into`), `exllamav3/model/dense_warmup.py` and `exllamav3/model/dense_gemm_tune_seed.txt`.
5. Which shapes and candidates: for Qwen3.8-27B nine classes (E2 and "Option B evidence" above); for the MoE priority models read the dense projection shapes from their configs, plus the expert knobs (`EXL3_DEC_MOE_KBS_A/B`, `_KTW_A/B`, `_MIN_BLOCKS_A/B`).
6. Bit-identity: the R-row kernels use the same `pick_ktw`/`kbs_of`, so the tuner has to feed both; plain == speculative must still hold.
7. Cost: tuning time per shape, the start-up budget (dense GEMM warm-up is already about 90 s), persistence across restarts.
8. A go/no-go with an effort estimate, compared with Option A (about 3-5 days) and Option B (about 1-2 days).

**First cheap steps (no code):**
- Run the Qwen3-30B-A3B EXL3 pack that is already on the operator's disk (`C:\exl3-probe\models\Qwen3-30B-A3B-exl3-4.0bpw`, 15.98 GB) through this fork's tree as the MoE sanity check. Stage 1 measured upstream EXL3 at 72.9 tok/s decode vs llama.cpp 86.0 on it. MoE kernels are listed as untested on Windows in [deferred.md](../../deferred.md), so expect build or JIT risk.
- Prototype per-shape `EXL3_DEC_MIN_BLOCKS` from Python before each `dec_gemv` call (`env_int` calls `getenv` each time) to measure the end-to-end gain before touching C++ **[I: assumes `os.environ` writes reach the extension's CRT]**.

**Measurement rules learned the hard way:** interleave configurations in fresh processes, n >= 4, and report medians and ranges (the same configuration varied 216.7-227.0 us across processes); discard the first pass (first-use tuning); validate a timing method on a known workload first (`torch.profiler` has no GPU timeline here and `Event.elapsed_time` under-counts small-kernel chains); use wall-clock replay with one final `synchronize`; compute roofline fractions from the bytes actually read per token (about 13.6 GB for the 27B, not the 16.9 GB pack); fp16 KV unless stated; state the power state.

**Where the evidence and tools are** (operator's `wsl setup` repo, local, commits `69df0db`, `33c6bf3`, `4ea8063`, `5dde33f`, `d5ef67c`): `kyojin_performance_ideas.md` (ideas 18-30 and "Next session"), `kyojin_instrumentation_profiling.md` (steps 13-22), `benchmarks.md` section 17, and `docs/engine-comparison-20261031/profiling/` (`kgemv.py`, `krepeat.py`, `ksweep.py`, `kverify.py`, `klinear.py`, `kreplay2.py`, `kfloor.py`, `kcachecheck.py`, `closure.py`, results).

## Evaluation: porting the autotuner (2026-11-01, cloud session, static reading only, no GPU)

**Execution plan for a local GPU agent:** [plan-08-decode-gemv.md](plan-08-decode-gemv.md) (phases, code locations, bit-identity rules, DoD per phase).

**Verdict: no-go on porting `CoopKernelAutotuner`. It does not run on upstream's single-row ROCm path, so porting it cannot recover the gap.** E4's premise is corrected below. The upstream kernel that produced the 201.6 us is the RDNA **multirow GEMV with a row tile of 1** (`rocm/quant/exl3_gemv_multirow_rdna.cu`). Porting that kernel, or reproducing its structure in `exl3_dec`, is the way to Option A.

**Recent changes (regression check).** The fork has 50 commits since its squashed root (`1d0ad27`). The only non-doc changes are Windows setup and serve scripts, the constraints file, a `clang_rt.builtins` link argument in `setup.py`, the Windows lookup in `hip_lib.py`/`hip_compiler.py` (JIT DLL lookup only) and `tools/startup_health.py`. Nothing touches `exllamav3_ext/`, `modules/` or `model/` **[V, git log]**. So the 12.7 vs 14.0 tok/s gap is not a regression in this tree. It is structural: this fork's snapshot (2026-10-02) predates upstream's ROCm kernels (2026-10-03 onward).

**What upstream actually runs for one row on gfx1151** (`a3bfbfd0`, read 2026-11-01) **[V from code]:**
- `BC_LinearEXL3::run_gr` -> `exl3_gemm_gr` (`quant/exl3_gemm.cu:118`). Under `USE_ROCM` it first calls `exl3_gemv_multirow_try_launch` for `1 <= m <= 8` (lines ~203-217, "ahead of every other path"). It reaches the autotuner (~line 312) only if that call declines.
- The call accepts m = 1 when `n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES` (2048) and `EXL3_GEMV_MR_M1` is on (default). Gate/up has `n_tiles` = 1088, so it is taken. Upstream's own comment on the choice: *"the fused kernels' LDS prologue costs occupancy, which outweighs the saved launch"*. This matches E2, where upstream production (201.6 us) beat its own direct fused `exl3_gemv` (213.8 us) while giving **bit-identical** output.
- Gate/up launch geometry: (1) `exl3_gemv_mr_had_in_single` rotates x once into a global `A_had` buffer, which then stays in L2. (2) `exl3_gemv_mr_dot_single` runs with `warps = exl3_gemv_splitk_warps(320, 1088, 1)` = **4** (blocks >= 1024), `tpb = 2` N-tiles per wave and a grid of **544 blocks x 128 threads**. Split-K happens **inside the block only**, with each wave walking 80 k-tiles. It needs no global partials and no counters for k. A fused epilogue (last-arriving warp per 128-column segment) does the output rotation.
- Every choice comes from fixed rules (`exl3_gemv_splitk_warps`, `exl3_gemv_tiles_tpb`, `EXL3_GEMV_*` env overrides). Nothing is timed at run time.

**This fork's `exl3_dec` on the same shape** (K = 4, kt = 320, 34 strips of 512 columns) **[V from code]:**
- `pick_ktw` gives ktw = 8 and `kbs_of` gives kbs = 5, so the grid is **170 blocks x 256 threads**. Each block rotates its 1,024 inputs into LDS (the prologue, repeated across the 34 strips). Each wave streams only 8 k-tiles, then does a block reduction. A **cross-block split-K over 5 blocks** writes 5 x 17408 fp32 partials to scratch, and the last-arriving block per strip sums them through an atomic counter.
- `EXL3_DEC_MIN_BLOCKS=192` only switches ktw from 8 to 4 (kbs 10, 340 blocks), because `pick_ktw` admits only power-of-two ktw dividing kt/8. The kernel already supports a **ragged** split (`k_split`), so `EXL3_DEC_KBS` = any value from 2 to 40 is legal for the 1-row kernel. The 1-row sweep in "Option B evidence" therefore covered only 2 of 39 grid sizes.
- Dense MLP gate and up run as **two** `dec_gemv` launches per layer (`mlp.py`: `multi_gu` is `None` on ROCm because `exl3_mgemm` is not built). Attention q/k/v and GDN already use `exl3_dec_gemv_multi`.

**Answers to the eight questions:**
1. *What the autotuner does:* for each GEMM shape key it times compatible cooperative-GEMM tile shapes x grid sizes with event pairs, then caches the winner in process memory. That only matters when m > 8 or when the multirow path declines. **It never sees a decode GEMV on ROCm.**
2. *Applies to `exl3_dec`?* Only as a pattern (key -> candidate set -> timing -> cache). The candidates would be `kbs` (2-40 ragged) and `ktw`. A static per-shape table is simpler, because shapes are known at load time and the tune cache already exists.
3. *Timing validity:* not tested here (no GPU). Earlier sessions showed `Event.elapsed_time` under-counts chains of small kernels on this stack. A runtime tuner would need wall-clock timing over several layers' weights to avoid MALL-warm bias. That is one more reason to tune offline and store a table.
4. *Storage:* `exllamav3/util/tune_cache.py` (keyed per device signature) or a seed file next to `dense_gemm_tune_seed.txt`.
5. *Shapes:* the nine Qwen3.8-27B classes from E2, plus the dense shapes of the MoE-priority models (attention, shared experts, lm_head).
6. *Bit-identity:* **hazard found.** `dec_gemv_r_impl` (verify) sums `strips_total` over every matrix in the launch, with no `strips_hint`. `mlp.py`'s `VERIFY_FUSE` branch fuses gate+up (2 x strips), while plain decode calls gate and up separately (1 x strips). Any `min_blocks` threshold that falls between those two counts gives plain and verify different ktw, hence different rounding. Examples: the global `EXL3_DEC_MIN_BLOCKS=192` on the 27B gate/up (plain 170 < 192 -> ktw 4, fused verify 340 -> ktw 8), and, at the default 32, a 2048 -> 8192 MLP (Llama-3.2-1B class: plain ktw 8, fused verify ktw 16) **[I, from code; `EXL3_VERIFY_FUSE` is opt-in, default off]**. Any per-shape table must be keyed per matrix (K, N, bits), not per launch. Alternatively, `dec_gemv_r_impl` should take the same `strips_hint` that `dec_gemv_impl` already has.
7. *Cost:* the cooperative autotuner adds startup time and complexity and buys nothing on m = 1. A static table costs nothing at run time.
8. *Go/no-go:* **no-go on the autotuner port. Go on the steps below, in order.**

**Recommended path** (each step is cheap to measure and cheap to abandon):
1. **No-rebuild sweep (about 0.5 day).** For each dense shape class, sweep `EXL3_DEC_KBS` over the ragged values (for example 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20), plus `EXL3_DEC_KTW` where it is valid, with the E2 method (24-layer round-robin, fresh interleaved processes, n >= 4). Bound: keep `ceil(40 / kbs) <= 16` (`MOE_R_KTW_MAX`) whenever the R-row verify kernel is in use, because its LDS row budget is sized from `pick_ktw`'s ktw, not from the ragged one **[I]**.
2. **Attribution with the existing ablation build** (`QWG_ABLATE=1`, env `EXL3_QWG_DABL`: bit 1 = no decode, bit 2 = no DRAM stream, bit 4 = no epilogue). This shows whether gate/up loses its 15 us in the DRAM stream (grid and occupancy), the trellis decode, or the split-K epilogue. Then try `EXL3_DEC_PF=1` (one-tile prefetch, documented bit-identical; compile-time flag, one rebuild).
3. **Fuse plain-decode gate+up** into one `exl3_dec_gemv_multi` launch (2 jobs sharing x). That is one launch instead of two, 64 launches/token fewer on the 27B, and the same grid shape as the fused verify path. Bit-identical to today only while the per-matrix ktw is unchanged: for the 27B, ktw = 8 either way at the default `min_blocks`. Pass `strips_hint` so ktw stays the per-matrix value.
4. **Per-shape table** from step 1 in `pick_ktw`/`kbs_of`, keyed (K, N, bits). It feeds both the 1-row and R-row kernels (question 6), so plain == speculative holds.
5. **Only if a gap above about 5 us/call remains: Option A' = port upstream's multirow GEMV.** This replaces Option A as written above. The include closure of `rocm/quant/exl3_gemv_multirow_rdna.cu` is **16 files / 5,498 lines**: 8 absent here (`rocm/quant/exl3_gemv_{multirow_rdna.cu,multirow_rdna.cuh,kernel_rdna.cuh,tiles_rdna.cuh}`, `exl3_mgemv_rdna.cuh`, `rocm/ptx_rocm.cuh`, `ptx_portable.cuh`, `arch.cuh`), 6 diverged (`ptx.cuh`, `exl3_dq.cuh`, `codebook.cuh`, `util.cuh`, `graph.cuh`, `compat.cuh`), 2 identical. It also needs about 4 host helpers from `rocm/quant/exl3_gemv_rdna.cu` (`exl3_gemv_splitk_warps`, `_core_mode`, `_tiles_tpb`, `_fuse_out_enabled`). It needs **no cooperative launch, no device barrier and no autotuner**, so the TDR/co-residency risk listed above does not apply. Vendor it as an isolated subtree, as the Effort section says, and route m = 1 behind a switch. Its m = 2..8 form is bit-identical per row to its own m = 1 form, so routing verify rows 2-8 through it as well keeps plain == speculative. It is also upstream's faster path in E5's 4-8-row band. **Estimate: 2-3 days**, versus 3-5 days for the cooperative-GEMM port: no comp units, no kernel map, no `rdna_wmma`.

**Priority note (MoE).** Steps 1-4 apply to every dense decode GEMV, which is 40% of the bytes in a Flash-Next-class forward. The routed-expert kernels (`exl3_dec` MoE A/B, `EXL3_DEC_MOE_*` knobs) use the same `pick_ktw`/`kbs_of` and the same ragged split, so the step 1 sweep method transfers directly. Upstream's equivalent there is `exl3_rocm_moe_decode` (`exl3_gemv_multirow_rdna.cuh`), worth the same structural comparison before any MoE kernel work.

## Next session: HANDOFF to the local GPU agent (2026-11-01)

You are on the target machine (Windows 11, gfx1151, `.venv-win` built), on branch `claude/modest-rubin-p69b5g`. The evaluation above and the plan were written without a GPU. **Every performance number in them is from earlier operator runs or is [I].** Your job is to turn the plan's first phase into measurements and a decision.

**Update (2026-11-01, later): `tools/sweep.py` automates step 2 below and much more** (every `SERVE_ENV` switch, the `EXL3_DEC_*` runtime switches and the five build flags, on 4K and 16K prompts, unattended, resumable). Run it first; see [doc/benchmarks/sweeps/README.md](../../../benchmarks/sweeps/README.md). Steps 3-5 below (bit-identity gate, occupancy, per-shape time) remain manual.

**Read first (in this order, ~20 min):**
1. [plan-08-decode-gemv.md](plan-08-decode-gemv.md): §0 (corrected premise, kernel comparison, the four invariants, measurement rules), §6 (Path A, the preferred route), §1 (decision tree), §2 (Phase 1).
2. The "Evaluation" section above (the regression check and the bit-identity hazard).
3. `quant/exl3_dec.cu`: `lane_gemv` (~378), `pair_dot`/`mul1_pair` (~115-300), `gemv_kernel` (~784), `block_reduce` (~633), `pick_ktw`/`kbs_of` (~2013), `dec_gemv_impl` (~2266), `dec_gemv_r_impl` (~2637).

**What is settled (do not redo):**
- Porting `CoopKernelAutotuner`: no-go. Upstream's 1-row ROCm path never reaches it [V, source].
- No code regression in this fork [V, git log]; the gap is structural.
- A global `EXL3_DEC_MIN_BLOCKS` is not the fix (operator, [V]).

**This session's scope: plan Phase 1 plus Path A's flag builds. Nothing else.**

1. **Baseline.** Reproduce E2 (gate/up us/call, production path) and the generator-loop tok/s with the default build. Record versions, driver, power state, and the repo commit plus dirty-path count (the `kyojin_win.py` run folder records both).
2. **Flag builds.** Bit-identical by design, one rebuild each (~12 min). `EXL3_HIP_DEFINES` **replaces** the default `EXL3_HIP_STG_PAD` (`kyojin_win.py:64`, `setup.py:70`), so always keep it in the list. Use `--clean`, because the Windows build does not track header or define changes reliably ([install.md](../../install.md)):

   ```powershell
   $env:EXL3_HIP_DEFINES = "EXL3_HIP_STG_PAD EXL3_DEC_PROD=2"                  # A1: full-rate hash multiply (mul1 only)
   uv run --no-project python scripts/windows/kyojin_win.py build --clean
   $env:EXL3_HIP_DEFINES = "EXL3_HIP_STG_PAD EXL3_DEC_PF=1"                    # A6: one-tile prefetch
   $env:EXL3_HIP_DEFINES = "EXL3_HIP_STG_PAD EXL3_DEC_PROD=2 EXL3_DEC_PF=1"    # both
   $env:EXL3_HIP_DEFINES = "EXL3_HIP_STG_PAD QWG_ABLATE=1"                     # attribution only: garbage outputs; run with EXL3_QWG_DABL=0,1,2,3,4
   Remove-Item Env:EXL3_HIP_DEFINES                                           # back to the default before you finish
   ```

   - If PF = 1 or PROD = 2 helps, also try `EXL3_DEC_SB` in {2, 8} with the winner (default 4).
   - First find out which codebook the 27B and the Qwen3-30B-A3B packs use (mcg or mul1). A1 does nothing for mcg **[I]**.
3. **Bit-identity gate for each non-ablation build.** Fixed inputs, real gate/up/down weights plus one R = 4 `exl3_dec_gemv_r` call, saved per build and compared with `torch.equal` against the default build.
   - Use the operator's `kverify.py` (`wsl setup` repo) if it compares outputs bitwise.
   - Otherwise, **ask the operator** before extending `tools/decode/test_dec.py` with a dump/compare mode (editing an existing tool is allowed, but say so first).
   - Any mismatch: stop and report the build and the shape.
4. **Occupancy facts** (plan §2.1): VGPR, SGPR and LDS of `gemv_kernel<8, CB>` for each build, from the code object. Also read the VGPR file size per SIMD for gfx1151 from `rocminfo`/ISA docs; it is **[?]**.
5. **Per-shape time** for each build, with the E2 method (24-layer round-robin, fresh interleaved processes, n >= 4, median and range): gate/up, down, and the q/k/v and GDN **multi launches as production groups them**, not single matrices.

**Report (append to this task's Evidence and to the journal):**
- one table: build x shape -> us/call (median, range) and GB/s
- the ablation arms for gate/up, down and the smallest shape
- VGPR/LDS/occupancy per build
- bit-identity PASS/FAIL per build
- generator-loop tok/s for the best build

**Decision to write down** (plan §1/§6.3):
- If a flag build wins with no bit change, propose flipping the default in `exl3_dec.cu` (one line). **Ask the operator first**, because the default also affects MoE (`exl3_dec_moe*` share `tile_dot`). Re-measure Qwen3-30B-A3B before flipping.
- Then name the next step: A5(a) barrier removal, the Phase 2 table (which needs the §3.1 split-key fix first), or Path M. Base the choice on what the ablation says limits the kernel.

**Out of scope this session:** source changes beyond a default flip, the per-shape table, the Path M port, and new scripts without consent (`scripts/windows/CLAUDE.md`).

**Stop and report if:**
- a bit-identical build changes outputs
- a build fails to compile (keep the run folder; add a `KNOWN` signature only if it is a new real failure)
- a TDR occurs (Display event 4101)
- the baseline does not reproduce E2 within ~5%

## Steps

1. Reproduce E2 and E5 on the target machine before changing anything; record torch/HIP/driver versions.
2. Diff this fork's `quant/` files against upstream `a3bfbfd0` (and `0ca49f29`) for the list above; decide whether to take upstream's versions (record origin and files in [assessment.md](../../assessment.md) section 7c; both trees are MIT) or merge. Do this in a **separate clone**, not in the working tree.
3. Scratch build with the exclusions lifted for the files above (about 12 minutes full compile, 4 jobs per [install.md](../../install.md)); fix link errors; do not change loader code ad hoc.
4. Add the single-row routing behind an env switch; A/B on the gate/up microbenchmark, then on the generator loop.
5. Gates, all of them: see Done when. If Option A cannot link within a reasonable effort, do Option B.

## Done when

- Gate/up microbenchmark (E2 method) at **<= 205 us** (upstream 201.6, current 217.0), outputs equal to the current path to fp16 rounding.
- Plain decode on the 27B generator loop **>= 12.95 tok/s** (current 12.7; target about +2%), with no change at other row counts.
- **plain == speculative bitwise** still holds (the property this fork documents for `--draft-policy`; use the existing gate, for example `tools/qwen/serve_check.py` **[?: confirm it covers this]**, and the distinct-prompt soak from [S4-05](../s4-serve-qwen/05-stability.md)).
- Prefill (421-472 tok/s at 2K-32K, E1) and the E5 TTFT table unchanged within noise (no row count slower).
- No hang or TDR over a soak (check Event Viewer, Display event 4101; see `../../review/tdr.md`); first-use autotune cost measured and either persisted in the tune cache or hidden by warm-up (the autotuner tunes per shape on first call **[I]**).
- Switch default and measurements written into [journal.md](../../journal.md); this task's Evidence filled.

## If it fails

Link errors from missing `rocm/` pieces: list them; importing upstream's whole ROCm layer is a larger task, then record it as its own task and fall back to Option B. A deadlock or TDR from the RDNA device-barrier launch path: set `EXL3_RDNA_OCC_MAX=1`, otherwise stop and report with the kernel name and the Event Viewer entry. Bit-identity broken under speculation: keep the new path for plain decode only, behind `--draft-policy off`. If no option reaches 205 us, record the best number and the shape.

## Follow-ups seen while measuring (not part of this task)

- **24-32-row band:** this fork's TTFT jumps from 213 ms at 17 rows to 746 ms at 24 rows (the `hip_gemv` row cap is 16, `EXL3_GEMV_HIP_MAX_M`, then `reconstruct_hgemm` costs a roughly 0.7 s fixed dequantisation floor), while upstream's `exl3_gemm` is 547 ms there. Any prompt of 17-191 tokens pays about 0.7-0.9 s [V], which is a typical agent follow-up. A cheaper kernel for 17-144 rows would help every conversation turn.
- **Other decode items** (about 5 ms of the gap not covered here): inter-block glue +2.2 ms, outside `model.forward` +2.4 ms, attention path +0.6 ms; see the operator's `kyojin_instrumentation_profiling.md`.

**Evidence.** (run folders, versions, numbers; fill in when implemented)

- Baseline numbers are in E1-E5 above, measured 2026-10-31/11-01 on the target machine; nothing in this repo was built or changed for them.

**Links.** [S4-05 stability](../s4-serve-qwen/05-stability.md) · [S6-04 Flash-Next](04-flash-next-85gib.md) · [assessment.md](../../assessment.md) · [deferred.md](../../deferred.md) · [journal.md](../../journal.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
