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

- **Option A (preferred if it links):** build `quant/exl3_gemm.cu`, `quant/coop_autotune.cu`, `quant/exl3_kernel_map.cu` and the `comp_units` for K = 4 (K = 6 optional) for gfx1151, together with whatever of upstream's `rocm/` layer the RDNA inner kernel needs; route **`rows == 1`** projections through the `exl3_gemm_gr`-equivalent behind a switch (default off until A/B), keep `dec_gemv_r` and every other row count as is.
- **Option B:** retune `exl3_dec` for 5120 -> 17408, K = 4 (tile and split-k sweep, lane count, the `EXL3_DEC_PF` prefetch depth) until the microbenchmark matches 201.6 us, without new sources.
- **Option C (measurement only):** compile one source file (for example `exl3_gemv.cu`) with both toolchains to separate compiler version from source.

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
