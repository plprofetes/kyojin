# S6-08 execution plan: single-row decode GEMV on gfx1151 (for a local agent with GPU access)

Companion to task [S6-08](08-decode-gemv-upstream-exl3-gemm.md). That file has the measured baseline (E1-E5) and the 2026-11-01 evaluation. This file says what to do, in what order, and when to stop. It concentrates on the kernel-level parts that are hard to get right: the decode mapping, split-K and reduction order, bit-identity, occupancy, and the port. Build and binding plumbing is mentioned only where it can break something.

**Marking:** **[V]** verified (source read or run), **[I]** inferred, **[?]** unknown. The plan was written without a GPU, so every performance statement here is [I] until a run on the target machine records it.

## 0. Context the agent must hold

### 0.1 Corrected premise (read this first)

Upstream's single-row decode on ROCm **does not** use the autotuned cooperative `exl3_gemm` [V, upstream `a3bfbfd0`]:

- `BC_LinearEXL3::run_gr` -> `exl3_gemm_gr` (`quant/exl3_gemm.cu:118`). Under `USE_ROCM` this first calls `exl3_gemv_multirow_try_launch` for 1 <= m <= 8 (around line 203). `CoopKernelAutotuner` (around line 312) is reached only if that call declines.
- For m = 1 the call is accepted when `n_tiles <= 2048` and `EXL3_GEMV_MR_M1` is unset or 1 (`rocm/quant/exl3_gemv_multirow_rdna.cu:77-100`). It is declined for lm_head-scale N, which is why lm_head is already equal (E2).
- Upstream's 201.6 us gate/up time therefore comes from `exl3_gemv_mr_had_in_single` + `exl3_gemv_mr_dot_single<bits=4, c_fp32, cb, WARPS=4, M=1>`.

**Consequence:** porting `CoopKernelAutotuner` cannot recover the gap. Do not start there. The two real options are:

- **Path T:** tune this fork's own `exl3_dec` (a static per-shape table, optionally produced by an offline tuner).
- **Path M:** migrate the m = 1..8 dense GEMV to upstream's multirow kernel.

### 0.2 The two kernels side by side (gate/up: K = 5120, N = 17408, 4 bpw, kt = 320 k-tiles, 1088 n-tiles)

| | this fork: `exl3dec::gemv_kernel<KB2=8>` (`quant/exl3_dec.cu:784`) | upstream: `exl3_gemv_mr_dot_single` (`rocm/quant/exl3_gemv_multirow_rdna.cu`) |
|---|---|---|
| work unit | **one lane = one 16x16 tile**; a wave covers 32 adjacent n-tiles (a 512-column strip) | **one wave = one tile** (32 lanes cooperate, each decodes 8 weights: `dq_dispatch(b_ptr, lane << 3, ...)`); `tpb = 2` tiles per wave |
| bytes per lane per k-tile | 128 B (8 x 16 B loads), then 256 serial trellis decodes | about 4-8 B, then 8 decodes |
| k split inside block | 8 waves x `ktw` (= 8) k-tiles each | 4 waves x `ceil(320/4)` = 80 k-tiles each |
| k split across blocks | **yes**: `kbs` = 5 blocks per strip, fp32 partials in global scratch, atomic arrival counter, last block sums in index order (`sum_coherent`) | **none** |
| grid | 34 strips x 5 = **170 blocks x 256 threads** (1,360 waves) | 1088 / 2 = **544 blocks x 128 threads** (2,176 waves) |
| input rotation | per block, in LDS (`prologue_x`), repeated across 34 strips | once per call into a global `A_had` (separate kernel; stays in L2) |
| load pipelining | `EXL3_DEC_PF = 0` (compile-time default): with ring 1, tile i+1's load is issued **after** tile i's decode, so one tile is in flight per lane and latency is hidden only by other waves | many short loads from many waves |
| output | block reduction in LDS, then fwht128 + svh in the last block per strip | fused epilogue: the last-arriving warp per 128-column segment rotates in place |
| launches per linear | 1 | 2 (rotation + dot) |

Hypothesis to test first [I]: this fork's gemv is latency-bound per wave. Each lane has one 128 B tile in flight and a long serial decode, the waves are short-lived (8 tiles), and every block pays the prologue, the reduction and the split-K handshake. `EXL3_DEC_MIN_BLOCKS=192` (more, shorter blocks) helping gate/up (218 -> 209 us) while hurting smaller shapes is consistent with this: more waves for memory-level parallelism, against more fixed cost per block.

### 0.3 Invariants that must survive (non-negotiable)

1. **plain == speculative, bitwise.** Row j of an R-row verify must equal a 1-row call on that row.
   - In this fork that holds only because `dec_gemv_r_impl` (`exl3_dec.cu:~2637`) makes the same `pick_ktw`/`kbs_of` choice as `dec_gemv_impl` (`:~2266`) and runs the same per-tile arithmetic (`lane_gemv_r` vs `lane_gemv`). Changing the split for one kernel without the other breaks it.
   - The split decides the accumulation order: within a wave, over `ktw` tiles; within a block, over 8 waves (`block_reduce`); across blocks, over `kbs` partials in index order. **Any change to `ktw`, `kbs`, WPB or the reduction trees changes logits.** That is allowed only if plain and verify change together.
2. **Existing hazard to fix before any tuning** [I, from code]. `dec_gemv_r_impl` sums `strips_total` over **all** matrices in the launch and has no `strips_hint`. `mlp.py`'s `VERIFY_FUSE` branch (opt-in, `EXL3_VERIFY_FUSE=1`) fuses gate+up into one `exl3_dec_gemv_r_multi`, while plain decode calls gate and up separately. Whenever `min_blocks` falls between `strips x kbs` and `2 x strips x kbs`, the two paths pick different `ktw`. Examples:
   - The 27B gate/up under `EXL3_DEC_MIN_BLOCKS=192`: plain ktw 4, verify ktw 8.
   - A 2048 -> 8192 MLP at the default 32: plain ktw 8, verify ktw 16.
3. **Ragged split limits** [V, from code]. `k_split` (`exl3_dec.cu:698`) accepts any `kbs` and gives each block `ktw = g1 - g0` tiles per wave. `EXL3_DEC_KBS` therefore reaches every kbs from `ceil(G/32)` to G (G = kt/8 = 40 for K = 5120) **for the 1-row kernel**. But:
   - **ring 2** (KB2 > 8, i.e. above 4 bpw): `lane_gemv` processes tiles in pairs and `ktw_valid` requires `ktw % ring == 0`. `kbs_of` does **not** check this for a forced kbs, so an odd ragged ktw is unchecked. Restrict to `G % kbs == 0 && (G / kbs) % ring == 0` above 4 bpw, or verify outputs.
   - **R-row kernel:** `rpb = GEMV_R_ROW_BUDGET / ktw` is sized from `pick_ktw`'s ktw, not from the ragged maximum `ceil(G/kbs)`. MODE 2's phases assume `ktw % 16 == 0` when `ktw > 16` (`exl3_dec.cu:2519`). A tuned kbs therefore needs `ceil(G/kbs) <= 16`, or `rpb` and `ktp` must be recomputed from the ragged maximum. Otherwise the LDS can overflow or rows can be silently skipped.
4. **No TDR or hang.** Neither path uses a device barrier, which is one reason Path M is safer than the cooperative GEMM.

### 0.4 Measurement rules (from the operator's earlier sessions; do not relitigate)

- Wall-clock replay: 300 calls pipelined, one final `synchronize`, median of 3.
- Round-robin over 24 layers' real weights (DRAM, not MALL-warm).
- Fresh interleaved processes, n >= 4; report the median and range. The same configuration varied 216.7-227.0 us across processes.
- Discard the first pass. `Event.elapsed_time` under-counts chains of small kernels on this stack, and `torch.profiler` has no GPU timeline here.
- Record the power state, driver, torch and HIP versions.
- Microbenchmark: the operator's `kgemv.py` / `klinear.py` (`wsl setup` repo, `docs/engine-comparison-20261031/profiling/`). In this repo, `tools/decode/sweep.py` and `tools/decode/test_dec.py` (`bench`, which flushes 256 MB between iterations) are the existing harnesses to extend; editing an existing tool needs no consent (`scripts/windows/CLAUDE.md`).

## 1. Decision structure (happy path)

**Updated 2026-11-01:** do **Path A** (section 6: adapt upstream's techniques inside `exl3_dec`; several already exist behind compile flags) before Path M. The tree below still holds, with Path A's bit-identical flag builds folded into Phase 1 and its source changes into Phase 2.

```
Phase 1 (attribute, 0.5-1 day) -> where do the 15 us go?
   |-- DRAM stream / occupancy dominated --> Phase 2 (Path T: kbs + PF + per-shape table, 1-2 days)
   |                                            |-- gate/up <= 205 us and every shape no worse --> DONE (skip Path M)
   |                                            '-- gap > ~5 us remains --> Phase 3 (Path M)
   '-- trellis-decode (ALU/latency) dominated -> Phase 3 (Path M) directly: the per-lane 256-decode chain
                                                is the design limit, and tuning the grid will not fix it
```

**Why not "autotune" as the happy path.** A runtime autotuner for `exl3_dec` would choose among (kbs, ktw, PF) per shape. The candidate set is small (about 10 per shape), shapes are known at load time, and timing on this stack is unreliable for 100-300 us kernels (see 0.4). An **offline sweep that writes a static table** gives the same result with none of the run-time risk. A runtime tuner is justified only for unknown models on unknown parts, which is not this fork's target (a single gfx1151). Verdict: offline-tuned table **yes**; port of `CoopKernelAutotuner` **no**; runtime tuner for `exl3_dec` **not now** (add to `deferred.md` if wanted).

## 2. Phase 1: attribution (no source changes except one build flag)

**Goal:** split gate/up's 218 us into DRAM stream, trellis decode, and epilogue/split-K, and know the occupancy of both kernels.

1. **Occupancy facts (static, from the code objects).** Build this fork's extension and upstream's tree with `--save-temps`, or extract the code object (`roc-obj-ls` / `llvm-objdump --offloading`) and read `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size` (LDS) for:
   - `exl3dec::gemv_kernel<8, CB>` (CB = the codebook of the 27B pack; Qwen packs are usually mul1 -> CB 3 or 2, so check `dense_kb2_of`/`launch_gemv`)
   - upstream `exl3_gemv_mr_dot_single<4, false, cb, 4, 1>`

   Compute waves per SIMD (gfx1151: wave32, 20 WGP = 40 CU = 80 SIMD; VGPR file size per SIMD **[?]**: gfx1100 has 1536, gfx1102/1103 have 1024, so read it from `rocminfo` or the ISA docs, do not assume). Write down: resident blocks per CU, total resident waves, and whether 170 blocks (or 340 with `MIN_BLOCKS=192`) fit in **one** wave of residency or leave a tail.
2. **Ablation build:** `QWG_ABLATE=1` is a compile-time define (`exl3_dec.cu:36`). Add `-DQWG_ABLATE=1` to the extension flags for a scratch build only. At run time `EXL3_QWG_DABL` = 1 (no decode), 2 (no DRAM stream: every k-tile read hits rows 0..ktw-1, cache-resident), 4 (no epilogue), and combinations. **Results are garbage by design: timing only.**

   | arm | reading |
   |---|---|
   | default - DABL=1 | cost of the trellis decode not hidden behind loads |
   | default - DABL=2 | cost of the DRAM stream not hidden behind decode |
   | default - DABL=4 | cost of block reduce + split-K + output rotation |
   | DABL=3 | floor: launch + prologue + reduction only |
3. **Prefetch:** rebuild with `-DEXL3_DEC_PF=1`. This is documented bit-identical: the same `tile_dot` order, only the load is issued one tile earlier (`exl3_dec.cu:384-405`). Check the VGPR count again (it adds 8 x u32x4 = 32 VGPRs per lane). PF=1 is the direct test of the "one tile in flight" hypothesis.

**DoD Phase 1:** a table in the S6-08 Evidence section with the four ablation arms and PF=1 vs PF=0 for gate/up, down (17408 -> 5120) and 5120 -> 1024, plus VGPR/LDS/occupancy for both kernels. Then a written call: Phase 2 or Phase 3.

## 3. Phase 2 (Path T): retune `exl3_dec` per shape

### 3.1 Fix the split-choice key first (bit-identity, invariant 2)

Make the split a function of the **per-matrix** shape, so that fusing matrices into one launch never changes it. Minimal form: give `dec_gemv_r_impl` the same `strips_hint` that `dec_gemv_impl` already has (`exl3_dec.cu:2270, 2292`), and have both route through one helper:

```cpp
// exl3_dec.cu, next to pick_ktw / kbs_of. One place decides the k partition for the 1-row AND R-row kernels.
struct Split { int ktw; int kbs; };
inline Split choose_split(int Kdim, int kb2, int strips_key, int ring)
{
    const int kt = Kdim / 16, G = kt / 8;
    const int forced_kbs = env_int("EXL3_DEC_KBS", 0);
    int kbs = forced_kbs > 0 ? forced_kbs : tuned_kbs(Kdim, kb2, strips_key);   // 3.3; 0 = no entry
    if (kbs > 0) {
        kbs = std::min(std::max(kbs, (G + KTW_MAX - 1) / KTW_MAX), G);
        const int ktw_max = (G + kbs - 1) / kbs;                                 // ragged maximum (k_split)
        TORCH_CHECK(ring == 1 || (G % kbs == 0 && (G / kbs) % ring == 0), "exl3_dec: kbs ", kbs, " breaks ring ", ring);
        return { ktw_max, kbs };                                                 // R-row callers size rpb/ktp from ktw_max
    }
    const int ktw = pick_ktw(kt, strips_key, ring, env_int("EXL3_DEC_MIN_BLOCKS", 32), env_int("EXL3_DEC_KTW", 0));
    return { ktw, kbs_of(kt, ktw, 0) };
}
```

- `strips_key` is the strip count of **one** matrix when all jobs share N. That covers gate+up and the 1-row/R-row pairs. Otherwise keep today's `strips_total`, which leaves the qkv / GDN / MLA multi launches unchanged.
- **Logits change** only where the hazard already exists (VERIFY_FUSE on, small-model band). Default plain decode on the 27B is unchanged (ktw 8 / kbs 5 either way).
- The R-row kernel: replace `GEMV_R_ROW_BUDGET / ktw` and the MODE 2 `ktp` logic with `ktw_max`, or reject `ktw_max > 16` for tuned entries (simplest; see invariant 3).

**Gate:** `tools/decode/test_dec.py`-style bit check. For R = 2..8 rows and every 27B shape, `exl3_dec_gemv_r(_multi)` row j must equal `exl3_dec_gemv` on row j **bitwise** (`torch.equal`), with and without a forced `EXL3_DEC_KBS`. Do it for gate+up fused and unfused.

### 3.2 Sweep (no rebuild; env read per call)

Per shape class (27B: 5120->17408, 17408->5120, 5120->10240, 5120->12288, 6144->5120, 5120->6144, 5120->1024, plus the MoE-priority models' dense shapes), in fresh processes:

- `EXL3_DEC_KBS` in {G/8 .. G}, e.g. G = 40: {3,4,5,6,7,8,10,12,14,16,20}. Respect the ring rule above 4 bpw.
- With and without the PF=1 build if Phase 1 showed it helps.
- Correctness of each candidate: max-abs-diff vs the default <= fp16 rounding (it will **not** be bitwise; the split changes the order).
- Record us/call and GB/s against the measured streaming ceiling (~238 GB/s).

Expected shape of the result [I]: big-N shapes (gate/up) want more blocks; small-N shapes (5120 -> 1024, 2 strips) want fewer, longer blocks. That is why a single `MIN_BLOCKS` cannot win everywhere.

### 3.3 Table

```cpp
// exl3_dec.cu. Offline sweep result, gfx1151 only (guard with the device's gcnArchName if other parts matter).
// key: (Kdim, kb2, strips of ONE matrix) -> kbs. Missing -> pick_ktw rule (unchanged behaviour).
static int tuned_kbs(int Kdim, int kb2, int strips)
{
    static const struct { int K, kb2, strips, kbs; } T[] = {
        // { 5120, 8, 34, <from sweep> },   // gate/up 27B
        // { 17408, 8, 10, <from sweep> },  // down 27B
    };
    for (auto& e : T) if (e.K == Kdim && e.kb2 == kb2 && e.strips == strips) return e.kbs;
    return 0;
}
```

A per-device tune-cache record (`exllamav3/util/tune_cache.py`) is the alternative to a compiled table. It is worth doing only if more than one GPU model is targeted. It is not needed for gfx1151 alone.

### 3.4 Fuse plain gate+up (one launch instead of two)

In `mlp.py`'s per-slice loop, before the `elif self.multi_gu[s] is None ...` branch, add a rows == 1 branch that calls `ext.exl3_dec_gemv_x` with `[gate, up]` and `strips_hint` = strips of one matrix. This mirrors the `VERIFY_FUSE` branch at `mlp.py:~761`. With 3.1 in place it is bit-identical to two separate calls by construction: same split, same per-job arithmetic, and jobs do not interact. Gain [I]: one launch (~10-20 us of ramp) per layer x 64 layers, and a 340-block grid instead of two 170-block grids (better tail behaviour). Gate it with `torch.equal` against the unfused path.

**DoD Phase 2 (= S6-08 "Done when", restated):**
- gate/up <= 205 us (E2 method), and **no** shape class slower than today beyond noise
- 27B plain decode >= 12.95 tok/s (generator loop)
- plain == speculative bitwise: the 3.1 gate, plus `tools/qwen/spec_bench.py` lossless arm, plus `tools/decode/dflash_check.py` (no first divergence attributable to numerics)
- E5 TTFT table and 2K-32K prefill unchanged within noise (Path T touches only R <= 32 `exl3_dec` paths; prefill above 32 rows does not use them)
- no TDR over the S4-05 soak

## 4. Phase 3 (Path M): migrate to upstream's multirow GEMV

Do this only if Phase 1 says decode-bound, or if Phase 2 leaves more than ~5 us/call on gate/up.

### 4.1 Scope and closure

Entry `rocm/quant/exl3_gemv_multirow_rdna.cu` (the single-matrix `exl3_gemv_multirow_try_launch` only). The include closure is **16 files / 5,498 lines** [V, static count]:

- **absent here:** `rocm/quant/exl3_gemv_{multirow_rdna.cu,multirow_rdna.cuh,kernel_rdna.cuh,tiles_rdna.cuh}`, `rocm/quant/exl3_mgemv_rdna.cuh`, `rocm/ptx_rocm.cuh`, `ptx_portable.cuh`, `arch.cuh`
- **diverged here:** `ptx.cuh`, `quant/exl3_dq.cuh`, `quant/codebook.cuh`, `util.cuh`, `graph.cuh`, `compat.cuh`
- **identical:** `quant/hadamard_inner.cuh`, `util.h`
- **plus:** four host helpers that live in `rocm/quant/exl3_gemv_rdna.cu` (`exl3_gemv_splitk_warps`, `exl3_gemv_core_mode`, `exl3_gemv_tiles_tpb`, `exl3_gemv_fuse_out_enabled`)

Not needed: the cooperative GEMM, comp units, kernel map, `rdna_wmma`, or a device barrier.

### 4.2 The hard parts (where the time goes)

1. **The decoder must match the weights bit for bit.**
   - Upstream's `dq_dispatch<ka, cb, half>` (`quant/exl3_dq.cuh` + `codebook.cuh`, both diverged here) decodes 8 weights per lane from `lane << 3`.
   - This fork's `tile_dot` decodes a whole tile per lane with its own ring/window math (`exl3_dec.cu` header comment) and has **extra codebook templates** (CB = 3: mul1 with the affine map folded into block sums, `dec_exact()`).
   - Vendor upstream's `exl3_dq.cuh`/`codebook.cuh` **under a separate path or namespace**. Do not replace this fork's copies in place: `ptx.cuh` is included by 41 sources and `util.cuh` by 92.
   - Then verify on **real packs**, not synthetic trellis, that the vendored kernel's outputs equal `reconstruct` + fp32 matmul to fp16 rounding for mcg, mul1 and the 3-codebook packs this fork serves.
   - Half-integer bitrates (`EXL3_HALF_BITS`, the `dq8_half` decoder): the 27B is integer K = 4. Defer half rates unless an MoE-priority pack needs them.
2. **The numeric chain changes, so plain and verify must move together.**
   - Upstream's per-row chain: per-lane `fdot2` into two accumulators, a quad `shfl_xor` reduction, then a 4-warp in-block sum (`exl3_gemv_dot_tile_splitk_mr`), then the epilogue Hadamard. This is not this fork's chain, so switching m = 1 alone breaks plain == speculative.
   - Upstream guarantees row r of an m-row call is bit-identical to its m = 1 call (the rows share the B decode and keep separate accumulators). Route **R = 1..8 together**, and chunk R = 9..32 into calls of <= 8 rows. Chunking is bitwise safe because a row's chain does not depend on M.
   - The bit gate is the same as 3.1, but now between the new 1-row and new R-row calls.
   - Expect logits to differ from the current fork (not from upstream) at the fp16-rounding level. Rerun `dflash_check.py` / `spec_bench.py` and the quality spot checks the fork uses (`tools/qwen/gate_texts.py`).
3. **Warp count and tiles per wave are fixed rules tuned on upstream's parts.**
   - `exl3_gemv_splitk_warps`: blocks >= 1024 -> 4; k_tiles >= 1024 or blocks < 128 -> 16; else 8.
   - `exl3_gemv_tiles_tpb`: 2 if `(n_tiles/2) * warps >= 1024 * sms / 20`.
   - Both are env-overridable (`EXL3_GEMV_SPLITK_WARPS`, `EXL3_GEMV_TILES_T`), which makes them the Path M "autotune" surface. Sweep them with the 3.2 method if the port lands below upstream's 201.6 us. Changing warps changes the in-block sum order, so again plain and verify must change together (they share `exl3_gemv_mr_warps`).
4. **The fused epilogue's arrival protocol.**
   - The last warp of each 128-column segment (8 tiles, spread over 4 blocks with tpb = 2) rotates the segment. Its counters live in a per-device `Exl3GemvMrParams` block that is allocated outside graph capture (`exl3_gemv_multirow_prewarm`).
   - This fork's decode runs eager and does not use upstream's `Graph`, so pass `graph = nullptr`. Make sure the parameter block is allocated before the first decode step (in a warm-up or on the first call) and that its counters self-reset as upstream's do.
   - **[?]** Check that the counters are stream-ordered-safe when `dec_workspace`'s "side" stream (`_dec_side`) is in use. This fork runs some decode work on a second stream, and upstream assumes one stream per parameter block.
5. **Rotation into global `A_had`.** Upstream needs an m x K fp16 scratch per call. Allocating with `torch.empty` per call is fine (host cost is already lower here: ~41 vs 75 us), or keep a per-device buffer like `dec_workspace`.
6. **Do not regress the 9-144-row band.** E5 shows this fork's `hip_gemv` (up to 16 rows) and `reconstruct_hgemm` beat upstream above 16 rows. Path M routes rows <= 8 only. Leave `EXL3_GEMV_HIP_MAX_M` and `AUTO_RECONSTRUCT_THRESHOLD` alone. For 9-16 rows keep today's path unless measured.

### 4.3 Plumbing (secondary; pointers only)

- **Build:** add `rocm/` sources to `build_config.py` for ROCm only. `get_sources` walks the whole tree, so a new `rocm/` directory must be **excluded for CUDA builds** (add it to the non-ROCm exclusions) and included for ROCm. The Linux container build (`.github/workflows/container.yml`) must stay green (S2-03).
- **Binding:** one new op, e.g. `exl3_mr_gemv(x[m,K], trellis, suh, svh, out[m,N], a_had, K, cb, out_fp32)`, in `bindings.cpp`.
- **Routing:** in `LinearEXL3.forward` (`exllamav3/modules/quant/exl3.py:~469`), `rows <= 8 and EXL3_DEC_MR=1` -> the new op, ahead of `dec_gemv` / `dec_gemv_r`. Default off until the DoD is met.
- **Licence:** upstream is MIT. Record the origin commit, the copied files and the notice in `doc/windows/assessment.md` §7c in the same commit (`scripts/windows/CLAUDE.md`, "Sources and licences").

**DoD Phase 3:**
- Gate/up <= 205 us (target: upstream's 201.6), down and the small shapes no slower than Path T's table
- Rows 1..8 bitwise per-row consistent (new 1-row vs new R-row), plain == speculative
- 4-8-row TTFT moves toward upstream's (E5: 146/182 ms vs 176/205 ms here)
- every other S6-08 "Done when" item
- Linux build green

## 5. MoE note (fork priority)

Routed-expert decode (`exl3_dec_moe*`, `EXL3_DEC_MOE_{KBS,KTW,MIN_BLOCKS}_{A,B}`) uses the same `pick_ktw`/`kbs_of`/`k_split` machinery. The Phase 1 ablation and the Phase 2 sweep method carry over unchanged, so run them on the Qwen3-30B-A3B pack (`C:\exl3-probe\models\Qwen3-30B-A3B-exl3-4.0bpw`) once the dense table works. Upstream's MoE counterpart is `exl3_rocm_moe_decode` (`rocm/quant/exl3_gemv_multirow_rdna.cuh`): same structure (gate|up as one 2S-slot GEMV, the activation folded into down's input rotation). Compare it with the same table as 0.2 before any MoE kernel port.

## 6. Path A: adapt upstream's improvements inside `exl3_dec` (preferred over Path M)

Operator question (2026-11-01): can upstream's fixes be adapted to the local kernels instead of porting? **Yes, for most of them.** Upstream's own notes (`rocm/quant/exl3_gemv_tiles_rdna.cuh:1-60`) say its decode GEMVs were **VALU-bound, not DRAM-bound**. The rewrite that made them fast consists of instruction-level techniques, and those carry over to `exl3_dec` without changing its lane-per-tile layout.

### 6.1 Headroom budget (27B, from E2 numbers, all [I])

| group | ms/token now | bytes/token | ms at 220 GB/s | possible gain |
|---|---|---|---|---|
| gate/up (128 calls) | 30.85 | 5.71 GB | 25.9 | ~4.9 |
| down (64 calls) | 15.15 | 2.85 GB | 13.0 | ~2.2 |
| lm_head | 4.23 | 0.95 GB | 4.3 | 0 (already ~225 GB/s) |
| all other dense linears | ~24.8 | ~4.1 GB (13.6 total minus the above) | ~18.6 | ~6 |
| **total linear time** | **75.0** | ~13.6 GB | ~61.8 | **~13** |

- Decode is ~79 ms/token now. Taking ~13 ms off would give about 66 ms, i.e. ~15 tok/s, which is above upstream's 14.0.
- The "other" group is where most of the relative loss sits. Small-N shapes run with few blocks: for example a lone 5120 -> 1024 has 2 strips x 5 blocks = 10 blocks = 80 waves, about one wave per SIMD on 80 SIMDs, at ~53 GB/s. In production several of these are fused into `exl3_dec_gemv_multi` launches (q/k/v, GDN), so **measure them in their real launch grouping**, not one at a time.
- Recompute this table from the real bytes per call before you rely on it.

### 6.2 Upstream techniques -> where they stand in `exl3_dec`

| # | upstream technique (`exl3_gemv_tiles_rdna.cuh`, `exl3_gemv_multirow_rdna.cu`) | `exl3_dec` today | adaptation | numerics |
|---|---|---|---|---|
| A1 | hash multiply as `v_pk_mul_lo_u16` + `v_mad_u32_u16` (2 full-rate ops instead of 1 quarter-rate `v_mul_lo_u32`) | **already implemented** as `EXL3_DEC_PROD=2` (`exl3_dec.cu:~115-180`, `mul1_pair` asm block), but **default 0** (quarter-rate `v_mul_lo_u32`). No recorded measurement in the repo **[V: grep]** | rebuild with `-DEXL3_DEC_PROD=2`, measure, flip the default if it wins. **Applies to mul1 (CB 2/3) only:** `mcg_fold` still does a full 32-bit `w * c.mul`. Check which codebook the 27B and MoE packs use (`mcg`/`mul1` tensors in the pack) **[?]**. For mcg the same split works when the window is known to be 16 bits wide, but upstream's mcg decode must be read first | **bit-identical** (exact integer arithmetic) |
| A2 | byte sum by `v_sad_u8` / `v_sad_hi_u8`, giving a packed fp16 pair with no repacking | already used (`PORT_SAD_U8`, `PORT_SAD_HI_U8`) | none | n/a |
| A3 | funnel shift by one `v_alignbit_b32` | already used (`window<>`, compile-time windows) | none | n/a |
| A4 | wave-uniform SGPR addressing, zero VALU address math per tile | per-lane 64-bit pointer advanced by `tstride` per tile: about 2 VALU ops per 128 pairs | low priority; check the ISA for per-tile 64-bit adds | bit-identical |
| A5 | input rotated **once** (separate kernel, `A_had` in L2), no LDS prologue; measured upstream as part of 213.8 -> 201.6 us | every block rotates its chunks into LDS, then `__syncthreads`; repeated across all strips | (a) when `ktw % 8 == 0`, wave w only reads the chunks it could rotate itself: assign chunk c to wave `c / (ktw/8)` and drop the prologue barrier for exact codebooks (`xsum` is unused when `dec_exact<CB>()`). (b) For multi launches (q/k/v, gate+up), rotate x once per launch into L2 and have blocks read their slice directly. Mind the strided `x_gstride` and per-job `x_off` cases | **bit-identical** if the same `prologue_chunk` arithmetic produces the same fp16 values |
| A6 | load latency hidden by many short waves | ring 1 with `EXL3_DEC_PF=0`: tile i+1 is loaded after tile i is decoded (one tile in flight per lane) | `-DEXL3_DEC_PF=1` (exists; costs ~32 VGPRs). Also sweep `EXL3_DEC_SB` (sched-barrier spacing, default 4) with PF on | bit-identical (documented) |
| A7 | no cross-block split-K for wide N (in-block only) | `kbs` blocks write fp32 partials; the last block sums `kbs x 512` floats with coherent loads (serial tail per strip) | per-shape `kbs` table (Phase 2): small kbs for wide N, large kbs for narrow N | changes order; plain and verify move together |
| A8 | narrow-N shapes get more in-block k-parallelism (`splitk_warps` = 16 when blocks < 128) | WPB fixed at 8; small-N parallelism only through `kbs` (max G = 40) | ragged `kbs` up to G for small N (in the table). If still short, a WPB = 16 instantiation for small-N shapes | changes order (as A7) |
| A9 | small decode body per lane (8 weights per tile), so no I-cache pressure | fully unrolled 256-weight tile, ~10 KB of code; the source says gfx1151 is **instruction-cache bound** here and ring 2 thrashes it | only by a different lane mapping (2 lanes per tile, 128 weights each), which is half of Path M. Do it last, and only if A1+A6 leave the kernel VALU- or I-cache-bound | changes order |
| A10 | fused output epilogue per 128-column segment | block reduce in LDS (3-4 barriers) + split-K handshake | follows A7: with kbs = 1 for a shape the handshake disappears | n/a |

### 6.3 Order of work (cheapest and bit-identical first)

1. Phase 1 attribution as written, **plus** builds for A1 (`EXL3_DEC_PROD=2`) and A6 (`EXL3_DEC_PF=1`, then `EXL3_DEC_SB` in {2, 4, 8}).
   - Each is a one-flag rebuild and bit-identical: gate it with `torch.equal` against the default build on all 27B shapes and on R-row verify.
   - Expected: if upstream's diagnosis holds for this kernel (VALU-bound), A1 alone moves gate/up noticeably. If the `QWG_ABLATE` "no DRAM stream" arm shows almost no change, the kernel is decode-bound and A1/A6/A9 are the levers. If the "no decode" arm shows almost no change, it is memory-bound and A5/A7/A8 are the levers.
2. A5 (a): drop the prologue barrier where wave-local. Bit-identical; a small source change in `gemv_kernel`, `prologue_x` and the R-row twin.
3. Phase 2 table (A7/A8) and fused plain gate+up (3.4).
4. A5 (b) for multi launches; A4 if the ISA shows real address cost.
5. A9 or Path M, only if items 1-4 leave gate/up above 205 us or the "other" group far from 200 GB/s.

**DoD Path A:**
- Every bit-identical step: `torch.equal` with the previous build on all shapes, 1-row and R-row.
- Each non-bit-identical step: plain == speculative gate (3.1) plus the quality spot check.
- Overall: the S6-08 numbers. A stretch target from 6.1 is a linear-time sum <= 66 ms/token, i.e. ~15 tok/s plain.
- MoE: the same flags apply to `exl3_dec_moe*` (same `tile_dot`), so re-measure Qwen3-30B-A3B after each default flip.

## 7. Stop conditions

- The Phase 1 ablation shows neither arm explains more than half the gap -> stop and report. The gap is then elsewhere (launch overhead, inter-block glue: E1 has 2.2 ms of glue and 2.4 ms outside `model.forward`).
- The 3.1 bit gate fails after the change -> revert and report the shape.
- Path M decoder mismatch on a real pack that cannot be explained within a day -> stop and keep Path T.
- Any TDR (Display event 4101) -> stop, record the kernel and the Event Viewer entry.
