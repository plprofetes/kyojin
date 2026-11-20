# Dynamic (shared) GPU memory on Strix Halo + Windows: limitations

Date: 2026-11-20. Tags: **[V]** measured here, **[I]** inferred, **[?]** unknown. Evidence: [shared-vram-research.md](shared-vram-research.md); versions and rollback: [versions-and-rollback.md](versions-and-rollback.md); drafted (unfiled) reports: [bug-reports/](bug-reports/README.md).

## Setup

ASUS ProArt PX13, Ryzen AI Max+ 395 (gfx1151), 127.15 GiB RAM, BIOS v3.14; Windows 11 26H2 build 26300.9974; Adrenalin 26.11.2 (32.0.32029.2010), also 26.10.1; wheel `rocm 10.1.0a20260822`, HIP 10.0.3581.0; GPU memory mode dynamic (about 102 GB "shared", 0.33 GiB dedicated).

## Limitations

| # | Limitation | Numbers | Status |
|---|---|---|---|
| L1 | **Residency wall.** Device allocations plus pinned host memory cannot exceed about 61 GiB; pageable host memory counts about half. Beyond it GPU work returns error 719 or never completes. HIP keeps every allocation resident (`alwaysResident_`); the OS refuses the submit (`PAL result:-5 ErrorOutOfGpuMemory`). | 61.33 GiB resident in WDDM segment #3 (no CommitLimit) at failure, 0.48 x RAM; 41 GiB RAM free; DXGI budget 101.3 GiB; no TDR or event-log entry | **[V]** pure C++ (`hip_wall.cpp`, `hip_ops_after_alloc.cpp`); cause of 61 GiB **[?]** |
| L2 | **Not exposed.** No HIP attribute, `hipMemGetInfo` or PAL value shows the limit. | HIP total 76.79 GiB (stock) / 98.22 (built); PAL heaps 101.72; budget 101.30 | **[V]** |
| L3 | **Same wall for every allocation API.** | `hipMalloc`, fine-grained, uncached, managed: 60 to 61 GiB; `hipHostMalloc`: 45 GiB; chunk size irrelevant | **[V]** |
| L4 | **VMM is worse.** `hipMemMap` fails at about half the wall. | about 31 GiB mapped (1 GiB, 256 MiB chunks); independent of reserved VA size | **[V]**; no double add in clr source; charge is below clr **[?]** |
| L5 | **`PAL_ALWAYS_RESIDENT=0` is unsafe.** It removes L1 but untracked buffers page-fault (PAL -87). | pointers inside a by-value struct (every torch kernel) fail; direct pointer args pass; a single struct kernel passes, repeated rounds fail | **[V]** `hip_fresh_alloc.cpp`; exact rule **[?]** |
| L6 | **Stock runtime accounting.** `mem_get_info` total 76.79 GiB, 2:1 charge for every allocation after the first, free not returned after free. The EXL3 loader check stops at about 38 GiB. | fixed upstream (rocm-systems #10974, #13193); no public Windows wheel has it; local build gives 98.22 GiB, 1:1 | **[V]** |
| L7 | **Failure behaviour.** After a failure the context is lost; the next process can fail immediately for about 15 s. Allocate-then-submit at 80 GiB can hang forever (kill the process). Outcome varies run to run. | wall 45 to 62 GiB across runs; Flash-Next load died at 35 GiB once and 59.9 GiB once | **[V]** |
| L8 | **Monitoring blind spots.** Task Manager "shared GPU memory" does not show raw `hipMalloc`/`hipMemset` allocations (RAM climbs to 120 GB while it reads 0). Use free RAM and `wddm_segments.cpp`. | | **[V]** |
| L9 | **Driver update did not help.** 26.11.2 plus chipset update: same wall, same loader failure. | | **[V]** |
| L10 | **Upstream.** No report found for L1 to L5. #8922 (similar chip and OS, closed without a fix) and #13351 (open) are related. Bug reports drafted, not filed; expected fix odds low (kernel-mode policy). | | **[V]** search; odds **[I]** |

## Consequences for models (dynamic mode)

| Model | Device-resident size | Result |
|---|---|---|
| Qwen3.8-Flash-Next 4.05bpw | about 64 GiB | does not load (loader check at 38 GiB; error 719 at 35 to 60 GiB with the check bypassed) **[V]** |
| Qwen3.5-122B heretic 4.06bpw | about 61 GiB | at the wall, not tried in dynamic mode **[?]** |
| 35B-A3B, 27B, 9B | far below 61 GiB | fit **[V]** |
| Flash-Next 3.05bpw | not checked | would be about 48 GiB if the 2.05 to 4.05 bpw trend holds **[?]** |

## What works

- Fixed 96 GB carve-out (S0): 100 GiB allocatable, 61 to 64 GiB models loaded and ran, no 2:1 accounting for models that fit in the carve-out; Windows keeps 31.6 GB **[V, journal]**. 72 to 80 GB fixed: untested **[?]**.
- Dynamic mode with models and KV cache below about 55 GiB device + pinned memory.
- Local runtime build (`C:\Users\plpro\.kyojin\hip-build\amdhip64_7.dll`, preloaded before torch): correct `hipMemGetInfo` only; the wall stays.

## Do not use

`PAL_ALWAYS_RESIDENT=0` (L5), VMM allocation (L4), pinned host buffers beyond what the 61 GiB budget leaves (L1), expert tiering as a way around L1 (dropped by the operator as not worth the time; untested; pinned host buffers would count 1:1, pageable half).

## Open

Origin of the 61.33 GiB figure (0.482 x RAM; WDDM or AMD KMD); hang versus 719; where VMM mappings are charged twice; exact rule behind the flag-off page faults; behaviour at a 72 to 80 GB fixed carve-out; other Windows or driver builds (the #8922 reporter saw 100 GiB at a 0.5 to 1 GiB carve-out on 26H2, driver 32.0.31041.5005, `KMD_RebarControlMode` 1 vs 0 here).

## Probes (`scripts/windows/probes/`)

`hip_wall.cpp`, `hip_ops_after_alloc.cpp`, `hip_vmm.cpp`, `hip_fresh_alloc.cpp`, `hip_stloader_mimic.cpp`, `wddm_segments.cpp`, `dxgi_budget.ps1`, plus Python helpers for torch-level checks. Each prints one result line; run one process per test and wait about 15 s after a failure.

## Lessons learned

1. **`mem_get_info` and Task Manager are not ground truth on this APU.** Verify with allocate + touch + a kernel, plus free RAM and `wddm_segments.cpp`.
2. **A passing test is not a safe lever.** `PAL_ALWAYS_RESIDENT=0` passed several tests because the touched buffers were still resident; it failed in the real load. Require an adversarial case (second round, torch-style struct arguments, a real workload) before calling something a workaround.
3. **Reduce in this order:** real workload, then per-module sync hooks (`Module.load` + `torch.cuda.synchronize`), then a ten-line torch repro, then pure C++. The C++ repro turned vague claims into checkable ones.
4. **One process per test, a timeout on every probe, kill by PID or image name only.** A failure loses the GPU context for about 15 s; a hung probe holds tens of GiB.
5. **Operator observations beat theory.** The note that RAM grew while Task Manager showed 0 shared memory led to the memset/hang finding.
6. **Check upstream source and PRs before blaming the loader**; the 2:1 accounting was already fixed upstream, the wall was not.
7. **State confidence on every claim** ([V]/[I]/[?]) and correct earlier claims in place when a later test contradicts them (done for the flag-off lever and the VMM double add).
