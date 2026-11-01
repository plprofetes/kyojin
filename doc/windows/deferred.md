# Windows port: deferred items and error catalogue

Part of the [Windows port docs](README.md); background in [assessment.md](assessment.md); plan in [sprints](sprints/). Target: Windows 11 x64, Strix Halo (gfx1151), Python 3.12, ROCm/torch from AMD's nightly index, `uv`-managed venv `.venv-win`. Goal: **serve a Qwen model through `tools/qwen/serve.py` on Windows.** Revisit triggers refer to sprints (`S3` = [sprints/s3-runtime-kernels](sprints/s3-runtime-kernels/README.md), etc.).

Rules for this file: an item moves out of "Deferred" only when a sprint needs it. Every deferral says why and what would trigger it. Update this file in the same commit that changes the status.

Legend: **[V]** verified, **[I]** inferred, **[?]** unknown (see the assessment).

## 1. Deferred (not doing now)

### 1.1 Model and feature scope

| Item | Why deferred | Revisit when |
|---|---|---|
| GLM, MiMo, DeepSeek-V4.1 servers (`tools/glm`, `tools/mimo`, `tools/ds41*`) | POC is Qwen only. GLM/MiMo add MLA/DSA kernels not exercised by Qwen. | Sprint 6, after Qwen serves |
| DS41 SSD expert tier (`modules/expert_store.py` `O_DIRECT`, `engram_io.py` `os.preadv`, `dsv41/*`) | Needs `FILE_FLAG_NO_BUFFERING` / overlapped IO rewrite; experimental even on Linux. | Only if someone wants 100 GB+ models on Windows |
| Qwen3.8-Flash-Next (95 GB) and 27B-class models | Ceiling measured at about 100 GiB usable GPU memory (S1-04): an 84.5 GiB pack (Qwen3.8-Flash-Next 2.18bpw, 90.7 GB on HF) plus KV cache is not ruled out; untried. The 27B (15.7 GiB) runs. | [S6-04](sprints/s6-scale-and-speed/04-flash-next-85gib.md) |
| Vision tower (`--no-vision`) | Fewer kernels, less memory. | After text serving is stable |
| Speculative decoding / MTP drafter (`--draft-policy off`) | Adds the drafter model and extra kernels; lossless, so it is purely a speed feature. | Done for Qwen3.8-27B (MTP works, +14.7 % decode on prose); more policies (`mix`, n-gram, code prompts) in Sprint 6 backlog |
| Uncensor presets (`presets/`) | Unrelated to the port. | Never needed for the port |
| Multi-GPU tensor parallel (`model/model_tp*.py`, NCCL/all-reduce `parallel/` sources) | Single APU; `parallel/` is already excluded from the ROCm build (`build_config.py`). | Not planned |

### 1.2 Linux-specific runtime code

| Item | Where | Why deferred | Revisit when |
|---|---|---|---|
| `libamdhip64.so` lookup and `/proc/self/maps` | `util/hip_lib.py` | Needed only once the extension imports; fix against the real DLL name found by the first `doctor` run. | **Sprint 3** (first to be pulled in) |
| `hipcc` resolver (`hipcc` vs `hipcc.exe`, `/opt/rocm`, `--gcc-install-dir`) | `util/hip_compiler.py`, 14 `*_hip.py` JIT modules | Same. `--genco` `.hsaco` output is portable. | Sprint 3 |
| `/proc/meminfo`, `smaps`, `pagemap`, `MADV_POPULATE_READ` | `util/memory.py`, `util/memguard.py`, `serve.py` startup | Memory accounting for tiering/swap-in; not needed for a small model. Must degrade quietly, not crash. | [S6-05](sprints/s6-scale-and-speed/05-windows-host-memory.md) (noise: `swap_in_process` TypeError line, harmless) |
| `posix_fadvise`, `drop_caches` page-cache control | `util/pagecache.py`, `util/memguard.py` | No Windows equivalent. | Not planned |
| `/sys/class/drm/...mem_info_gtt_used`, `/proc/self/statm` metrics | `tools/glm/serve.py:1233,1237` (check `tools/qwen/serve.py`) | Cosmetic. | Sprint 4 (wrap in try) |
| CPU MoE host path Windows branch (affinity, timers) | `cpu/moe_mul1.cpp` (authors mark it UNVERIFIED) | Only used for CPU-offloaded experts; small dense/MoE-on-GPU test avoids it. | When a MoE model is offloaded |
| `prctl(PR_SET_PDEATHSIG)` | `util/misc.py` | Already platform-guarded. | Not needed |

### 1.3 Tooling, packaging, CI

| Item | Why deferred | Revisit when |
|---|---|---|
| CI for the Windows port, three tiers (decided 2026-10-31): always-on Linux checks, a manual hosted wheel build, a local GPU release gate. Previously: CI job for Windows (`windows_rocm_smoke`) | Disk is tight, not free: the footprint measured here is about 10 GB (venv 8.7 + build 0.2 + Triton cache 0.7) but GitHub documents 14 GB SSD for standard Windows runners (docs.github.com, runners reference, read 2026-10-31), and the uv download cache adds to it unless wheels are hardlinked from a cache on the same drive or installed with `--no-cache` **[?: actual free space on the runner not measured]**. Hosted runners have no gfx1151 GPU, so CI could only verify setup + build + CPU-side import, not `check`/`serve`; build time on a hosted runner is unmeasured **[?]**. The local script is the gate. | [S6-01](sprints/s6-scale-and-speed/01-ci-windows-rocm-wheel.md): manual `workflow_dispatch` build-only job; trial first. Self-hosted runner only for trusted triggers |
| Prebuilt wheels / releases, "download and use" bundle for Windows users | POC is source build. Findings so far: the wheel packages only `exllamav3*` (no `tools/qwen/serve.py`); torch/ROCm must come from AMD's index at the pinned versions; `rocm[devel]` (about 9 GB) is needed at run time (hipcc JIT, Triton's clang-cl); **Visual Studio Build Tools (VC tools + Windows SDK) are still needed at run time**: triton-windows 3.9.0.post30 compiles its launcher with the SDK's clang-cl and finds the MSVC/Windows SDK headers via vswhere (`triton/windows_utils.py`, `triton/runtime/build.py`). A bundle would drop the 12-minute extension compile, not these. Design and open questions in the journal entry of 2026-10-31 | [S6-01](sprints/s6-scale-and-speed/01-ci-windows-rocm-wheel.md) and [S6-02](sprints/s6-scale-and-speed/02-installer-and-bundle.md); needs a test on a machine without VS |
| `uv.lock` / pinned Windows requirements | First run produces `pip-freeze.txt`; pin from a known-good one. | Done: `scripts/windows/constraints-win.txt` (S5-03); a real `uv.lock` for Windows is still not planned |
| sccache / compile cache | Optimisation. | If rebuild time hurts (it will) |
| Stable AMD wheel fallback (`repo.amd.com/rocm/whl/gfx1151/`) | Kyojin is only verified on nightlies; fallback if the pinned nightly disappears. | If `doctor` amd_index fails |
| Index host equivalence (`nightly.repo.amd.com/rocm/whl-next/` vs `rocm.nightlies.amd.com/whl-multi-arch/`) | One constant (`KYOJIN_ROCM_INDEX`); verify when convenient **[?]**. | Sprint 1 |
| Docker / nix / `container.yml` | Linux only. | Not planned |
| Auto-installing Visual Studio / drivers from the script | `doctor` prints exact commands instead; installers need admin and reboot. | After the manual path is stable |
| Python 3.13/3.14, Windows 10, ARM64, other GPUs (gfx110x/gfx120x) | One target only. | Not planned |
| `tools/strix_halo/*.sh` benchmarks, `ck_pagedkv` / `libckpf.so` CK paths | Linux shell scripts; CK falls back to Triton silently. | After serving works, if perf work starts |
| Linux system tuning (IOMMU, DPM level, fan curve) | Linux sysfs. Windows equivalents (Adrenalin tuning, power plan) are untested. | Sprint 6 perf work |
| Perf parity vs Linux, benchmarks | Correctness first. | Sprint 6 |
| Windows Triton kernel seed (`kernel_seed/triton-<version>-<arch>.tar.xz`) | The repo ships a Linux seed keyed to Triton version and arch; the Windows run needed about 50 s of cold Triton compile (GDN test) and minutes for the 27B. A seed must be built per Triton version (3.9.0.post30 here) with `tools/kernel_seed/build.py` **[?]** | [S6-02](sprints/s6-scale-and-speed/02-installer-and-bundle.md) step 5, after the bundle works |
| Upstream pull request | Draft PR to the fork's `main` exists (user, 2026-10-31). The upstream PR is built from a separate branch so PM files (journal, sprints, review, tracker) stay in the fork | [S6-03](sprints/s6-scale-and-speed/03-upstream-pr.md), after S5-01 (clean-machine run) and S2-03 (Linux build) |
| Visual Studio Build Tools needed at run time (Triton launcher compile, hipcc JIT) | triton-windows finds MSVC and the Windows SDK via vswhere; a "no VS" run is untested **[?]** | [S6-02](sprints/s6-scale-and-speed/02-installer-and-bundle.md) step 4 |
| More JIT HIP kernels on Windows (`gr_mix`, `qsa_pf`, `kda_fused_h`, DS41 kernels) | Only `gdn_hf` and `ple` were exercised (dense Qwen); the others build with the same hipcc path but are untested | When a model needs them (Flash-Next: S6-04) |
| Tune caches in `<drive>:\tmp\.cache\exllamav3` (dense_gemm_tune.txt) because `HOME` is unset when serving; the `.hsaco` kernels use `%USERPROFILE%\.cache\exllamav3` | Works as is (verified: 9 s warm start). Moving it would force a re-tune. | If `C:\tmp` is cleaned or the location confuses someone: set `HOME=%USERPROFILE%` for `serve` and accept one re-tune |
| Triton `TRITON_DEFAULT_FP_FUSION` (warning on every run; default changes from enabled to disabled in Triton 3.10) | Pinned triton-windows 3.9.0.post30 keeps today's behaviour; setting `=1` explicitly would silence the warning and survive an upgrade. Numerics unaffected at 3.9. | When the Triton pin moves, or if the warning confuses logs |
| Prove the shared-file edits leave Linux unchanged: `moe_mul1.cpp` target macros now `... || defined(__clang__)`, `setup.py` Windows branches | Preprocessor check only so far (gcc/clang on Linux and clang for the Windows target evaluate the condition as intended); no Linux build or CPU-MoE test run from the authoring sandbox (no torch). The Linux condition `__GNUC__ && __linux__` is unchanged, so no behaviour change is expected **[I]**. | After the Windows happy path (`serve` works): build on the Linux box, run the CPU MoE test, diff shared files vs upstream |
| Windows ISA dispatch of `moe_mul1.cpp` under clang-cl: AVX-512 tiers (`BW/VNNI/VBMI`) compile but are untested on Windows; `cl.exe` host builds keep the empty-macro path (untested, not a goal) | Compile-only so far | When the CPU MoE path is exercised on Windows (`EXL3_MOE_CPU_MAX_ISA` to compare tiers) |

## 2. Error catalogue (how "helpful errors" evolves)

Every failing step prints and saves (`FAILURE.md` in the run's log folder): command, exit code (Windows crash codes named), the matching **known signature** (cause + fix), the **first** error lines with line numbers into the step log, and the last output. Signatures live in `KNOWN` in `scripts/windows/kyojin_win.py` and are **hypotheses** until observed.

Process after each real failure:
1. Open `FAILURE.md`. If "no known signature matched": add a `Known(...)` entry (regex from the real first-error line, cause, concrete fix) and a sample line in `test_kyojin_win.py::HelpfulErrors.SAMPLES`.
2. If a signature matched but the advice was wrong, fix the advice, not just the code.
3. Record the failure and fix in [journal.md](journal.md).
