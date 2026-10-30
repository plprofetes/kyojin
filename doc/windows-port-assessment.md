# Windows port assessment (gfx1151, Win11 x64)

Status: assessment only, no code changed. Date: 2026-10-30. Kyojin 1.8.3 vs upstream exllamav3 1.6.5.
Target: one configuration. Windows 11 x64, Strix Halo (gfx1151), latest Adrenalin driver, Python 3.12, ROCm wheels from AMD's pip index.

Legend: **[V]** verified from sources or the AMD wheel index this session; **[I]** inferred, needs a Windows run; **[?]** unknown.

## 1. Verdict

- **Build the extension on Windows: plausible, small delta.** The C++ host code is already mostly Windows-aware (inherited from upstream). Torch's own `cpp_extension.py` in AMD's Windows ROCm wheel has HIP-on-Windows paths (`hipcc.exe`, MSVC runtime flag for hipcc). **[V]**
- **Run the engine on Windows: blocked by Triton.** 52 files import `triton`, 43 at module level (unguarded). Which of them sit on the `import exllamav3` / serve import path was not traced. No Windows ROCm Triton wheel exists on AMD's index, and `triton-windows` has AMD support only as an open PR. **[V]** Gate everything else on this.
- **Memory model is a second unknown.** The product premise is 100 GB-class models in 128 GB. Windows WDDM caps GPU-addressable memory differently from Linux GTT, and the Kyojin tiering code assumes Linux memory semantics. **[?]**
- Recommendation: do a 1-day **spike** before writing the build script: (a) torch+ROCm import and matmul on gfx1151, (b) compile one `.cu` with `CUDAExtension`, (c) `import triton` and run one kernel. Script work is only worth it if (c) has a path.

## 2. Upstream exllamav3 Windows build (documented)

CUDA only. No ROCm on Windows upstream (ROCm wheels are Linux-only in `build.yml`). **[V]**

| Step | Upstream | Source |
|---|---|---|
| Host compiler | MSVC (VS 2022 Build Tools, pinned 17.9.7 in CI). JIT path searches `ProgramFiles*\Microsoft Visual Studio\{2022,2019,2017}\...\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe` and appends to PATH | `ext.py:find_msvc`, `build.yml` |
| Device compiler | nvcc from CUDA 12.8 / 13.2 (CI installs from NVIDIA redist zips, no installer) | `build.yml` |
| Torch | CUDA wheel, user-installed first (`--index-url .../cu128`) | README |
| Flags | cl: `/Ox /Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX`; nvcc: `-DWIN32_LEAN_AND_MEAN -DNOMINMAX -Xcompiler=/Zc:preprocessor`; no `-std` (torch adds it) | `setup.py` |
| Link | `libraries=["cublas"]` on Windows | `setup.py` |
| Python deps | `triton-windows` (marker `platform_system == 'Windows'`); import fails without it | `pyproject.toml`, README |
| Build | `python -m build -n --wheel`, `TORCH_CUDA_ARCH_LIST`, `EXLLAMA_EXT_COMPRESS=require` (loader limit on fat binaries, issue #413), `windows-2022` runner, pwsh, uv, sccache in front of nvcc | `build.yml` |
| User install | prebuilt wheel, or `pip install --no-build-isolation .` in a venv that has torch; else JIT on first import | README |

Upstream newer `setup.py` is more ROCm-aware than Kyojin's: `libraries = ["cublas"] if windows else ["hipblas"] if hip else []` plus `cuda_flags.use_rocm_sdk_devel()`. Kyojin's `setup.py` still has `["cublas"] if windows`, which is wrong for HIP. **[V]**

## 3. Kyojin state

Already Windows-capable (inherited, low risk) **[V]**: `ext.py` MSVC discovery, `setup.py` MSVC flags, `avx2_target.*`/`avx512_target.*` (`__linux__` guards + `__cpuidex`), `stloader.cpp` (`_fseeki64` branch), `ngram_gather_win.cpp`, `cuda_drv.cpp`, `cuda_host.cpp`, `moe_cpu_host.py` (`os.name == "nt"` branches), `util/shm.py`, `exllamav3/__init__.py` (skips expandable segments on win32).

Windows branches marked UNVERIFIED by the authors: `cpu/moe_mul1.cpp` CPU affinity branch ("no Windows toolchain was available"). **[V]**

ROCm build builds 72 TUs (after `ROCM_EXCLUDE_*` in `build_config.py`). Only these have POSIX/GCC constructs: `cpu/moe_mul1.cpp`, `cpu/moe_handoff_amdgpu.cu` (`__builtin_ia32_pause`, fine under clang), `hgemm.cu` (`sys/stat.h`, exists on MSVC). **[V]**

## 4. Linux-only / Linux-dependent inventory

### 4.1 Build and environment (must replace)

| Item | Where | Windows action |
|---|---|---|
| `build.sh`, `tools/strix_halo/rebuild.sh`, `env.sh` | repo root, `tools/strix_halo/` | New PowerShell script. Bash-only, `LD_PRELOAD`, `$HOME`, `gcc -print-libgcc-file-name`. |
| TRAP 1 `LD_PRELOAD libhsa-runtime64` | `env.sh` | N/A on Windows (Linux KFD/HSA path). Drop. Failure mode on Windows unknown **[?]**. |
| `--gcc-install-dir`, `HIPCC_COMPILE_FLAGS_APPEND` | `env.sh`, 14 `*_hip.py` | Replace with MSVC + Windows SDK discovery. hipcc.exe on Windows needs MSVC headers/libs for host side **[I]**. |
| `HIP_DEVICE_LIB_PATH`, `ROCM_HOME/HIP_PATH` | `env.sh` | Set from `rocm-sdk path --root`; layout differs on Windows **[I]**. |
| `libraries=["cublas"]` | `setup.py` | Change to `hipblas` when `torch.version.hip`. |
| `-Wno-register`, `-include format` in `extra_cflags` | `setup.py` | Passed to `cl.exe` too. Unknown option warning D9002 (harmless) or error; gate to non-Windows host. |
| hipify twin cleanup (`*.hip`, `*_hip.cuh`) | `rebuild.sh` | Port to Python (already Python inline) or drop. |
| `pip install --target` + copy `.so` | `rebuild.sh` | Output is `.pyd`; copy while loaded is blocked on Windows. |
| Docker, nix, `container.yml` | `docker/`, `flake.nix` | Not portable. Out of scope. |

### 4.2 Runtime Python

| Item | Where | Notes |
|---|---|---|
| `libamdhip64.so` lookup, `/proc/self/maps` | `util/hip_lib.py` | Windows DLL is versioned (`amdhip64_N.dll`) and lives in `_rocm_sdk_core\bin`; rewrite lookup. Used by every JIT HIP kernel via `hipModuleLoad`. |
| `hipcc` resolver | `util/hip_compiler.py` | Add `hipcc.exe`; `/opt/rocm` fallback n/a. Output `--genco` `.hsaco` is portable, so the JIT kernel binaries themselves are fine. |
| `~/.cache/exllamav3`, `XDG_CACHE_HOME`, `/usr/lib/gcc/...` | 14 `*_hip.py`, `ck_pagedkv.py` | `expanduser` works; GCC default path guarded by `isdir`. Low. |
| `/proc/meminfo`, `/proc/self/{smaps,pagemap,status}`, `MADV_POPULATE_READ` | `util/memory.py`, `util/memguard.py`, `modules/expert_store.py` | Linux-only memory accounting (swap-in, RAM-fit decisions). Needs psutil/Win32 equivalents or disable. Behavior-affecting. |
| `posix_fadvise`, `drop_caches`, `/proc/sys/vm` | `util/pagecache.py`, `util/memguard.py` | No Windows equivalent. Disable. |
| `O_DIRECT` aligned reads, pinned staging ring | `modules/expert_store.py` (DS41 SSD tier) | Windows needs `FILE_FLAG_NO_BUFFERING`. DS41 is experimental; recommend **out of scope**. |
| `os.preadv` | `modules/dsv41_fn/engram_io.py`, `dsv41/engram.py` | `preadv` not on Windows. DS41 only; out of scope. |
| `prctl(PR_SET_PDEATHSIG)` | `util/misc.py` | Already `sys.platform == "linux"` guarded. |
| `/dev/shm` | `util/shm.py` | Guarded (`os.name != "posix"`). |
| `/sys/class/drm/card1/.../mem_info_gtt_used`, `/proc/self/statm` | `tools/glm/serve.py:1233,1237` | Metrics only; wrap in try or skip. Check qwen/mimo servers too. |
| `tools/strix_halo/*.sh`, `ck_pagedkv` `.so` build, `libckpf.so` | `tools/strix_halo` | Benchmarks and optional CK paths. Out of scope. Falls back to Triton "silently". |
| System tuning (IOMMU off, `power_dpm_force_performance_level`, EC fan driver) | `doc/install.md` | Linux sysfs. Not applicable. Perf will differ. |

### 4.3 Hard blockers to verify

| Item | Evidence |
|---|---|
| **Triton on Windows + ROCm** | 52 files import triton, 43 at module level, incl. `vendor/fla/*`, `modules/router_topk.py`, `attention_fn/triton_paged*.py`, `gated_delta_net_fn/*`, `draft2s.py`, `fq_moe_prefill.py`, `util/measures.py`. AMD index: 0 Windows triton wheels (`whl-multi-arch/triton/`). `triton-windows` 3.9.0.post30: AMD only in PR #179 ("beginning to add support"). **[V]** Options: (a) wait for/contribute AMD `triton-windows`; (b) try woct0rdho PR build; (c) guard imports + fall back to torch path (`attention_fn/torch.py` exists, slow; whether FLA/GDN and the other kernels have a torch fallback was not checked **[?]**). |
| **GPU memory ceiling under WDDM** | Windows exposes unified memory via driver "Variable Graphics Memory" and shared GPU memory, not Linux GTT. 128 GB boxes may cap well under what the 95-105 GB packs need. Not verified; check on your machine: `torch.cuda.mem_get_info()` after init. **[?]** |
| **Compiling this device code with Windows hipcc** | Kyojin has heavy gfx1151-specific HIP (WMMA, `hip/hip_mma.cuh`, inline `__builtin_amdgcn_*`, ext_vector_type). clang accepts these on Windows. Risk is MSVC-ABI/header interaction, not syntax. 8-10 min on Linux per `doc/install.md`; expect longer. **[I]** |
| **Torch version skew** | Linux pin: torch `2.14.0+rocm10.1.0a20260822`. Windows on the same index/date: `2.15.0a0+rocm10.1.0a20260822` (cp310..cp314). There is no matching 2.14 for Windows. Nightly API drift risk. **[V]** |

## 5. Windows dependency set (what the script must check)

Available on AMD's nightly multi-arch index for win_amd64 **[V]**: `torch 2.15.0a0+rocm10.1.0a20260822`, `rocm-sdk-core`, `rocm-sdk-devel` (both `10.1.0a20260822`), `amd-torch-device-gfx1151` / `gfx115x`. Not present for Windows at this date: `triton`, `rocm-sdk-libraries-gfx1151`, `rocm`, `rocm-bootstrap` at `10.1.0a20260822` (device libs may come from the `amd-torch-device-*` wheel instead **[?]**; resolve in spike).

Preflight checks, each with a one-line fix hint:

1. Windows 11 x64 (build >= 22000); not ARM.
2. Latest AMD Adrenalin driver; `Get-CimInstance Win32_VideoController` shows Radeon 8060S / gfx1151. Driver version printout.
3. Python 3.12 x64 (cp312). Reject MS Store python and 32-bit.
4. Git, and `long paths` enabled (hipify/venv paths).
5. Visual Studio 2022 Build Tools with "Desktop development with C++" and a Windows 11 SDK. Find via `vswhere.exe`; launch via `VsDevCmd.bat -arch=amd64`. Offer `winget install Microsoft.VisualStudio.2022.BuildTools --override ...`.
6. Disk: `_rocm_sdk_devel` is about 12 GB expanded (Linux figure) plus torch plus build tree. Require >= 40 GB free.
7. RAM/jobs: `MAX_JOBS` default 6 (Linux note: unbounded ninja OOMs).
8. venv created; torch is ROCm (`torch.version.hip` non-empty), `torch.cuda.is_available()`, device arch `gfx1151`, 4x4 matmul passes.
9. `hipcc.exe --version` runs from `rocm-sdk path --root`; `hipsparse/` and `thrust/` headers present (same check as `build.sh`).
10. `import triton` works and a trivial kernel runs (reports clearly "Triton for AMD on Windows is not available" otherwise).

## 6. Build script POC (not started, needs your consent)

Shape: single `scripts/windows/build.ps1`, idempotent, happy path, each failed check prints cause + exact next command.

1. Preflight (section 5), no side effects.
2. Create `.venv`, install torch/rocm from `https://rocm.nightlies.amd.com/whl-multi-arch/` with pinned versions.
3. `rocm-sdk init`; set `ROCM_HOME/ROCM_PATH/HIP_PATH/HIP_DEVICE_LIB_PATH` from `rocm-sdk path --root`; enter VS dev shell.
4. `pip install --no-build-isolation --no-deps .` with `PYTORCH_ROCM_ARCH=gfx1151`, `MAX_JOBS`.
5. Smoke test: import `exllamav3_ext`, tiny GPU op, `exllamav3` import (this fails at Triton today).

Needed source changes (small, guarded): `setup.py` hipblas + MSVC-safe flags; `hip_lib.py`/`hip_compiler.py` Windows lookup; `moe_mul1.cpp` verify; `serve.py` metrics guard. Estimate: script about 250 lines PowerShell; source diff under 100 lines; iteration time dominated by 10+ min rebuilds on your box.

## 7. CI promotion

- Build-only job is feasible on `windows-2022`: no GPU needed to compile `--offload-arch=gfx1151`. Reuse the upstream pattern (`windows_smoke` target, pwsh, uv, VS pin, sccache). **[I]**
- Constraint: free runner disk (about 14 GB free on C:) versus a 12 GB SDK plus torch. May need the larger-runner tier or aggressive cleanup / D: drive. **[?]**
- Cannot run tests on hosted runners (no GPU). Wheel is a build-validation artifact; runtime validation stays manual on your machine.
- Kyojin `build.yml` has no ROCm or Windows-ROCm job today; add as new `target: windows_rocm_smoke`. Promote only after the local script passes.

## 8. Risk ranking

1. Triton absent on Windows ROCm (runtime blocker).
2. GPU memory ceiling / Windows unified-memory behavior (product-fit blocker).
3. hipcc.exe plus MSVC toolchain integration with this kernel set (build blocker, likely tractable).
4. Nightly torch version skew (2.14 Linux vs 2.15 Windows).
5. Linux memory-accounting code paths (`util/memory.py`, `memguard`) degrade behavior silently.
6. DS41 SSD tier (`O_DIRECT`, `preadv`): descope.

## 9. Questions for you

1. Is "runs end to end (serve a model)" required, or is "extension builds and imports" the POC exit? Triton decides which is reachable.
2. Which model for the first runtime test (Qwen3.8-Flash-Next 95 GB needs the memory ceiling answered)? A small EXL3 model would separate build from memory issues.
3. OK to descope DS41, CK paths and `tools/strix_halo/*.sh` benches for the POC?
4. Python 3.12 only, and torch 2.15 nightly (not 2.14) acceptable?
5. Can you run the section 1 spike (3 commands) and paste output? That resolves most `[?]` items.
