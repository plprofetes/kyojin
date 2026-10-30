# Windows port assessment (gfx1151, Win11 x64)

Status: assessment only, no code changed. Date: 2026-10-30. Rev 2: Triton finding corrected, decisions and tooling added. Kyojin 1.8.3 vs upstream exllamav3 1.6.5.
Target: one configuration. Windows 11 x64, Strix Halo (gfx1151), latest Adrenalin driver, Python 3.12, ROCm wheels from AMD's pip index.

Plan, tasks and session journal: [README.md](README.md). Deferred items: [deferred.md](deferred.md).

Legend: **[V]** verified from sources or the AMD wheel index this session; **[I]** inferred, needs a Windows run; **[?]** unknown.

## 1. Verdict

- **Build the extension on Windows: plausible, small delta.** The C++ host code is already mostly Windows-aware (inherited from upstream). Torch's own `cpp_extension.py` in AMD's Windows ROCm wheel has HIP-on-Windows paths (`hipcc.exe`, MSVC runtime flag for hipcc). **[V]**
- **Run the engine on Windows: Triton is available, memory is the gate.** 52 files import `triton`, 43 at module level. Rev 1 said Triton was blocked; that was stale. `triton-windows` (now `triton-lang/triton-windows`) lists RDNA3.5 (gfx115x, Strix Halo) as supported, uses `clang-cl` found automatically from the TheRock wheel, and 3.8.0.post29 / 3.9.0.post30 are on PyPI. **[V]** (README + releases pages.) Kyojin's Linux pin is triton 3.8.0, so start with `triton-windows<3.9`. Whether Kyojin's kernels (GDN/FLA, paged attention) compile and match on Windows is unverified **[?]**.
- **Memory model is a second unknown.** The product premise is 100 GB-class models in 128 GB. Windows WDDM caps GPU-addressable memory differently from Linux GTT, and the Kyojin tiering code assumes Linux memory semantics. **[?]**
- Recommendation: the build script itself doubles as the spike. Its `doctor` stage (section 6) runs (a) torch+ROCm matmul on gfx1151, (b) a one-file `CUDAExtension` compile, (c) a Triton kernel, so the first run on your box answers the open `[?]` items.

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
| **Triton on Windows + ROCm** (downgraded from blocker) | 52 files import triton, 43 at module level, incl. `vendor/fla/*`, `modules/router_topk.py`, `attention_fn/triton_paged*.py`, `gated_delta_net_fn/*`. AMD's index has no Windows `triton` wheel, but PyPI `triton-windows` covers RDNA3.5 with `clang-cl` from the TheRock wheel (the old `woct0rdho/triton-windows` README still says "PR #179"; it is stale, development moved to `triton-lang/triton-windows`). **[V]** Residual risk: Kyojin Triton kernels untested on Windows. **[?]** |
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
10. `import triton` works and a trivial `@triton.jit` kernel runs on the GPU (needs `clang-cl` from the ROCm SDK, found automatically per triton-windows).

## 6. Build tooling choice and POC shape

Status: `scripts/windows/kyojin_win.py` and `test_kyojin_win.py` written and unit-tested on Linux (25 tests); `doctor` dry-run on Linux works end to end. Never run on Windows. `setup.py` Windows+HIP edits made. Not committed.

### 6.1 PowerShell vs Python

Use **Python for the orchestrator, with a tiny bootstrap only to get `uv`**. The "small build system" already exists and stays: setuptools + torch `BuildExtension` + ninja drives the compile (`setup.py`); adding CMake/meson would duplicate it and fight torch's hipify/arch handling. What is missing is orchestration (preflight, venv, env capture, smoke test), not a compiler driver.

Why not PowerShell for the logic:
- Native command failures do not raise by default (`$ErrorActionPreference = "Stop"` ignores non-zero exit codes from `uv`/`curl`). The community gist in section 7 hits exactly this, plus a lexicographic wheel pick.
- Hard to unit test (Pester only); the repo is Python and its `tests/` suite already runs without a GPU.
- Cannot be exercised on Linux, so I could not test it before you do.

Why Python (stdlib only, one file, `argparse` subcommands):
- Each check is a function returning `(ok, detail, fix_hint)`; mockable, so I can unit-test the logic here with pytest (fake `vswhere`, fake driver) before it touches your machine.
- `subprocess.run(check=True)` fails loudly; paths via `pathlib`; no quoting traps.
- VS environment capture is a well-trodden pattern: `vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`, then `cmd /c "VsDevCmd.bat -arch=amd64 && set"` merged into the child environment. (setuptools already does this for `cl.exe`; hipcc/clang-cl also needs `INCLUDE`/`LIB`.)
- **uv** (single exe) as env manager: `uv python install 3.12`, `uv venv`, `uv pip install`. It brings its own Python, so nothing touches the system Python or PATH, which is the "venv, don't fight paths" requirement. Bootstrap is two lines: `winget install astral-sh.uv`, then `uv run scripts/windows/kyojin_win.py doctor`.
- Task-runner layers (nox, invoke, doit) are not worth it for five subcommands.

### 6.2 Shape

`scripts/windows/kyojin_win.py` (about 300 lines), subcommands, all idempotent, happy path, each failure prints the cause and the exact next command:

1. `doctor`: preflight (section 5), no side effects. Also reports free RAM for the compile.
2. `setup`: `uv venv .venv-win` (Python 3.12), install torch/rocm from the AMD index with pinned versions, `triton-windows<3.9`, `requirements.txt`; `rocm-sdk init`; `rocm-sdk test`.
3. `build`: capture VS env, set `ROCM_HOME/ROCM_PATH/HIP_PATH/HIP_DEVICE_LIB_PATH` from `rocm-sdk path --root`, `PYTORCH_ROCM_ARCH=gfx1151`, `MAX_JOBS` from available RAM, then `pip install --no-build-isolation --no-deps .` into the venv.
4. `check`: import `exllamav3_ext`, 512x512 fp16 matmul, one Triton kernel, `import exllamav3`.
5. `serve`: runs `tools/qwen/serve.py` with the flags for the small test model (6.3).

Source changes needed (small, guarded): `setup.py` (hipblas, MSVC-safe flags), `util/hip_lib.py` and `util/hip_compiler.py` (Windows lookup), `moe_mul1.cpp` verify, Linux-only metrics in `tools/qwen/serve.py` (check; `tools/glm/serve.py` reads `/sys/class/drm` and `/proc/self/statm`). Estimate: source diff under 100 lines; iteration time dominated by 10+ minute rebuilds on your box.

### 6.3 First model for the POC

Serve Qwen (decision 1) with a small pack so RAM stays free for the compiler. Recommended: **`turboderp/Qwen3.5-9B-exl3`**, about 4 bpw (roughly 5-6 GB), `-c 8192`, `--no-vision --draft-policy off`.
- Same `qwen3_5` architecture family Kyojin implements (`exllamav3/architecture/qwen3_5.py`) and the model Kyojin's own ROCm decode benchmarks use (`doc/roc_hip_decode.md`, 54 tok/s on gfx1201), so the GEMV path is exercised.
- It has Gated DeltaNet layers, so it also smoke-tests the Triton FLA kernels (the main Windows runtime risk).
- Caveats: the repo name comes from a Hugging Face likes page and was not opened (HF is not reachable from my sandbox) **[?]**. `tools/qwen/serve.py` is documented for Qwen3.8-Flash Yamz packs and wants `chat_template.jinja` + `generation_config.json` in the pack, so a stock turboderp pack may need those copied from the base model **[?]**. Fallback: a turboderp Qwen3.8-27B quant at about 2 bpw, or the repo's `tests/test_hip_gemv_decode.py` with `EXL3_TEST_MODEL`.

## 7. CI promotion (and prior art below)

- Build-only job is feasible on `windows-2022`: no GPU needed to compile `--offload-arch=gfx1151`. Reuse the upstream pattern (`windows_smoke` target, pwsh, uv, VS pin, sccache). **[I]**
- Constraint: free runner disk (about 14 GB free on C:) versus a 12 GB SDK plus torch. May need the larger-runner tier or aggressive cleanup / D: drive. **[?]**
- Cannot run tests on hosted runners (no GPU). Wheel is a build-validation artifact; runtime validation stays manual on your machine.
- Kyojin `build.yml` has no ROCm or Windows-ROCm job today; add as new `target: windows_rocm_smoke`. Promote only after the local script passes. With the Python orchestrator, CI just calls `uv run scripts/windows/kyojin_win.py build`, so local and CI share one code path.

## 7b. Prior art (reuse, do not rebuild)

- **TheRock** [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md): the supported install pattern on Windows. `pip install --index-url <index> "torch[device-gfx1151]"`, `rocm-sdk init`, `rocm-sdk test`, `hipInfo.exe` as GPU check, Adrenalin driver as the only stated prerequisite. Its index host is `https://nightly.repo.amd.com/rocm/whl-next/`; Kyojin docs and my checks use `https://rocm.nightlies.amd.com/whl-multi-arch/`. Keep the index URL as one constant and verify equivalence in `doctor` **[?]**.
- **AMD docs, Windows PyTorch**: [Radeon/Ryzen install page](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html) and [ROCm 7.9.0 PyTorch guide](https://rocm.docs.amd.com/en/docs-7.9.0/install/pytorch-comfyui.html): stable (non-nightly) wheels from `repo.amd.com/rocm/whl/gfx1151/`, driver minimum, Python 3.12. Fallback if nightlies break; Kyojin is verified only on nightlies.
- **triton-windows**, [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows): RDNA3.5 supported, clang-cl from the TheRock wheel, `pip install "triton-windows<3.9"`. Do not build Triton ourselves.
- **Upstream exllamav3 CI**: VS Build Tools pin, `windows-2022`, sccache, `windows_smoke` target. Reuse the shape for the CI job.
- **Community PowerShell gist**, [kundeng](https://gist.github.com/kundeng/7ae987bc1a6dfdf75175f9c0f0af9711): uv + Python 3.12 venv + GPU smoke test is the right idea, but it pins ROCm 6.5.0rc wheels for gfx110x, never checks gfx1151 or VS Build Tools, and has the exit-code gaps noted in 6.1. Ideas only (uv-managed Python 3.12 venv + GPU smoke test); no code is used, and no licence was found for the gist, so none may be copied.
- **Not found**: any public guide to building PyTorch C++/HIP extensions with hipcc on Windows. We are on our own for build step 3, so it is the first thing to test.

## 7c. Sources and licences

Rule for this port: copy no third-party code unless its licence is recorded here with the copied lines. Nothing below was copied; "used" means a documented command, flag or fact was followed, or a tool is invoked.

| Source | Licence (checked) | How it was used |
|---|---|---|
| Kyojin (this repo): `tools/strix_halo/env.sh`, `rebuild.sh`, `setup.py` | MIT, (c) 2025 Turboderp (`LICENSE`) | Read. `kyojin_win.py` re-implements the build-time ROCm variables and the `EXL3_HIP_DEFINES` default for a Windows SDK layout. `setup.py` edited in place. |
| Upstream exllamav3 v1.6.5: `setup.py`, `ext.py`, `build.yml` | MIT, (c) 2025 Turboderp | Read. Only borrowed fact: link `hipblas` instead of `cublas` for ROCm (one line in `setup.py`, same project lineage). MSVC discovery and CI shape are described, not copied. |
| PyTorch `torch/utils/cpp_extension.py` (2.15.0a0+rocm10.1.0a20260822 Windows wheel) | BSD-3-Clause (PyTorch; licence file not re-read, standard for the project) | Read in a scratch dir outside the repo to confirm HIP-on-Windows code paths. Not copied. |
| ROCm TheRock `RELEASES.md` | MIT, (c) AMD (`LICENSE` fetched) | Documented install pattern followed (`torch[device-gfx1151]`, `rocm-sdk init/test/path`). |
| triton-lang/triton-windows | `LICENSE` fetched: MIT-style header (Tillet, OpenAI) | Runtime dependency from PyPI only. Support claims from its README and releases page. |
| astral-sh/uv | MIT (`LICENSE-MIT` fetched; dual MIT/Apache-2.0) | Tool invoked. |
| Microsoft `vswhere.exe`, `VsDevCmd.bat` | Microsoft tools, not redistributed | Invoked from the user's own Visual Studio install; documented usage only. |
| AMD docs (Radeon/Ryzen Windows PyTorch, ROCm 7.9.0) | Documentation | Read for the stable-wheel fallback index. |
| Community gist (kundeng) | **No licence identified** (could not retrieve metadata) | Ideas only. No code reused. |
| Third-party web pages from search results (for example a "Claude skill" guide for Strix Halo) | Unknown | Not relied on. |

Verification status: AMD index contents and the torch wheel `METADATA` were read directly over HTTP from a Linux sandbox. Hugging Face was unreachable from the sandbox (proxy 403 at every check), so model repo names remain unconfirmed.

## 8. Risk ranking

1. Kyojin Triton kernels (GDN/FLA, paged attention) untested on triton-windows AMD; runtime correctness risk.
2. GPU memory ceiling / Windows unified-memory behavior (product-fit blocker).
3. hipcc.exe plus MSVC toolchain integration with this kernel set (build blocker, likely tractable).
4. Nightly torch version skew (2.14 Linux vs 2.15 Windows).
5. Linux memory-accounting code paths (`util/memory.py`, `memguard`) degrade behavior silently.
6. DS41 SSD tier (`O_DIRECT`, `preadv`): descope.

## 9. Decisions (2026-10-30) and open items

Decided by you:
1. POC target: serve a Qwen model through `tools/qwen/serve.py`.
2. Descope DS41, CK paths, `tools/strix_halo/*.sh`; a small model is fine so RAM remains for compiling.
3. Python 3.12 and torch 2.15 nightly accepted, everything inside a venv (uv-managed).
4. Small model recommended: section 6.3.

Open, need your input or a run on the box:
- OK with a Python orchestrator + uv bootstrap (6.1) instead of PowerShell?
- Windows GPU-addressable memory on your 128 GB box (BIOS/Adrenalin "Variable Graphics Memory" setting). The 9B test avoids this; Qwen3.8-Flash-Next (95 GB) later does not.
- Hugging Face download of the model is on your side (not reachable from my sandbox).
