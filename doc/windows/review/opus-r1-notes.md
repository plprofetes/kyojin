# Review evidence notes (Windows gfx1151 POC), 2026-10-30

Repo under review: /home/user/kyojin @ cca1e1706c654cb3b729a643f01bd4eee431c7ea (branch claude/modest-rubin-p69b5g).
Upstream checkout: /tmp/claude-0/ext/exllamav3 @ cdc785db89e263e65e5849cb53e1ada156411414 (v1.6.5).
torch cpp_extension.py read from /tmp/claude-0/scratch/cpp_extension.py (torch 2.15.0a0+rocm10.1.0a20260822 win_amd64, 3340 lines).
External sources (SHA at read time):
- ROCm/TheRock main 682b3407528d7ee817a7bf092941070b50755ed7: RELEASES.md (sha256 73a698af...), docs/development/windows_support.md (d69526cd...), docs/development/wsl_rocdxg.md, external-builds/pytorch/build_prod_wheels.py (9dcb4b42...).
- ROCm/llvm-project amd-staging c3cd044af8b7ce9e72d6905a64a473e84ec6e7ee: amd/hipcc/src/hipBin_amd.h (5cc7d844...).
- llvm/llvm-project main c7a69e6fc6d40c220f58487039f05fe6b9f5d16c: clang/include/clang/Options/Options.td (the old path clang/include/clang/Driver/Options.td returned 0 bytes: moved).
- triton-lang/triton-windows release/3.8.x-windows 143426c4cbead9d3baf826b017ce1d3da5e51d3c (README there is upstream Triton's, no Windows/AMD text). Wheel read: triton_windows-3.8.0.post29-cp312-cp312-win_amd64.whl (PyPI, uploaded 2026-09-28).
- AMD index https://rocm.nightlies.amd.com/whl-multi-arch/ (listings fetched 2026-10-30); rocm-10.1.0a20260822.tar.gz sha256 a7160972...
- AMD WSL guide https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html (latest, fetched 2026-10-30).
Method for wheel contents: HTTP range reads of the zip central directory (/tmp/claude-0/review/scripts/rzip.py), no full download.

## Build (torch cpp_extension, Windows + HIP)
- [V] Host .cpp compile rule on Windows+HIP is `$cxx /showIncludes $cflags -c $in /Fo$out $post_cflags`, cxx = env CXX or `cl` (cpp_extension.py:493-498, 3206-3213). Default = cl.exe.
- [V] .cu/.hip go to `ROCM_HOME/bin/hipcc.exe` (ROCm >= 6.4; else hipcc.bat) (cpp_extension.py:2433-2437, 3176-3177); PYTORCH_NVCC overrides (3172).
- [V] Host post_cflags get COMMON_HIP_FLAGS prepended (-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2, plus -fms-runtime-lib=dll appended by _set_hipcc_runtime_lib, which mutates the global list) (397-401, 698-709, 1196-1198, 1214-1216, 3186-3187). With cl.exe, -fms-runtime-lib=dll -> D9002 warning only [I].
- [V] hipcc cuda_cflags = -std=c++20 + `-Xcompiler <flag>` pairs for every COMMON_HIP_FLAGS entry + pp_opts; cuda_post_cflags = COMMON_HIPCC_FLAGS(-fms-extensions -Wno-ignored-attributes on Windows) + COMMON_HIP_FLAGS + extra nvcc flags + --offload-arch (1219-1239, 1062-1067, 413-416).
- [V] clang ignores separate `-Xcompiler` (Options.td: `def : Flag<["-"], "Xcompiler">, IgnoredGCCCompat;`), so torch's pairs work; the joined form `-Xcompiler=/Zc:preprocessor` matches no option -> "unknown argument" [I from Options.td]. So setup.py gating of -Xcompiler= is needed/correct. hipcc does not translate -Xcompiler (no occurrence in hipBin_amd.h).
- [V] ext.py (JIT fallback) still passes `-Xcompiler=/Zc:preprocessor` and `-Ofast` to hipcc on Windows (exllamav3/ext.py:104 area); not gated. Any `import exllamav3` without a built .pyd triggers this JIT build.
- [V] No header dependency tracking for Windows+HIP: compile rule has no `deps = msvc` when IS_HIP_EXTENSION (3211-3213); hipcc rule depfile only for CUDA (3191-3198). Header edits do not trigger rebuilds -> use clean builds after header changes.
- [V] hipcc include paths: spaces replaced by backslash (1186-1188); paths with spaces (user name, Python install) are fragile [I].
- [V] Link: link.exe found via `where cl` (3286-3297) -> VS env needed even with clang-cl. Libraries: c10.lib c10_hip.lib torch_cpu.lib torch_hip.lib torch.lib torch_python.lib + /LIBPATH:ROCM_HOME/lib amdhip64.lib (2589-2656), plus setuptools `libraries` from setup.py.
- [V] ROCM_HOME discovery: ROCM_HOME/ROCM_PATH env, else `_rocm_sdk_core` package dir, else hipcc on PATH, else /opt/rocm (186-215). Upstream exllamav3 util/cuda_flags.py:118-150 explains _rocm_sdk_core lacks hipblas headers; point at devel tree.
- [V] TheRock builds PyTorch on Windows with CC=CXX=<rocm>/lib/llvm/bin/clang-cl.exe, HIP_CLANG_PATH=<rocm>/lib/llvm/bin, DISTUTILS_USE_SDK=1 (build_prod_wheels.py:699-706, 1204, 1347). Plan uses cl.exe: diverges from AMD's tested config.
- [V] HIP headers have MSVC host branches (amd_hip_fp16.h `#elif defined(__GNUC__) || defined(_MSC_VER)` -> hip_fp16_gcc.h; host_defines.h `_MSC_VER` branches) in _rocm_sdk_core/include (rocm_sdk_core-10.1.0a20260822 win_amd64). host_defines.h does NOT define `__restrict__` for MSVC.
- [V] bindings.cpp (host TU) includes quant/exl3_kernel_map.cuh (bindings.cpp:39, unguarded), whose typedef at line 53 expands EXL3_GEMM_ARGS containing `__restrict__` (lines 17-27). [I] cl.exe rejects `__restrict__` -> compile error. clang-cl accepts it.
- [V] hgemm.cu:161-164 uses getenv("HOME") and POSIX 2-arg mkdir(path, 0755) under USE_ROCM; [I] fails with MSVC UCRT headers (no 2-arg mkdir); HOME unset on Windows -> "/tmp/.cache".
- [V] hgemm.cu:109-849 (USE_ROCM) calls rocBLAS directly (rocblas_create_handle, rocblas_get_version_string, rocblas_gemm_ex_get_solutions; include rocblas/rocblas.h:124). setup.py Windows links only hipblas -> [I] LNK2019 unresolved rocblas_* (Linux tolerates undefined symbols in .so; Windows does not). Need "rocblas" too.
- [V] 72 ROCm TUs: 60 .cu + 12 .cpp (build_config.get_sources run on Linux).
- [V] `long` used in device/host code (84 non-`long long` uses in the ROCm TU include closure; 27 in .hip JIT sources, mostly dsv41/MoE). [I] clang HIP on Windows uses the host LLP64 data model, so `long` is 32-bit in device code too: silent overflow risk for offsets > 2^31.
- [V] cpu/moe_mul1.cpp has Windows branches (intrin.h, windows.h, Synchronization.lib, __forceinline); moe_handoff_amdgpu.cu uses __builtin_ia32_pause under __x86_64__ (clang defines it on win64 too).
- [V] ngram.cu POSIX parts guarded by _WIN32 (ngram.cu:6-8, 140-311); ngram_gather_win.cpp is the Windows port.

## Wheels / index (rocm.nightlies.amd.com/whl-multi-arch, date 20260822)
- [V] win_amd64 present: torch 2.11.0, 2.12.0, 2.13.0, 2.15.0a0 (+rocm10.1.0a20260822); 2.14.0 Linux only. Linux also has 2.15.0a0+rocm10.1.0a20260822 (cp312) -> the Windows torch can be tested on Linux.
- [V] torch 2.15.0a0 win METADATA: Requires-Dist rocm[libraries]==10.1.0a20260822, rocm-bootstrap, extra device-gfx1151 -> amd-torch-device-gfx1151 + amd-torch-device-gfx115x ==2.15.0a0+rocm10.1.0a20260822. No triton requirement.
- [V] Present (refutes assessment §5 "not present for Windows"): rocm-10.1.0a20260822.tar.gz (sdist, platform independent), rocm_bootstrap-0.1.0-py3-none-any.whl, rocm_sdk_libraries-10.1.0a20260822-py3-none-win_amd64.whl, rocm_sdk_device_gfx1151-10.1.0a20260822 win_amd64, amd_torch_device_gfx1151-2.15.0a0 win cp310..cp314. No Windows `triton` on the AMD index (Linux triton-3.8.0+gitdf3f91dd.rocm10.1.0a20260820 only).
- [V] rocm_sdk_core win wheel: bin/amdhip64_7.dll, bin/hipcc.exe, bin/hipcc.bat, bin/hipInfo.exe, bin/hipconfig.exe, lib/amdhip64.lib, lib/hiprtc.lib, lib/llvm/bin/{clang.exe,clang++.exe,clang-cl.exe,lld-link.exe,ld.lld.exe,amdgpu-arch.exe}, lib/llvm/amdgcn/bitcode/*.bc, include/hip/*.
- [V] rocm_sdk_libraries win wheel (57 entries): bin/hipblas.dll, bin/libhipblaslt.dll, bin/rocblas.dll; no .lib import libraries in it.
- [V] rocm_sdk_devel win wheel = rocm_sdk_devel/_devel.tar (1,434,736,640 bytes) + CLI; content not listed (would need a sequential tar scan). [?] whether hipblas.lib / rocblas.lib are in it.
- [V] rocm sdist rocm_sdk/_dist_info.py:247-288: Windows DLL patterns amdhip64*.dll, *hipblas*.dll, *hipblaslt*.dll, hiprtc0*.dll; windows_relpath = "bin". `rocm_sdk.find_libraries("amdhip64")` is the supported lookup (used by triton-windows).
- [V] pyproject.toml rocm extra pins torch 2.14.0 (Linux) and triton==3.8.0+git675c5987.rocm10.1.0a20260822; pyproject line 35 has unpinned `triton-windows; platform_system == 'Windows'`.

## triton-windows
- [V] triton_windows-3.8.0.post29 cp312 win_amd64 wheel contains triton/backends/amd/{compiler.py,driver.py,driver.c,lib/ocml.bc,ockl.bc} and HIP headers: AMD backend is shipped.
- [V] backends/amd/driver.py: HIP lib lookup via TRITON_LIBHIP_PATH, then rocm_sdk.find_libraries("amdhip64"), HIP_PATH, ROCM_PATH/ROCM_HOME; Windows name amdhip64.dll (driver.py ~644-846).
- [V] triton/runtime/build.py: on Windows prefers <platlib>/_rocm_sdk_core/lib/llvm/bin/clang-cl.exe (comment: "handles HIP C headers that mix C/C++ constructs"), else MSVC via find_msvc_winsdk, else bundled tcc; needs MSVC + Windows SDK include/lib dirs (build.py:22-70, 173-176).
- [?] "RDNA3.5 supported" claim: could not read the triton-windows Windows README (github.com HTML blocked by proxy for unattached repos; raw README on main and release/3.8.x-windows is upstream Triton's). Assessment's [V] on this is not reproducible from here. Dead end: gh api -> 403 (repo not attached); PyPI description empty.
- [V] PyPI triton-windows latest 3.9.0.post30 (abi3, 2026-10-28), 3.8.0.post29 (2026-09-28). requires_python >=3.11,<3.16 (from scratch tw.json).
- [V] Kyojin ships only exllamav3/kernel_seed/triton-3.7.0-gfx1151.tar.xz; seed keyed by triton.__version__ (util/kernel_seed.py:44-53): no seed with 3.8 on either OS -> all Triton kernels JIT on first start.

## Runtime
- [V] util/hip_lib.py: only libamdhip64.so names + /proc/self/maps; on Windows find_hip_runtime returns None (open raises OSError, caught) -> load_hip_runtime raises OSError. Users: modules/{attention_fn/qdec_hip,attention_fn/qsa_prefill_hip,gr_fn/gr_mix_hip,ple_fn/ple_hip,dsv41_fn/*}.py and vendor/fla/hip/{gdn_fused_h,gdn_hf,kda_fused_h,kda_hf}_hip.py.
- [V] util/hip_compiler.py: EXL3_HIPCC override exists; `_rocm_sdk_devel/bin/hipcc` (no .exe) check fails on Windows, but shutil.which("hipcc") honours PATHEXT -> hipcc.exe found if SDK bin on PATH [I]. --gcc-install-dir guarded by os.path.isdir in every module (e.g. gr_mix_hip.py:19-20). ~/.cache via expanduser works. => hip_compiler needs no code change for POC if EXL3_HIPCC is set.
- [V] JIT loader pattern: ctypes hipModuleLoad(byref(mod), path.encode()) + hipModuleLaunchKernel on torch.cuda.current_stream().cuda_stream (gr_mix_hip.py:31-49). Must load the SAME amdhip64_7.dll torch loaded [I]; bytes path breaks on non-ASCII user names [I].
- [V] serve.py SERVE_ENV (tools/qwen/serve.py:512-600) setdefault-enables HIP JIT paths: EXL3_PLE_HIP, EXL3_DQ_HIP, EXL3_GR_HIP, EXL3_GDN_FUSE, EXL3_GDN_HF, EXL3_PF_GR_FUSE, EXL3_QSA_ROWINV, etc. User-set env wins (setdefault, serve.py:723-726).
- [V] gdn_hf_hip shape gate: B==1, K==V==128, HV%H==0 else falls back to Triton/HIP chain with a warning (vendor/fla/hip/gdn_hf_hip.py:43-60); load failure sets fail flag then raises (66-70).
- [V] CUDA/HIP graphs default off: EXL3_CUDA_GRAPH default "0" (model/cuda_graph.py:62), EXL3_BLOCK_GRAPH default "0" (modules/block_graph.py:37). CUDA_SERVE_ENV enables graphs only on CUDA.
- [V] hipHostRegister only in TP and CPU-MoE paths (model/model_tp_*.py, moe_cpu_host.py); stloader.cpp:344 uses cudaMallocHost (pinned) on every load.
- [V] No cooperative launch, managed memory, IPC, HSA or /dev/kfd use in the ROCm TU set outside parallel/ (grep).
- [V] exllamav3/__init__.py:27-35 skips expandable_segments on win32.
- [V] vendor/fla imports nothing from exllamav3 at module level (only kda_inter_split.py:201 inside a function) -> `sys.path.insert(0, r"exllamav3\vendor"); import fla` tests the real GDN Triton kernels without the extension. `import exllamav3.vendor.fla` instead triggers exllamav3/__init__ -> ext.py -> JIT build.

## serve.py on Windows
- [V] tools/qwen/{image_cache,serve_metrics,serve_ping,serve_reqlog}.py are git symlinks (mode 120000; 15 symlinks repo-wide, `git ls-files -s`). serve.py:43-53 imports them. [I] Default Git for Windows checkout (core.symlinks=false) writes text files -> SyntaxError at import.
- [V] tools/startup_health.py:269 calls os.register_at_fork (Unix only) -> AttributeError on Windows when the early listener starts (startup_health.start, serve.py main).
- [V] startup_health.check_model_dir requires config.json + chat_template.jinja in the pack (startup_health.py:288-295); generation_config.json optional (serve.py:424-431 FALLBACK_DEFAULTS). Template rendered from tools/qwen/chat_template.jinja by default (serve.py:80, 2271).
- [V] serve.py flags --no-vision, --draft-policy off, -c exist (serve.py:2348-2368).
- [V] swap_in_process "never raises" (util/memory.py:660-672): try/except around ctypes.CDLL(None) and /proc reads.
- [V] doc/roc_hip_decode.md:5-25 describes the gfx1200/gfx1201 decode backend; "non-gfx12 ROCm devices" fall back (line 25) -- possibly stale vs gfx1151 GEMV code (exl3_gemv.cu mentions gfx1151 at 82,181,400,691,845). The 54 tok/s Qwen3.5-9B number is gfx1201, not gfx1151.

## Orchestrator
- [V] `uv run scripts/windows/kyojin_win.py ...` inside the repo syncs the project env first: test in a copy with pyproject.toml + uv.lock: `uv run --offline scripts/s.py` -> "Creating virtual environment at: .venv" then tries to install exllamav3 deps (torch 2.14.1 + cuda-toolkit from PyPI on Linux). kyojin_win.py has no PEP 723 header; pyproject [tool.uv] package = true. Fix: `uv run --no-project` or a `# /// script` block, or call python directly.
- [V] uv keeps an installed prerelease that satisfies a plain specifier (test: packaging 26.0rc3 installed, `uv pip install "packaging>=20.0"` -> "Checked 1 package", unchanged). So setup step 2 (requirements.txt has torch>=2.6.0) will not replace ROCm torch.
- [V] kyojin_win.py is 1056 lines (assessment §6.2 says ~300; journal/brief says ~900). Assessment §6 status line says "Not committed": stale (commits 2ff5c02 etc.).
- [V] kyojin_win serve model_dir check WARNs on missing generation_config.json (not needed) but is right about chat_template.jinja.

## WSL2
- [V] AMD: ROCDXG (librocdxg) WSL production support with Adrenalin 26.2.2 + ROCm 7.2.1, "first time AMD is supporting Ryzen Strix and Strix Halo SKUs" (AMD WSL how-to, latest). Ubuntu 22.04/24.04 only.
- [V] TheRock RELEASES.md: WSL section commented out ("TODO: include WSL documentation once we are satisfied with quality in nightly releases"); DXG detection default on since ROCm 7.13; needs `amdrocm-wsl` package.

## Upstream / contribution
- [V] Upstream setup.py:80 `libraries = ["cublas"] if windows else ["hipblas"] if hip else []` -> upstream does not support HIP on Windows either; assessment "cf. upstream" is about Linux HIP only.
- [V] Upstream build.yml has windows / windows_smoke targets (CUDA 12.8.1 only).
- [V] Kyojin CONTRIBUTING.md: one change per PR, explain how checked, no new deps without reason, commit subjects imperative with no prefix; changes that belong to ExLlamaV3 (CUDA path, shared logic) go upstream first; "Other ROCm GPUs" are wanted; no mention of Windows. README badges: gfx1151 | ROCm 7, Python 3.12. CI: build.yml is workflow_dispatch only; nix.yml and container.yml run on PRs (Linux).
- [V] Branch commit subjects use "Windows port:"/"Windows POC:" prefixes (git log), against CONTRIBUTING.

## Dead ends
- github.com HTML and gh api for triton-lang/triton-windows: blocked (proxy/unattached repo 403). raw.githubusercontent works.
- clang Options.td old path (clang/include/clang/Driver/Options.td) now empty; new path clang/include/clang/Options/Options.td.
- Listing rocm_sdk_devel _devel.tar contents remotely: not attempted (1.4 GB tar inside zip; needs sequential header scan).
- Hugging Face: not tried (known proxy 403).
- No web source found on whether torch.cuda.mem_get_info on Windows reports VGM dedicated only or dedicated+shared.
