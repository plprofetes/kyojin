# Expected failure messages for the Windows gfx1151 POC (for `KNOWN` in scripts/windows/kyojin_win.py)

Repo read at c4a9f87 (KNOWN unchanged since cca1e17). Search the step output case-insensitively, as `diagnose()` does.
[V] = the message text is quoted from the tool's source at the given SHA/line. [I] = expected wording, not seen.
Diagnostics format matters for the regexes ([V] llvm-project c7a69e6 clang/lib/Driver/ToolChains/Clang.cpp:9135-9137):
clang-cl adds `-fdiagnostics-format msvc`, so host .cpp errors look like `C:\k\x.cuh(53,21): error: ...`; hipcc.exe drives
clang.exe in GNU mode, so device-file errors look like `C:\k\x.cu:163:5: error: ...`. cl.exe errors look like
`x.cuh(53): error C2146: ...`.

## New signatures (predicted failures)

| id | Where it comes from | Expected text | Regex | |
|---|---|---|---|---|
| distutils-use-sdk | torch cpp_extension.py:1341-1345 (`_check_abi`), raised when VsDevCmd ran but DISTUTILS_USE_SDK is unset. kyojin_win `build` captures the VS env and does not set it (fixed by patches/kyojin_win_sdk_env_c4a9f87.patch) | `UserWarning: It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.This may lead to multiple activations of the VC env.Please set \`DISTUTILS_USE_SDK=1\` and try again.` | `VC environment is activated but DISTUTILS_USE_SDK is not set` | [V] |
| where-cl | torch cpp_extension.py:3286-3297: `subprocess.check_output(['where', 'cl'])` while writing build.ninja, so cl.exe must be on PATH even when CXX=clang-cl | `INFO: Could not find files for the given pattern(s).` then `subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.` | `Command '\['where', 'cl'\]' returned non-zero` | [V] code / [I] `where` text |
| restrict-cl | bindings.cpp:39 -> quant/exl3_kernel_map.cuh:17-27,53 (`__restrict__`), host TU compiled by cl.exe (CXX unset) | `exl3_kernel_map.cuh(53): error C2146: syntax error: missing ')' before identifier 'A'` and/or `error C2061: syntax error: identifier '__restrict__'` | `exl3_kernel_map\.cuh\(\d+\): error C2(146\|061\|143)\|identifier '__restrict__'` | [I] (exact C-number unknown) |
| mkdir-2arg | hgemm.cu:163-164 (POSIX 2-argument mkdir), compiled by hipcc/clang with MSVC UCRT headers (fixed by patches/exllamav3_exllamav3_ext_hgemm_cu.patch) | `hgemm.cu:163:5: error: use of undeclared identifier 'mkdir'`, or with <direct.h> in scope `error: no matching function for call to 'mkdir'` + `note: candidate function not viable: requires single argument` | `hgemm\.cu:\d+:\d+: error: (use of undeclared identifier 'mkdir'\|no matching function for call to 'mkdir')` | [I] |
| unresolved-rocblas | hgemm.cu:109-849 calls rocblas_* (rocblas_create_handle, rocblas_get_version_string, rocblas_gemm_ex_get_solutions); setup.py linked only hipblas (fixed by patches/setup_py.patch). The link is done by setuptools' MSVCCompiler (link.exe), not ninja | `hgemm.obj : error LNK2019: unresolved external symbol rocblas_create_handle referenced in function ...`, `fatal error LNK1120: N unresolved externals`, then `error: command '...\link.exe' failed with exit code 1120` | `LNK2019: unresolved external symbol (__imp_)?(rocblas_\|hipblas)\w*` | [I] (LNK format [V] general) |
| unresolved-other | same mechanism for any symbol Linux resolved through libtorch_hip.so transitively | `error LNK2019: unresolved external symbol <name> referenced in function` | `LNK2019: unresolved external symbol (\S+)` (print group 1) | [I] |
| missing-import-lib (replace) | link.exe cannot find a .lib named in `libraries` / torch's ldflags | `LINK : fatal error LNK1181: cannot open input file 'rocblas.lib'` | `LNK1181: cannot open input file '?([\w.-]+\.lib)` | [I] wording, LNK1181 is the documented code |
| missing-crt-lib | LIB not set (no VS env) | `LINK : fatal error LNK1104: cannot open file 'msvcprt.lib'` (or `kernel32.lib`, `python312.lib`) | `LNK1104: cannot open file '(msvcprt\|msvcrt\|kernel32\|ucrt\|vcruntime\|python3\d+)\.lib'` | [I] |
| missing-win-headers (replace) | clang-cl/clang without INCLUDE | `C:\...\hip_runtime.h(..): fatal error: 'cstdint' file not found` or `fatal error: 'stdio.h' file not found` | `fatal error: '(windows\.h\|vcruntime\.h\|type_traits\|stdio\.h\|stdlib\.h\|cstdint\|cmath\|cstddef)' file not found\|C1083: Cannot open include file: '(windows\.h\|type_traits\|stdio\.h\|vcruntime\.h)'` | [I] (clang wording [V] pattern) |
| no-rocm-device-lib (replace `device-libs`) | clang cannot find the amdgcn bitcode | `clang: error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library` | `cannot find ROCm device library` | [V] ROCm/llvm-project c3cd044 clang/include/clang/Basic/DiagnosticDriverKinds.td:71-75 |
| no-hip-runtime | clang cannot find HIP headers/runtime (HIP_PATH/--rocm-path wrong) | `clang: error: cannot find HIP runtime; provide its path via '--rocm-path', or pass '-nogpuinc' to build without HIP runtime` | `cannot find HIP runtime` | [V] same file :76-78 |
| flag-leak (keep, tighten) | nvcc-style flag reaching clang (ext.py JIT on Windows, or a new setup.py flag) | `clang: error: unknown argument: '-Xcompiler=/Zc:preprocessor'` (or `unknown argument '...'; did you mean '...'?`) | `clang(\.exe)?: error: unknown argument:? '([^']+)'` | [V] DiagnosticDriverKinds.td:348-350 |
| ninja-summary | any compile failure; pip ninja 1.11 prints `FAILED: <outputs>`, ninja master `FAILED: [code=N] <outputs>` | `FAILED: [code=1] C:/k/build/temp.win-amd64-cpython-312/Release/exllamav3/exllamav3_ext/hgemm.obj`, then the command, then compiler output; at the end `ninja: build stopped: subcommand failed.` (default) or `ninja: build stopped: cannot make progress due to previous errors.` (with `-k 0`) | per failure: `^FAILED: (?:\[code=\d+\] )?(\S+)`; end: `ninja: build stopped: (subcommand failed\|cannot make progress due to previous errors)` | [V] ninja-build/ninja 3ccb227 src/status_printer.cc:244, src/build.cc:838-840, src/ninja.cc:1793 |
| uv-project-sync | `uv run scripts/windows/kyojin_win.py` without `--no-project` in the repo (pyproject [tool.uv] package=true) | `Creating virtual environment at: .venv`, then resolution/installs of the project's deps (torch from PyPI) and a build of exllamav3 | `Creating virtual environment at: \.venv\b` | [V] reproduced with uv 0.11.32 on Linux. Better: kyojin_win checks `Path(sys.prefix).name == ".venv"` at start and exits with the fix |
| symlink-checkout | tools/qwen/{image_cache,serve_metrics,serve_ping,serve_reqlog}.py are git symlinks (mode 120000); with core.symlinks=false Git for Windows writes a text file holding the target path | `File "C:\k\tools\qwen\image_cache.py", line 1` / `../strix_halo/image_cache.py` / `SyntaxError: invalid syntax` | `tools[\\/]qwen[\\/](image_cache\|serve_metrics\|serve_ping\|serve_reqlog)\.py", line 1` | [I] text; better: a doctor check that each of the 4 files is not a <100-byte file starting with `../` |
| register-at-fork | tools/startup_health.py:269 (fixed by patches/tools_startup_health_py.patch) | `AttributeError: module 'os' has no attribute 'register_at_fork'` | `module 'os' has no attribute 'register_at_fork'` | [V] Python attribute-error format |
| hip-lib-windows | exllamav3/util/hip_lib.py:64 before patches/hip_lib_windows.patch; first JIT HIP kernel (gdn_fused_h / gdn_hf / qdec / ple under SERVE_ENV) | `OSError: no libamdhip64.so or libamdhip64.so.<version> in the torch lib directory` | `no libamdhip64\.so or libamdhip64\.so\.<version>` | [V] hip_lib.py:64 |
| ext-jit-windows | patched ext.py (patches/exllamav3_ext_py.patch) when the .pyd is missing | `ImportError: exllamav3_ext is not built. On Windows with ROCm torch the JIT build is not supported: ...` | `exllamav3_ext is not built\. On Windows with ROCm` | [V] (patch text) |
| dll-missing (keep, add text) | `import exllamav3_ext` cannot resolve a dependent DLL | `ImportError: DLL load failed while importing exllamav3_ext: The specified module could not be found.` | keep current; add `DLL load failed while importing (\w+)` to print the module | [V] CPython message format |
| entrypoint (fix) | Python wording for 0xC0000139 / WinError 127 | `ImportError: DLL load failed while importing exllamav3_ext: The specified procedure could not be found.` | add `The specified procedure could not be found` (now only in dll-missing via `WinError 127`) | [V] format |
| triton-hip-dll | triton-windows backends/amd/driver.py:840-847 | `RuntimeError: cannot locate amdhip64.dll after attempted paths [...]` followed by `On Windows, ensure that:` | `cannot locate amdhip64\.dll after attempted paths` | [V] triton_windows-3.8.0.post29 wheel |
| triton-cc-fail (replace `triton-compiler`) | triton/runtime/build.py:182 when clang-cl (from _rocm_sdk_core) fails, e.g. no Windows SDK INCLUDE/LIB | `Failed to compile. cc_cmd: [...]` then the CalledProcessError | `Failed to compile\. cc_cmd:` | [V] same wheel |
| triton-import-order | triton-windows commits e520c2e3 / 602b6073 ("TheRock ROCm requires importing torch before triton") | [?] likely `ImportError: DLL load failed while importing libtriton` | `DLL load failed while importing libtriton` | [I] |
| hip-oom | runtime allocation beyond the WDDM-visible pool | `torch.OutOfMemoryError: HIP out of memory. Tried to allocate ...` | `HIP out of memory` (keep separate from compiler `oom`) | [V] torch message family |
| no-binary-for-gpu (keep `invalid-device-function`) | code object for another arch | `hipErrorNoBinaryForGpu: Unable to find code object for all current devices!` / `invalid device function` | keep | [I] |
| tdr | GPU watchdog reset during a long kernel (see tdr.md) | [?] torch: `RuntimeError: HIP error: unspecified launch failure` or `... device lost`; Windows: System log, source Display, Event ID 4101 | `HIP error: (unspecified launch failure\|.*device.*lost\|the launch timed out)` + always print "check Event Viewer > System > Display 4101" | [?] |
| compiler-abi-noise (not a failure) | torch cpp_extension.py:588-637 runs `clang-cl` with no input to read a version | `Error checking compiler version for C:\...\clang-cl.exe` (warning) | exclude from ERROR_LINE (or set TORCH_DONT_CHECK_COMPILER_ABI=1) | [V] code |
| clang-cl-unused-arg (not a failure) | /Zc:preprocessor is in clang-cl's unsupported list (Options.td `_SLASH_Zc : CLJoined<"Zc:">`) | `clang-cl: warning: argument unused during compilation: '/Zc:preprocessor'` | add to ERROR_NOISE | [I] wording |

## Current KNOWN entries (20) at c4a9f87

| id | Verdict |
|---|---|
| no-msvc | Keep. Add the where-cl text. |
| missing-win-headers | Keep, but the regex misses clang wording for most headers (only windows.h/vcruntime.h). Use the replacement above. |
| missing-hip-headers | Keep. Clang wording is right. |
| missing-import-lib | Keep. Widen to any `*.lib` (rocblas is missing from the list) and print the name. |
| device-libs | **Replace.** `amdgcn[\\/]bitcode` also appears in text that is not an error: hipcc `-v` / `-verbose` command echoes (`-mlink-builtin-bitcode .../amdgcn/bitcode/...`) and any log that prints HIP_DEVICE_LIB_PATH. So it can false-match a failure log whose real cause is elsewhere [I]. Use `cannot find ROCm device library` [V]. |
| torch-not-rocm | Keep, change the cause: with ROCm torch, cpp_extension falls back to `_rocm_sdk_core` (cpp_extension.py:186-215), so this fires mainly when a CPU/CUDA torch got installed (e.g. by `uv run` project sync). |
| build-isolation | Keep. |
| path-too-long | Keep. |
| dll-missing | Keep. Add capture of the module name. |
| entrypoint | Keep. Add the Python wording (above). |
| no-matching-dist | Keep. Generalise `no version of torch` to `there is no version of` (uv wording covers every package). |
| network | Keep. `timed out` is broad but only searched on failing steps. |
| oom | Keep for the compiler. Split out `HIP out of memory`. |
| file-locked | Keep. |
| flag-leak | Keep. Regex fine. |
| msvc-flag-on-clang | **Drop** once CXX=clang-cl (no cl.exe D9002/D8021 output). If cl.exe is ever used, restrict-cl is the real failure. |
| python-h | **Fix or drop.** It misses clang's `fatal error: 'Python.h' file not found`. uv-managed CPython has headers, so it is unlikely to fire at all. |
| invalid-device-function | Keep (runtime). |
| triton-compiler | **Trim.** triton-windows falls back to the bundled tcc.exe (runtime/build.py:44), so "Failed to find C compiler" / "Please specify one with the CC" cannot fire on Windows [V]. Keep only the `CalledProcessError.*(clang-cl\|cl\.exe)` branch and add `Failed to compile\. cc_cmd:` (triton-cc-fail). |
| no-triton | Keep. |
