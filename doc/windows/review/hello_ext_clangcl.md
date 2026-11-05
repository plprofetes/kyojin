# Minimal HIP extension with clang-cl on Windows, plus the `ninja -k 0` recipe

Evidence for every step is torch 2.15.0a0+rocm10.1.0a20260822 win_amd64 `torch/utils/cpp_extension.py`
(copy: /tmp/claude-0/scratch/cpp_extension.py), line numbers below. Nothing here has been run on Windows.

## Which shell

**cmd.exe** from the Start menu: "x64 Native Tools Command Prompt for VS 2022". Or a plain cmd.exe that first runs
`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64`
(path differs for Community/Professional; `vswhere -latest -property installationPath` finds it).

Why a VS shell is required even with clang-cl:
- torch writes the link rule with `where cl` and calls `<cl dir>\link.exe` (cpp_extension.py:3286-3297) [V].
  setuptools' MSVCCompiler also links the .pyd with link.exe.
- clang-cl needs INCLUDE/LIB for the MSVC STL, UCRT and Windows SDK [I]. clang can find VS on its own, but LIB for
  link.exe cannot be found that way.

**ninja must be on PATH** (`.venv-win\Scripts`, i.e. an activated venv, as in the steps below). torch runs `ninja --version` (cpp_extension.py:2697-2704); if that fails it silently falls back to the serial distutils path, whose `-Xcompiler /EHsc` pairs break hipcc (see the `-Xcompiler` note below). `kyojin_win.py build` prepends the venv `Scripts` directory itself.

Once VsDevCmd has run, **DISTUTILS_USE_SDK=1 is mandatory**. Otherwise torch stops with
`UserWarning: It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set...` (cpp_extension.py:1341-1345) [V].

## Steps (cmd.exe; `set` lines have no spaces around `=` and no trailing spaces)

```bat
cd /d C:\k
.venv-win\Scripts\activate
for /f "delims=" %i in ('rocm-sdk path --root') do set ROCM_HOME=%i
set ROCM_PATH=%ROCM_HOME%
set HIP_PATH=%ROCM_HOME%
set HIP_CLANG_PATH=%ROCM_HOME%\lib\llvm\bin
set CC=%HIP_CLANG_PATH%\clang-cl.exe
set CXX=%HIP_CLANG_PATH%\clang-cl.exe
set DISTUTILS_USE_SDK=1
set PYTORCH_ROCM_ARCH=gfx1151
set HOME=%USERPROFILE%
set PYTHONUTF8=1
set PATH=%ROCM_HOME%\bin;%HIP_CLANG_PATH%;%PATH%
where hipcc.exe & where clang-cl.exe & where link.exe & dir "%ROCM_HOME%\lib\*.lib"
mkdir logs\windows\scratch\hello_ext & cd logs\windows\scratch\hello_ext
```

Notes:
- If `rocm-sdk path --root` does not contain `bin\hipcc.exe`, use `...\site-packages\_rocm_sdk_core` for the
  hipcc/clang paths. Whether the devel root contains them is unverified [?]. On Linux, Kyojin's build uses the devel
  root, and upstream exllamav3's util/cuda_flags.py:118-150 says core lacks the hipBLAS headers.
- `dir ...\lib\*.lib` answers the open question: are `hipblas.lib` and `rocblas.lib` present?
- Inside a .bat file, write `%%i` instead of `%i`.

Create `hello.cu`, `hello.cpp` and `setup.py` exactly as in doc/windows/sprints/s1-risk-spikes/02-hip-extension-spike.md
(unchanged). Then:

```bat
python setup.py build_ext --inplace -v > build.log 2>&1
python -c "import torch, hello_ext; x = torch.zeros(1000, device='cuda'); hello_ext.add1(x); print(x.sum().item())"
```

Expected: `1000.0`. Order matters: torch is imported first, which loads `amdhip64_7.dll` and the torch DLLs.

## What build.log should contain on success

From the ninja file torch writes (cpp_extension.py:3160-3300) [V]:
- `[1/2] C:\...\clang-cl.exe /showIncludes <cflags> -c C:\k\...\hello.cpp /FoC:\k\...\build\temp.win-amd64-cpython-312\Release\hello.obj -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fms-runtime-lib=dll ... /std:c++20`
  The host rule is `$cxx /showIncludes $cflags -c $in /Fo$out $post_cflags`, with cxx = CXX (3206-3213). `/std:c++20` is appended when no -std is given (`append_std17_if_no_std_present`, 882-889, despite its name) [V].
  `-fms-runtime-lib=dll` is accepted by clang-cl (Options.td fms_runtime_lib_EQ has CLOption visibility, llvm c7a69e6) [V].
- `[2/2] C:\...\hipcc.exe -c C:\k\...\hello.hip -o ...\hello.obj -std=c++20 -Xcompiler -D__HIP_PLATFORM_AMD__=1 ... --offload-arch=gfx1151 -fno-gpu-rdc`
  The `.cu` source is hipified to `.hip` first. hipcc.exe is chosen for ROCm >= 6.4 (2433-2437). clang ignores the
  bare `-Xcompiler` (Options.td: `def : Flag<["-"], "Xcompiler">, IgnoredGCCCompat`) [V] but not its argument: with `-Xcompiler /EHsc` the `/EHsc` becomes an input file. This is what the non-ninja fallback produced on the real run (2026-10-30). The ninja path avoids it (no `-Xcompiler` pairs for HIP) [V].
  The arch flag comes from PYTORCH_ROCM_ARCH via `_get_rocm_arch_flags` [I on exact spelling].
- Then setuptools' link line: `"C:\...\link.exe" /nologo /INCREMENTAL:NO /LTCG /DLL /MANIFEST:EMBED,ID=2 /MANIFESTUAC:NO /LIBPATH:...\torch\lib /LIBPATH:%ROCM_HOME%\lib ... c10.lib torch.lib torch_cpu.lib torch_python.lib amdhip64.lib c10_hip.lib torch_hip.lib /EXPORT:PyInit_hello_ext ... /OUT:...\hello_ext.cp312-win_amd64.pyd`
  (libraries from CUDAExtension: 1594-1597) [I on exact flag order].
- `hello_ext.cp312-win_amd64.pyd` copied next to setup.py.

Expected noise, not failure:
- `Error checking compiler version for ...clang-cl.exe` [V code path 588-637: torch runs the compiler with no input].
  Silence it with `set TORCH_DONT_CHECK_COMPILER_ABI=1`.
- `clang-cl: warning: argument unused during compilation: '/Zc:preprocessor'` (only in the Kyojin build, where
  setup.py passes it) [I wording]. `/Zc:` is in clang-cl's unsupported-but-accepted list (Options.td `_SLASH_Zc : CLJoined<"Zc:">`) [V].
- hipcc warnings about `-Wno-ignored-attributes` / dllexport [I].

## `ninja -k 0`: every compile error in one round trip

Where build.ninja is [V]: torch writes `build.ninja` into the setuptools `build_temp` directory that `compile()`
receives (`build_directory=output_dir`, cpp_extension.py:1054/1285, file name at 2469). For
`python setup.py build_ext --inplace` on Windows that is `build\temp.win-amd64-cpython-312\Release\`. setuptools
adds `Release` on Windows [I, standard distutils behaviour].

Avoid `pip install -e .` for this. A PEP 660 editable build may compile in a temporary directory that is deleted on
failure [I]. Use `python setup.py build_ext --inplace` from the repo root for the Kyojin extension too; it is what
setup.py supports and it leaves build.ninja in place.

Recipe (same VS shell and variables as above; Kyojin repo root):

```bat
python setup.py build_ext --inplace -v > build.log 2>&1
rem first failure stops ninja; now rebuild everything that is left, keep going past failures
cd build\temp.win-amd64-cpython-312\Release
ninja -k 0 -j 8 > ..\..\..\ninja-all.log 2>&1
cd \k
findstr /b /c:"FAILED:" ninja-all.log > ninja-failed.txt
findstr /r /c:": error" /c:"): error" ninja-all.log > ninja-errors.txt
```

Explanation:
- `-k 0` = keep going without limit. At the end ninja prints `ninja: build stopped: cannot make progress due to previous errors.`
  (ninja 3ccb227 src/build.cc:838-840) [V]. Every failed edge is printed as `FAILED: <object>` (pip ninja 1.11) or
  `FAILED: [code=N] <object>` (ninja master, status_printer.cc:244), followed by the full command and the compiler output [V].
- `ninja` is `.venv-win\Scripts\ninja.exe` (the `ninja` pip package from requirements.txt).
- Run it from the same environment. build.ninja hard-codes the compiler paths, but clang-cl needs INCLUDE/LIB and
  hipcc needs HIP_PATH/ROCM_PATH.
- build.ninja on Windows+HIP records **no header dependencies** (no `deps = msvc` for HIP, no depfile for hipcc:
  cpp_extension.py:3191-3213) [V]. After editing a header, delete the affected `.obj` files or the whole
  `build\temp.*` directory; ninja will not rebuild them on its own.
- Linking is not in build.ninja (setuptools links afterwards), so `ninja -k 0` only collects compile errors. Once it
  is clean, rerun `python setup.py build_ext --inplace` to link. LNK errors appear then.

What to send back: `ninja-failed.txt`, `ninja-errors.txt`, the first 40 lines after each distinct `FAILED:` in
ninja-all.log, and `build.log` trimmed to its last 200 lines.
