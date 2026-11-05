# Spike: a minimal HIP extension compiles and loads with hipcc.exe + cl.exe

- Status: done
- Depends: S1-01
- Risk: HIGHEST. No public guide for building PyTorch C++/HIP extensions with hipcc on Windows was found; torch has the code paths but nothing proves them for this SDK. If this fails, the port as designed fails.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-02 · priority order 2 in the sprint

**Goal.** Prove the compile/link/load pipeline on a 30-line extension before spending time on the 72-file build.

**Steps.**
No new repo files or scripts. Everything lives in the gitignored scratch folder `logs\windows\scratch\hello_ext` (cleanup: delete `logs\windows\scratch`).

1. Follow [hello_ext_clangcl.md](../../review/hello_ext_clangcl.md) for the exact cmd.exe sequence and the expected `build.log` (it is the reviewed version of the commands below, including `HOME`, `PYTHONUTF8` and where `build.ninja` lives). Open a VS developer shell (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"` finds the install; run its `Common7\Tools\VsDevCmd.bat -arch=amd64`, or use "x64 Native Tools Command Prompt"). From the repo root:
   ```bat
   .venv-win\Scripts\activate
   for /f "delims=" %i in ('rocm-sdk path --root') do set ROCM_HOME=%i
   set ROCM_PATH=%ROCM_HOME%
   set HIP_PATH=%ROCM_HOME%
   set PATH=%ROCM_HOME%\bin;%ROCM_HOME%\lib\llvm\bin;%PATH%
   set PYTORCH_ROCM_ARCH=gfx1151
   set HIP_CLANG_PATH=%ROCM_HOME%\lib\llvm\bin
   set CC=%ROCM_HOME%\lib\llvm\bin\clang-cl.exe
   set CXX=%ROCM_HOME%\lib\llvm\bin\clang-cl.exe
   set DISTUTILS_USE_SDK=1
   mkdir logs\windows\scratch\hello_ext & cd logs\windows\scratch\hello_ext
   ```
2. Create three files in that folder (a trivial extension that exercises both hipcc for the `.cu` and `cl.exe` for the `.cpp`):
   - `hello.cu`
     ```cpp
     #include <torch/extension.h>
     #include <ATen/cuda/CUDAContext.h>
     __global__ void add1_kernel(float* x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.0f; }
     void add1_cuda(torch::Tensor x) {
         int n = x.numel();
         add1_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), n);
     }
     ```
   - `hello.cpp`
     ```cpp
     #include <torch/extension.h>
     void add1_cuda(torch::Tensor x);
     void add1(torch::Tensor x) { TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "float32 cuda tensor"); add1_cuda(x); }
     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add1", &add1); }
     ```
   - `setup.py`
     ```python
     from setuptools import setup
     from torch.utils.cpp_extension import CUDAExtension, BuildExtension
     setup(name="hello_ext", ext_modules=[CUDAExtension("hello_ext", ["hello.cpp", "hello.cu"])], cmdclass={"build_ext": BuildExtension})
     ```
3. Build and test:
   ```bat
   python setup.py build_ext --inplace -v > build.log 2>&1
   python -c "import torch, hello_ext; x = torch.zeros(1000, device='cuda'); hello_ext.add1(x); print(x.sum().item())"
   ```
   Expected output `1000.0`.
4. On failure send `build.log`, or at least the first `FAILED:` block (ninja prints the exact failing command) and the first error line. Record each fix (env var, flag) in the journal; these become the `build` step's environment.

**Done when.**
- `.pyd` builds, imports, kernel output equals the torch reference; run folder saved.

**If it fails.** Decision tree, in order: (a) env fixes (HIP_PATH/ROCM_PATH, INCLUDE/LIB, bitcode path); (b) force clang-cl as host compiler; (c) hipcc for all files (note: the first attempt already uses `clang-cl` as host compiler as AMD does for its own Windows PyTorch build, TheRock `build_prod_wheels.py` @ `682b3407` lines 701-706; try plain `cl` only to compare); (d) the AMD HIP SDK installer instead of the pip SDK. If none works: stop, report, re-plan (possibly WSL2 or Linux dual boot, which the user has not asked for).

**Evidence.** (run folders, versions, numbers; fill in)

- 
- 2026-10-30: superseded by [S2-01](../s2-extension-builds/01-full-build-triage.md): the full extension (not a hello-world one) builds, links and imports with MSVC host + clang-cl + hipcc **[V]**; the findings of the spike (ninja on PATH, `-Xcompiler` pairs, VsDevCmd env) are in the journal.

**Links.** [setup.py](../../../../setup.py) · [torch cpp_extension notes](../../assessment.md) (§7b) · [env.sh (Linux reference)](../../../../tools/strix_halo/env.sh) · next: [S1-03](03-triton-on-gfx1151.md) (independent, can run in parallel)

Depends on: [S1-01](01-preflight-green.md)
