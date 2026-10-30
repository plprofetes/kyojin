# Spike: a minimal HIP extension compiles and loads with hipcc.exe + cl.exe

- Status: todo
- Depends: S1-01
- Risk: HIGHEST. No public guide for building PyTorch C++/HIP extensions with hipcc on Windows was found; torch has the code paths but nothing proves them for this SDK. If this fails, the port as designed fails.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-02 · priority order 2 in the sprint

**Goal.** Prove the compile/link/load pipeline on a 30-line extension before spending time on the 72-file build.

**Steps.**
1. **Ask for consent first (new files/subcommand, see [CLAUDE.md](../../../../scripts/windows/CLAUDE.md)); simplest option: have the user run a hand-written 30-line extension.** Create `scripts/windows/spikes/hello_ext/`: `setup.py` (CUDAExtension), one `.cu` (trivial kernel + launcher) and one `.cpp` (host function + pybind), so both hipcc (device) and cl (host) run.
2. Add a `kyojin_win.py spike-ext` subcommand reusing `capture_vs_env` and `sdk_env` (the same environment as `build`); build into a temp dir under `logs/`.
3. Import the `.pyd`, launch the kernel on the GPU, compare with torch.
4. Record the exact ninja `FAILED:` command for every failure; fix env/flags; iterate.

**Done when.**
- `.pyd` builds, imports, kernel output equals the torch reference; run folder saved.

**If it fails.** Decision tree, in order: (a) env fixes (HIP_PATH/ROCM_PATH, INCLUDE/LIB, bitcode path); (b) force clang-cl as host compiler; (c) hipcc for all files; (d) the AMD HIP SDK installer instead of the pip SDK. If none works: stop, report, re-plan (possibly WSL2 or Linux dual boot, which the user has not asked for).

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [setup.py](../../../../setup.py) · [torch cpp_extension notes](../../assessment.md) (§7b) · [env.sh (Linux reference)](../../../../tools/strix_halo/env.sh) · next: [S1-03](03-triton-on-gfx1151.md) (independent, can run in parallel)

Depends on: [S1-01](01-preflight-green.md)
