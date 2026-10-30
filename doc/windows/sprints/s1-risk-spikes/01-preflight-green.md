# Preflight green: doctor and setup on the target machine

- Status: active
- Depends: -
- Risk: Prerequisite for everything, ordered first by dependency. Unknowns: VS detection, index host, `rocm-sdk init` on Windows, driver version.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-01 · priority order 1 in the sprint

**Goal.** Get `doctor` to no FAIL and `setup` to completion on the Windows 11 / gfx1151 machine.

**Steps.**
1. Install `uv` (`winget install astral-sh.uv`). Enable Developer Mode (symlinks: 15 repo files, 4 in `tools/qwen`, are git symlinks). Clone to a short path without spaces: `git clone -c core.symlinks=true <repo> C:\k`, `cd C:\k`, `git checkout claude/modest-rubin-p69b5g`. Optional but recommended: Defender exclusion for `C:\k`.
2. `uv run --no-project python scripts/windows/kyojin_win.py -v doctor`; fix each FAIL using its printed hint.
3. `uv run --no-project python scripts/windows/kyojin_win.py -v setup`.
4. Re-run `doctor`; confirm `rocm_sdk`, `torch_gpu` and `triton` details.
5. In the SDK root (`rocm-sdk path --root`) note whether `lib\hipblas.lib`, `lib\rocblas.lib` and `bin\hipcc.exe` exist (`dir`), and `torch.cuda.mem_get_info()`; also try one 8 GiB and one 16 GiB `torch.empty(..., device='cuda')`.
6. Send the log folders (or the `FAILURE.md` contents) to the session; journal the results.

**Done when.**
- `doctor` has no FAIL; `setup` finished; `pip-freeze.txt` exists in the setup run folder.
- `torch_gpu` shows arch `gfx1151`; the visible GPU memory figure is copied to the journal.

**If it fails.** Use `FAILURE.md`. If no signature matched, add one (see [deferred.md](../../deferred.md) error catalogue). If the index/pin is the problem set `KYOJIN_TORCH_VER`/`KYOJIN_ROCM_VER` or `KYOJIN_ROCM_INDEX`.

**Evidence.** (run folders, versions, numbers; fill in)

- 2026-10-30, first `doctor` on the target (`C:\Projects\kyojin`, report.json): **[V]** Windows 11 build 26300 (10.0.26300), Python 3.12.10 (python.org), git 2.56.0.windows.2, uv 0.13.6, VS Build Tools 2022 17.14.37806.7, Windows SDK 10.0.28000.0 (very new; compatibility with the SDK's clang untested), long paths enabled, `core.symlinks=true`, symlinks intact, 214 GB free, AMD index reachable with all 4 pinned wheels. CPU Ryzen AI Max+ 395.
- **[V] RAM: Windows sees 31.6 GB total (6.6 GB free, 79% load), page file 127.6 GB.** The machine has 128 GB, so about 96 GB is presumably carved out as GPU memory in BIOS/Adrenalin **[I]**; `dedicated_vram_gb` from the (fixed) `gpu` probe will confirm. Consequence: compile jobs get about 6 GB free, so `doctor` suggests `--jobs 1`; close other programs before `build`.
- `gpu` probe crashed (registry key missing, FileNotFoundError, Polish OS message): my bug, fixed (CIM primary, registry optional). Windows messages are localized (Polish): signatures keyed on English OS text may not match; `VSLANG=1033` is now forced for MSVC tools, and the signatures use WinError codes where possible.

**Links.** [kyojin_win.py](../../../../scripts/windows/kyojin_win.py) · [assessment §5](../../assessment.md) (dependency set) · next: [S1-02](02-hip-extension-spike.md)

- 2026-10-30, `setup` run on the target (report e39d2713): **[V]** torch+ROCm install 76 s, requirements 13 s, `rocm-sdk init` 20 s (devel tree expanded to `.venv-win\Lib\site-packages\_rocm_sdk_devel`), `rocm-sdk targets` lists gfx1151. The venv uses a **uv-managed CPython 3.12** (`%APPDATA%\uv\python\cpython-3.12-windows-x86_64-none`), not the python.org install. hipcc: `HIP version: 7.16.26332`, `AMD clang version 23.0.0git`, target `x86_64-pc-windows-msvc`. Headers hipblas/hipsparse/thrust/hip present. `bin\amdhip64_7.dll` present (name predicted by the review). GPU: Radeon 8060S, PCI DEV_1586, driver 32.0.32015.7029.
- `rocm-sdk test`: 26 of 27 pass; the one error (`testCLIUsesDevelRootPath`, FileNotFoundError) spawns `rocm-sdk` by name and the venv `Scripts` dir was not on PATH: harness artifact, fixed in the script (PATH now set for those steps). Not a toolchain problem.
- `torch_gpu` and `triton` probes **failed with `OpenBLAS error: Memory allocation still failed after 10 retries`** (numpy import, before torch): the machine was under memory pressure (about 6 GB free of 31.6 GB visible, GPU busy with another workload per the user). Probes now cap `OPENBLAS_NUM_THREADS=4`. **Re-run `check` after closing the GPU/RAM-heavy program.**
- Still open for S1-01: `dir <sdk>\lib\hipblas.lib` and `lib\rocblas.lib` (root_listing shows a `lib` dir; the specific libs are not listed).
- The registry display-class key does not exist on this Windows build (WinError 2); the registry VRAM lookup was removed, torch's `mem_get_info` is the source for GPU memory.
- 2026-10-30, `check` run (report 1d5dfea2) with another LLM workload still running on the GPU: **[V]** `torch_gpu` died with `LLVM ERROR: out of memory / Allocation failed`, exit 0xC0000409 (probe printed no result; stage unknown, now logged with `STEP` markers); `triton` imported (`3.9.0`) but my probe failed with `@jit functions should be defined in a Python file` (**my bug**: probes ran via `python -c`; they now run from a file in the run folder); `extension` not built yet (expected). `rocm_sdk` unchanged PASS. GPU/RAM tests need the machine's memory free: re-run when the other workload is stopped.
- 2026-10-30, `check` again: `rocm_sdk` PASS, `triton` PASS; `torch_gpu` FAIL `HIPBLAS_STATUS_ALLOC_FAILED` in `hipblasCreate` (GPU memory exhausted by the other LLM, same cause as the earlier LLVM OOM). Needs the GPU free; `device.mem_free_gb` is in that run's report.json.
- 2026-10-30, `check` report 5be57594: `torch_gpu` reaches the matmul, then `HIPBLAS_STATUS_ALLOC_FAILED` in `hipblasCreate` although 107.72 of 107.87 GiB GPU memory are reported free and `randn` on the device works. The earlier theory (GPU memory exhausted by the other LLM) is **not supported by this data**. The probe now tests an elementwise kernel and the matmul per BLAS backend (default, cublas, cublaslt) so the next run says whether kernels, rocBLAS or hipBLASLt is at fault. Triton PASS confirmed with `max_abs_err 0.0`.
