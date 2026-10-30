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
