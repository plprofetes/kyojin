# Preflight green: doctor and setup on the target machine

- Status: todo
- Depends: -
- Risk: Prerequisite for everything, ordered first by dependency. Unknowns: VS detection, index host, `rocm-sdk init` on Windows, driver version.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-01 · priority order 1 in the sprint

**Goal.** Get `doctor` to no FAIL and `setup` to completion on the Windows 11 / gfx1151 machine.

**Steps.**
1. Install `uv` (`winget install astral-sh.uv`), clone, `git checkout claude/modest-rubin-p69b5g`.
2. `uv run scripts/windows/kyojin_win.py -v doctor`; fix each FAIL using its printed hint.
3. `uv run scripts/windows/kyojin_win.py -v setup`.
4. Re-run `doctor`; confirm `rocm_sdk`, `torch_gpu` and `triton` details.
5. Send the log folders (or the `FAILURE.md` contents) to the session; journal the results.

**Done when.**
- `doctor` has no FAIL; `setup` finished; `pip-freeze.txt` exists in the setup run folder.
- `torch_gpu` shows arch `gfx1151`; the visible GPU memory figure is copied to the journal.

**If it fails.** Use `FAILURE.md`. If no signature matched, add one (see [deferred.md](../../deferred.md) error catalogue). If the index/pin is the problem set `KYOJIN_TORCH_VER`/`KYOJIN_ROCM_VER` or `KYOJIN_ROCM_INDEX`.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [kyojin_win.py](../../../../scripts/windows/kyojin_win.py) · [assessment §5](../../assessment.md) (dependency set) · next: [S1-02](02-hip-extension-spike.md)
