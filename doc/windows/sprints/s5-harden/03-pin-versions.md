# Pin versions from a known-good run

- Status: active
- Depends: S4-05
- Risk: LOW-MEDIUM: nightlies get pruned; an unpinned setup rots.

Sprint: [Sprint 5: harden and publish the path](README.md) · ID S5-03 · priority order 3 in the sprint

**Goal.** Constants in `kyojin_win.py` and a requirements/lock file reflect the exact working set.

**Steps.**
1. Take `pip-freeze.txt` from the successful setup run.
2. Update the pins in `kyojin_win.py`; add a Windows requirements file or uv lock.
3. Add a note on how to move the pin set.

**Done when.**
- Re-running `setup` from the pins reproduces `pip-freeze.txt`.

**If it fails.** If a pinned wheel disappears, the stable-index fallback in [deferred.md](../../deferred.md) applies.

**Evidence.** (run folders, versions, numbers; fill in)

- 
- 2026-10-31: exact pins in place: torch/ROCm/triton constants in `kyojin_win.py` and every other package in `scripts/windows/constraints-win.txt` (from the working venv's `uv pip freeze`; pydantic, pydantic-core, safetensors, typing-extensions replaced by the stable `uv.lock` versions; consistency proven with `uv pip compile` for Windows/3.12). The note on moving the pins is in [install.md](../../install.md). Not yet shown: that `setup --recreate` from the pins reproduces the working set (done-when). Check after a recreate: every line of the constraints file must appear identically in `uv pip freeze --python .venv-win\Scripts\python.exe`.

**Links.** next: [S5-04](04-install-doc.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
