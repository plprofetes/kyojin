# Pin versions from a known-good run

- Status: done
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
- 2026-10-31, `setup --recreate` from the pins on the target (`20261031-014210-setup`): PASS=19, FAIL=0; torch+ROCm step 5 s and requirements step 1 s because the uv wheel cache was warm (so this is not a clean-machine timing); `rocm-sdk init/test/targets` PASS; probes torch_gpu and triton PASS. Then `check` (`20261031-014303-check`): PASS=4 including the extension built before the recreate (`exllamav3_ext` 135 symbols, `exllamav3` imported with the new pinned packages) **[V: user's output]**. Still missing for the done-when: the line-by-line comparison of `uv pip freeze` with `constraints-win.txt` (command in the journal), and one served request on the pinned set (pydantic 2.13.5 replaced the 2.14.0b1 that ran before).
- 2026-10-31: comparison run after the recreate: every `name==version` line of `constraints-win.txt` is present identically in `uv pip freeze --python .venv-win\Scripts\python.exe` (the check printed no MISSING/DIFFERENT line) **[V: user's output]**. Done-when met: `setup` from the pins reproduces the pinned set. Follow-up not yet seen: a served request on pydantic 2.13.5 (before: 2.14.0b1).


**Links.** next: [S5-04](04-install-doc.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
