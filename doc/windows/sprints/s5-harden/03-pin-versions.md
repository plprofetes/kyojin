# Pin versions from a known-good run

- Status: todo
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

**Links.** next: [S5-04](04-install-doc.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
