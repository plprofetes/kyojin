# CI tier 1: always-on Linux checks for the Windows port

- Status: todo
- Depends: S5-03
- Risk: LOW: no Windows, no GPU, seconds to minutes; the unknown is how much of the script imports cleanly on Linux (the unit tests already run there).

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-06 · priority order 6 in the sprint

**Goal.** A workflow on `ubuntu` that runs on pushes and pull requests touching `scripts/windows/**`, `requirements.txt`, `pyproject.toml` or `doc/windows/**`, and on a weekly schedule, so drift is found without a Windows machine. Covered by the user's consent for GH CI/CD (2026-10-31, "three tiers sound reasonable"); confirm the file name at the start of the session.

**Steps.**
1. Unit tests: `python scripts/windows/test_kyojin_win.py`, `python scripts/windows/test_tasks.py`, `python scripts/windows/tasks.py check` (all stdlib), and `pytest --noconftest tests/test_hip_lib.py`.
2. Pin resolution: the two `uv pip compile` commands from the header of `scripts/windows/constraints-win.txt` (step 1 against the AMD index with `--prerelease allow`, step 2 against PyPI), using `--python-platform windows --python-version 3.12`.
3. Pinned wheels still on the AMD index: `probe_index` in `kyojin_win.py` is "pure HTTP, works on any OS"; call it from a short step (or `doctor` if it degrades cleanly on Linux) and fail the weekly run when the pinned nightly is missing. This is the early warning for the biggest long-term risk (AMD pruning the nightly).
4. Keep it fast (target under 5 minutes) and not required for merges until it has been green for a while.

**Done when.**
- Green on the branch; a deliberate break (change a pin to a missing version) turns it red.

**If it fails.** If the AMD index is slow or rate-limits, run the pin job weekly only and keep the unit tests on every push.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [constraints-win.txt header](../../../../scripts/windows/constraints-win.txt) · [deferred.md CI row](../../deferred.md) · next: [S6-07](07-local-release-gate.md)

Depends on: [S5-03](../s5-harden/03-pin-versions.md)
