# Linux no-regression check for the setup.py changes

- Status: todo
- Depends: S2-01
- Risk: LOW-MEDIUM but unrecoverable if forgotten: the Linux build is the product today.

Sprint: [Sprint 2: the full extension builds and imports](README.md) · ID S2-03 · priority order 3 in the sprint

**Goal.** Show the Windows-motivated edits did not change the Linux build.

**Steps.**
1. Review `git diff 2132bd2 -- setup.py`: every change must be inside a Windows or hip-on-Windows condition.
2. Build on Linux if a ROCm Linux environment is available (`./build.sh`), otherwise rely on a `container.yml` pull-request build.
3. Run `python -m unittest scripts/windows/test_kyojin_win.py` and `python scripts/windows/tasks.py check`.

**Done when.**
- Diff reviewed; Linux build or container build green.

**If it fails.** Revert the offending hunk or gate it further; do not merge Windows fixes that change Linux flags.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [setup.py](../../../../setup.py) · [container workflow](../../../../.github/workflows/container.yml) · next sprint: [Sprint 3](../s3-runtime-kernels/README.md)

Depends on: [S2-01](01-full-build-triage.md)
