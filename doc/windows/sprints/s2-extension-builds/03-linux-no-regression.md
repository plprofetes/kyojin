# Linux no-regression check for the setup.py changes

- Status: active
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
- 2026-10-31, diff review of every non-Windows-folder file against the merge base `2132bd2` (`git diff 2132bd2..HEAD`), Linux behaviour **[V by reading; no Linux build run]**: `setup.py` (all new branches under `windows`; `extra_link_args=[]` on Linux; `libraries` unchanged for non-Windows; the `-include format` probe now `not windows`, true on Linux), `moe_mul1.cpp` (`(__GNUC__ && __linux__) || __clang__`: the first term is the old condition, so Linux gcc/clang keep the same branch), `hgemm.cu` (`DTUNE_MKDIR` expands to `mkdir(p, 0755)` as before), `exllamav3/ext.py` (new `ImportError` only for `windows and torch.version.hip`), `util/hip_compiler.py` (`_EXE` is `""` on Linux), `util/hip_lib.py` (new branch under `sys.platform == "win32"`), `tools/startup_health.py` (`allow_reuse_address` stays True off Windows; `register_at_fork` guarded by `hasattr`, true on Linux), `.gitignore`, `README.md` (one link). Tests run here: `tests/test_hip_lib.py` 11 passed; `tools/test_startup_health.py` 40 passed / 4 failed, **the same 4 fail on the base commit in this sandbox** (so not caused by the branch); `tests/test_hip_compiler_cpu.py` cannot be collected without torch (also on base), so its new Windows test was exercised standalone only. Still open: a real Linux ROCm build (`./build.sh`) and the full `tests/` run on a GPU machine.


**Links.** [setup.py](../../../../setup.py) · [container workflow](../../../../.github/workflows/container.yml) · next sprint: [Sprint 3](../s3-runtime-kernels/README.md)

Depends on: [S2-01](01-full-build-triage.md)
