# Linux no-regression check for the setup.py changes

- Status: done
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
- 2026-10-31, plan for the Linux build step with GitHub infrastructure (user: "GH can compile"; nothing run yet). Reading the workflows: (a) `.github/workflows/nix.yml` runs on every pull request and on `workflow_dispatch`: `nix flake check` (its comment says the ROCm extension is reused from the cache and **recompiles when `setup.py` or `exllamav3_ext/` change**, which this branch does; it also runs the CPU test subset) plus `nix fmt` (hooks: nixfmt, deadnix, statix only; the branch touches no `.nix` file). (b) `.github/workflows/container.yml` is dispatchable with `ref` = the branch and `push` unticked: `docker/Dockerfile` builds the Linux ROCm extension for gfx1151 (about 17 minutes cold) and ends with `python -c "import torch, exllamav3_ext"`; this is the "container.yml build" the step above names. Neither runs a GPU test. State of the fork (read through the GitHub API): PR #1 (draft, `claude/modest-rubin-p69b5g` into `main`) exists, but **no workflow is registered and no run exists** on `plprofetes/kyojin`: Actions are disabled on the fork **[V]**. Steps: enable Actions in the fork (Actions tab), dispatch `Nix` and `Build & Publish Container Images` on the branch (or push a real commit to PR #1), read both results, record run URLs here. Done when both are green or the failures are explained.
- 2026-10-31, results on GitHub's `ubuntu-latest` runners, fork `plprofetes/kyojin`, branch `claude/modest-rubin-p69b5g`, Actions enabled by the user and the runs dispatched through the GitHub API **[V: run and job data read through the API]**:
  - **Build & Publish Container Images**, `workflow_dispatch`, `ref` = the branch, push off, head `892967c`: [run 41155094475](https://github.com/plprofetes/kyojin/actions/runs/41155094475) **success**, job 30 minutes, "Build and push images" step 28 minutes (cold cache). `docker/Dockerfile` compiles the Linux gfx1151 ROCm extension (with the branch's `setup.py`, `moe_mul1.cpp`, `hgemm.cu`) and its last stage runs `python -c "import torch, exllamav3_ext"`; a failure there would have failed the step, so the import passed **[I from the Dockerfile; I did not read the build output]**. Nothing was pushed to a registry.
  - **Nix flake check + fmt**, `workflow_dispatch`, head `892967c`: [run 41155091469](https://github.com/plprofetes/kyojin/actions/runs/41155091469) **success**, `nix flake check` 26 minutes, results `checks.x86_64-linux.pre-commit`, `kyojin-tests` (**442 passed, 188 skipped, 349 subtests passed**), `kyojin-hipcc-smoke` all passed; `nix fmt` gate: deadnix, nixfmt, statix Passed.
  - **Nix**, `pull_request` on PR #1, head `8437b09` (the handoff commit): [run 41155286136](https://github.com/plprofetes/kyojin/actions/runs/41155286136) **success**, 23 minutes.
  - Not shown by these runs: whether the Nix check recompiled the extension (the long run time suggests real work; I did not read that part of the log); any GPU execution; a Linux run of the full `tests/` suite on a GPU machine. Linux is therefore compile-checked and CPU-tested on the branch, not GPU-tested, and the Windows-gated edits were also reviewed by reading (previous bullet).
\n\n**Links.** [setup.py](../../../../setup.py) · [container workflow](../../../../.github/workflows/container.yml) · next sprint: [Sprint 3](../s3-runtime-kernels/README.md)

Depends on: [S2-01](01-full-build-triage.md)
