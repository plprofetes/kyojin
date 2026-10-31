# CI: build the Windows ROCm wheel on a hosted runner

- Status: todo
- Depends: S5-02
- Risk: HIGH: nothing is known about a hosted-runner build (time, RAM, arch detection without a GPU); needs the user's consent for the workflow target.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-01 · priority order 1 in the sprint

**Goal.** A manual `workflow_dispatch` target `windows_rocm` in `.github/workflows/build.yml` that builds `exllamav3-<ver>+rocm<rocm>.torch<torch>-cp312-cp312-win_amd64.whl` and uploads it as an artifact. No GPU test, no automatic release.

**Steps.**
1. Ask for consent (new workflow target). Reuse the existing `windows_smoke` pattern: windows-2022, uv, VS Build Tools pin, `build_wheels` job.
2. Install the pinned torch/ROCm/SDK with `-c scripts/windows/constraints-win.txt` and the AMD index (about 9 GB); set `PYTORCH_ROCM_ARCH=gfx1151`, `DISTUTILS_USE_SDK=1`, the VS developer environment and `EXL3_ROCM_SDK`; reuse `kyojin_win.py` where it fits instead of re-implementing the environment.
3. Build the wheel (`python -m build --wheel -n` or `setup.py bdist_wheel`); name it with the local version tag `+rocm...torch...` as the CUDA wheels do (`build.yml` BUILDTAG).
4. Smoke without a GPU: wheel contents, DLL dependencies (`dumpbin /dependents`), and whether `import exllamav3_ext` works without an AMD driver (unknown **[?]**).
5. Record build time, disk and RAM of the runner. Disk first: GitHub documents 14 GB SSD for standard Windows runners; our footprint is about 10 GB plus the uv download cache. Print free space on C: and D: at the start; point `UV_CACHE_DIR` at the same drive as the venv (hardlinks) or use `--no-cache`; delete wheels after install; if still short, a larger runner (paid) or a self-hosted runner.
6. Public repository: standard runners are free and unlimited (4 vCPU / 16 GB); a private repository gets 2 vCPU / 8 GB and a monthly minutes quota (not verified here), so `jobs` from `compute_jobs` will be lower. Forks have Actions disabled until enabled in the fork's Actions tab **[?, from memory]**.

**Done when.**
- A green manual run with the wheel as an artifact, or a written reason (numbers) why not, in [deferred.md](../../deferred.md).

**If it fails.** If `setup.py` needs a GPU for architecture detection, set `PYTORCH_ROCM_ARCH` explicitly or patch the detection behind a Windows condition. If the runner is too small, defer and keep the local script as the gate.

**Assessment (2026-10-31, reading only; nothing implemented, user: "assess only, next session").** User consent for the GH CI/CD job is given and the repository is public (standard runners free and unlimited: 4 vCPU, 16 GB RAM, 14 GB SSD, 6 h job limit, artifacts 500 MB on Free).
- **No GPU needed to compile [V by reading]:** `exllamav3/util/arch_list.py` skips device detection when `PYTORCH_ROCM_ARCH` is set, and `kyojin_win.py` `sdk_env` sets it to `gfx1151`.
- **Script gaps (extend existing subcommands, no new subcommand):** (1) `setup` preflight FAILs on a machine without an AMD adapter (`probe_gpu`) and on free disk below `MIN_FREE_GB = 40` (the runner documents 14 GB), and its tail runs `probe_torch`/`probe_triton`, which need a GPU: a `--ci` flag that downgrades those two FAILs to WARN and skips the GPU probes. (2) `build` only runs `setup.py build_ext --inplace` and ends with `probe_extension`: a `--wheel` option that runs `setup.py egg_info --tag-build=<tag> bdist_wheel --dist-dir dist` under the same environment (the `build` package is not in the pins; `wheel` is), checks that the wheel contains `exllamav3_ext*.pyd`, and under `--ci` skips `probe_extension`. (3) A pure function for the tag, `+rocm<ROCM_VER>.torch<torch base>` = `+rocm10.1.0a20260822.torch2.15.0a0`, with a unit test. Existing CUDA wheels use `+cu...-torch...` via `python -m build -n --wheel -C--build-option=egg_info -C--build-option=--tag-build=...` (`build.yml` line 396).
- **Workflow shape:** a separate file `.github/workflows/windows-rocm.yml` (leaves the 526-line `build.yml` CUDA matrix untouched; easier to review): `workflow_dispatch`, plus a temporary `push` trigger for the working branch while iterating (a `workflow_dispatch` workflow must exist on the default branch to be dispatched; remove the push trigger before the upstream PR). Steps: print CPU/RAM/free disk of C: and D:; `astral-sh/setup-uv`; `UV_CACHE_DIR` on the same drive as the venv (hardlinks) or `--no-cache`; `kyojin_win.py setup --ci`; `build --ci --wheel`; upload `dist/*.whl` as an artifact; informational steps with `continue-on-error`: `dumpbin /dependents` of the `.pyd`, `import torch, exllamav3_ext` without a GPU driver, wheel size.
- **Runner facts to measure on the first run, not assume:** the VS 2022 edition and version on `windows-2022` (the CUDA job installs Build Tools 17.9.7 via choco; our local build used 17.14.37806 and Windows SDK 10.0.28000.0), free disk on C: (docs say 14 GB), whether the 9 GB SDK download plus venv fits, wall time with 3 jobs, whether `rocm-sdk init`/`test` work without a GPU (`test` is only a WARN in the script).
- **Prerequisite on the user side:** Actions must be enabled in the fork (disabled by default on forks).
- **Not covered by this job:** `check`, `serve`, any GPU run, numeric checks; a released wheel still needs a manual test on the Strix Halo machine.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [existing workflow](../../../../.github/workflows/build.yml) · [S5-02 decision](../s5-harden/02-ci-feasibility.md) · next: [S6-02](02-installer-and-bundle.md)

Depends on: [S5-02](../s5-harden/02-ci-feasibility.md)
