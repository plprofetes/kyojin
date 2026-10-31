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

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [existing workflow](../../../../.github/workflows/build.yml) · [S5-02 decision](../s5-harden/02-ci-feasibility.md) · next: [S6-02](02-installer-and-bundle.md)

Depends on: [S5-02](../s5-harden/02-ci-feasibility.md)
