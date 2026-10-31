# CI feasibility: compile-only Windows job

- Status: done
- Depends: S4-05
- Risk: MEDIUM-HIGH: hosted `windows-2022` runner disk (about 14 GB free on C:) versus a 12 GB SDK plus torch plus build tree [?].

Sprint: [Sprint 5: harden and publish the path](README.md) · ID S5-02 · priority order 2 in the sprint

**Goal.** A decision, and if feasible a `windows_rocm_smoke` job in `build.yml`.

**Steps.**
1. Measure disk used by `.venv-win`, `build/` and logs from a real run.
2. Try a hosted runner with the D: drive or a larger runner; reuse the upstream pattern (VS pin, sccache, `windows_smoke`).

**Done when.**
- Job green, or a written reason and an alternative in [deferred.md](../../deferred.md).

**If it fails.** If disk is impossible on free runners, defer CI and keep the local script as the gate.

**Evidence.** (run folders, versions, numbers; fill in)

- 
- 2026-10-31, disk measured on the target after the full path worked (`Get-ChildItem -Recurse` sums): `.venv-win` **8.7 GB**, `build` 0.2 GB, Triton cache `~\.triton` 0.7 GB, JIT/tune caches about 0 GB, models 22.4 GB (9B 6.7 + 27B 15.7); the uv download cache and the uv-managed Python were not measured **[V: user's output]**. So the environment, build tree and caches need roughly 10 GB, well below the script's `MIN_FREE_GB = 40` estimate and the 12 GB SDK figure that made disk look like the CI problem.
- Decision: **no `windows_rocm_smoke` job now.** Disk is not the blocker. What cannot be done on a hosted runner is the part that matters: there is no gfx1151 GPU, so `check` (torch_gpu, triton), `serve` and all numbers need the real machine. A build-only job (setup + build + import without GPU) is the possible CI value; its runtime on a 4-core hosted runner is unknown (12 min here with 4 jobs on 31 GB visible RAM) and was not tried: the sandbox cannot start workflow runs **[?]**. The local script remains the gate (`check` PASS=4 and a served completion). Written into deferred.md.

**Links.** [existing workflow](../../../../.github/workflows/build.yml) · next: [S5-03](03-pin-versions.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
