# CI feasibility: compile-only Windows job

- Status: todo
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

**Links.** [existing workflow](../../../../.github/workflows/build.yml) · next: [S5-03](03-pin-versions.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
