# Clean-machine run from the docs only

- Status: todo
- Depends: S4-05
- Risk: MEDIUM: exposes hidden state (PATH, caches, installed tools, the developer's habits).

Sprint: [Sprint 5: harden and publish the path](README.md) · ID S5-01 · priority order 1 in the sprint

**Goal.** Fresh Windows user profile or machine reaches READY using only the documentation.

**Steps.**
1. Fresh clone, no existing `.venv-win`, no uv cache assumptions.
2. Follow the documented commands literally; record every deviation.

**Done when.**
- Succeeds; deviations folded back into the docs and the script.

**If it fails.** Every manual step needed is a bug in the script or the docs.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** next: [S5-02](02-ci-feasibility.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
