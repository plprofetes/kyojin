# Prune and verify the error catalogue

- Status: done
- Depends: S5-01
- Risk: LOW.

Sprint: [Sprint 5: harden and publish the path](README.md) · ID S5-05 · priority order 5 in the sprint

**Goal.** Every signature is observed or marked as hypothetical, each with a test sample.

**Steps.**
1. Compare `KNOWN` in the script against failures actually seen (journal); remove or flag unobserved ones.

**Done when.**
- Catalogue matches reality; tests updated.

**If it fails.** -

**Evidence.** (run folders, versions, numbers; fill in)

- 
- 2026-10-31: `OBSERVED` in `kyojin_win.py` lists the 10 signatures seen on the target (winsock-provider, no-ninja-fallback, cpu-target-feature, compiler-rt-builtin, tokenizers-rc, jit-hipcc-missing, register-at-fork, dll-missing, openblas-oom, gpu-alloc-failed); the other 33 stay as hypotheses and are printed with "(hypothesis, not yet observed)". None removed: each still has a test sample. Unit test covers the tagging **[V: tests pass on Linux]**. Caveat: `winsock-provider` was observed but its documented cause ("SystemRoot missing / damaged catalogue") is now believed to be the empty VsDevCmd environment **[I]**; `gpu-alloc-failed` was observed once with an unestablished cause.


**Links.** next sprint: [Sprint 6](../s6-scale-and-speed/README.md)

Depends on: [S5-01](01-clean-machine-run.md)
