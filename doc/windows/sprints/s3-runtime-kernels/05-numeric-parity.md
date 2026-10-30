# Numeric parity with a reference

- Status: todo
- Depends: S3-01
- Risk: MEDIUM: tolerances and a trustworthy reference are the hard part.

Sprint: [Sprint 3: runtime kernels run](README.md) · ID S3-05 · priority order 5 in the sprint

**Goal.** Windows output agrees with a known-good run.

**Steps.**
1. Preferred reference: the same pack and prompt on a Linux gfx1151 Kyojin run (greedy tokens). Alternative: Hugging Face transformers fp16 on the base model, first 16 tokens, allowing near-ties to differ.
2. Save prompt, tokens and logits summary under the run folder; document tolerance.

**Done when.**
- 16/16 greedy tokens equal, or documented near-tie differences with logit deltas.

**If it fails.** If outputs diverge from token 1: bisect per layer with the repo's logit-check tools ([logit_check.py](../../../../tools/strix_halo/logit_check.py)).

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** next sprint: [Sprint 4](../s4-serve-qwen/README.md)

Depends on: [S3-01](01-end-to-end-smoke.md)
