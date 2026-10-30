# Greedy parity through the API

- Status: todo
- Depends: S4-03, S3-05
- Risk: MEDIUM: depends on the reference from S3-05.

Sprint: [Sprint 4: serve a small Qwen over HTTP](README.md) · ID S4-04 · priority order 4 in the sprint

**Goal.** Same prompt, temperature 0, agrees with the reference for the documented number of tokens.

**Steps.**
1. Reuse the S3-05 prompt; send with `temperature: 0`; compare tokens.

**Done when.**
- Comparison saved; deviations explained.

**If it fails.** Divergence that S3-05 did not show means sampling/serving-path differences; bisect there.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** next: [S4-05](05-stability.md)

Depends on: [S4-03](03-chat-completion.md), [S3-05](../s3-runtime-kernels/05-numeric-parity.md)
