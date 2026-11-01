# Greedy parity through the API

- Status: done
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
- 2026-10-31: the S3-05 A/B was run through the server API (`temperature: 0`, `logprobs`): HIP kernels on vs off identical **[V]**; no Linux reference exists for this device (see S3-05). Speculative (MTP) vs plain decoding on Qwen3.8-27B 4.00bpw: identical text at temperature 0 **[V]**.

**Links.** next: [S4-05](05-stability.md)

Depends on: [S4-03](03-chat-completion.md), [S3-05](../s3-runtime-kernels/05-numeric-parity.md)
