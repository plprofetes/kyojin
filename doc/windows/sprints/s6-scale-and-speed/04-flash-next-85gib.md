# Load Qwen3.8-Flash-Next 2.18bpw (about 85 GiB)

- Status: todo
- Depends: S1-04
- Risk: MEDIUM-HIGH: the pack is 90.7 GB (about 84.5 GiB) against a measured allocatable limit of about 100 GiB; more kernels (PLE, n-gram) and host RAM pressure (31.6 GB visible) are unknown.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-04 · priority order 4 in the sprint

**Goal.** Load `turboderp/Qwen3.8-Flash-Next-exl3` branch `2.18bpw_h6_ng8` and generate; record memory and speed, or explain the failure.

**Steps.**
1. Check disk (90.7 GB download) and stop other GPU users; run the S1-04 sweep first if anything changed.
2. `hf.exe download turboderp/Qwen3.8-Flash-Next-exl3 --revision 2.18bpw_h6_ng8 --local-dir C:\models\...`; check the file list for `mtp.*` tensors and `ngram_embedding.safetensors`; also look at yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz (size not checked).
3. `serve --model <dir> -c 8192` (small context first); watch GPU memory (`Dedicated Usage`) and the host RAM during the load.
4. One greedy request; then the soak commands from S4-05 with a short context.

**Done when.**
- Generates, with GPU memory at load and at idle recorded; or the failing step and error named.

**If it fails.** If it does not fit, record the margin; try a smaller context or a lower-bit branch; do not change loader code ad hoc (add a task).

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [S1-04 ceiling](../s1-risk-spikes/04-gpu-memory-ceiling.md) · [deferred.md](../../deferred.md) · next: [S6-05](05-windows-host-memory.md)

Depends on: [S1-04](../s1-risk-spikes/04-gpu-memory-ceiling.md)
