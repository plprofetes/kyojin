# Numeric parity with a reference

- Status: done
- Depends: S3-01
- Risk: MEDIUM: tolerances and a trustworthy reference are the hard part.

Sprint: [Sprint 3: runtime kernels run](README.md) · ID S3-05 · priority order 5 in the sprint

**Goal.** Windows output agrees with a known-good run.

**Steps.**
0. **No Linux gfx1151 box is available (user, 2026-10-30)**, so the reference is the engine's own non-HIP path on the same Windows machine: `tools/qwen/serve.py` documents that the PLE/DQ/GR/GDN HIP kernels are "bitwise equal to the torch/triton paths" and "set any of them to 0 to go back" (SERVE_ENV comment, ~line 508). Run A: defaults. Run B: `EXL3_PLE_HIP=0 EXL3_DQ_HIP=0 EXL3_GR_HIP=0 EXL3_GDN_FUSE=0 EXL3_GDN_HF=0` (set in the shell before `serve`). Same request, `temperature: 0`, `logprobs: true`, `top_logprobs: 5`, `return_token_ids: true`, non-streaming; compare token ids and logprobs. This checks the Windows-compiled kernels against the shipped fallback, not the whole stack against another platform; say so in the evidence.
1. Preferred reference (if a Linux box appears): the same pack and prompt on a Linux gfx1151 Kyojin run (greedy tokens). Alternative: Hugging Face transformers fp16 on the base model, first 16 tokens, allowing near-ties to differ.
2. Save prompt, tokens and logits summary under the run folder; document tolerance.

**Done when.**
- 16/16 greedy tokens equal, or documented near-tie differences with logit deltas.

**If it fails.** If outputs diverge from token 1: bisect per layer with the repo's logit-check tools ([logit_check.py](../../../../tools/strix_halo/logit_check.py)).

**Evidence.** (run folders, versions, numbers; fill in)

- 
- 2026-10-30, Qwen3.5-9B EXL3 4.00bpw, `temperature: 0`, 48 tokens, `logprobs: true`, `top_logprobs: 5`, `return_token_ids: true`, non-streaming, thinking off. Run A (defaults: PLE/DQ/GR/GDN HIP on, startup 99 s) vs run B (`EXL3_PLE_HIP=0 EXL3_DQ_HIP=0 EXL3_GR_HIP=0 EXL3_GDN_FUSE=0 EXL3_GDN_HF=0`, startup 24 s): `choices[0]` compared as JSON (text, token ids, logprobs) is **identical (True)** **[V: user's PowerShell comparison]**. The reference is the engine's own fallback path on the same machine, not another platform and not the unquantized model: it shows the Windows-compiled HIP kernels agree bitwise with the fallback on this prompt, no more. Only one prompt, 48 tokens; `gr_mix`/`qsa_pf` kernels were not built for this model, so they are not covered.


**Links.** next sprint: [Sprint 4](../s4-serve-qwen/README.md)

Depends on: [S3-01](01-end-to-end-smoke.md)
