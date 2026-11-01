# A chat completion returns coherent text

- Status: done
- Depends: S4-01
- Risk: LOW once S4-01 passes; included to pin down the exact request used for later regression.

Sprint: [Sprint 4: serve a small Qwen over HTTP](README.md) · ID S4-03 · priority order 3 in the sprint

**Goal.** A saved curl command and a coherent answer.

**Steps.**
1. `curl.exe localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @req.json` with thinking disabled and a short `max_tokens`.
2. Check `/health`, `/v1/models`, streaming on and off.

**Done when.**
- Request file and response saved in the run folder.

**If it fails.** Incoherent text with correct token plumbing points to numerics (go back to S3-05).

**Evidence.** (run folders, versions, numbers; fill in)

- 2026-10-30, Qwen3.5-9B EXL3 4.00bpw, ctx 8192, thinking off (`chat_template_kwargs.enable_thinking=false`), `model` must equal the served name `Qwen3.8-Flash-Yamz` (`"x"` gets HTTP 400 `unknown model; expected ...`). Prompt "Reply with one short sentence: what is 2+2?" -> `"2+2 equals 4."`, `finish_reason: stop`, prompt 25 tok in 231 ms (108 tok/s), 8 tokens decoded in 573 ms (**13.97 tok/s**), `speculative: false` **[V: user's curl output]**. First request after READY, 8 tokens: a smoke number, not a benchmark. Request file `logs\windows\req.json` (gitignored). `/health` and `/v1/models` answer correctly (`hip_compiler` path shown after the fix below) **[V]**.
- 2026-10-30, after the `hipcc.exe` fix (HIP JIT kernels on): "about 150 words on hash table collisions", `enable_thinking=false`, `max_tokens=300`: coherent 200-token answer (chaining vs open addressing, correct), `finish_reason: stop`; prompt 28 tok in 1519 ms (18 tok/s, first long request, includes first-use work), **decode 200 tok in 6590 ms = 30.3 tok/s** vs 13.97 tok/s of the earlier 8-token run on the Triton/PyTorch fallback chain **[V: user's curl output]**. Single runs, no repeat, no Linux reference. Still open: streaming.
- 
- 2026-10-31: `/health`, `/v1/models`, non-streaming and streaming (`stream: true`, `curl.exe -N`) answered correctly; request files `req*.json` in `logs\windows` **[V: user reports]**.

**Links.** next: [S4-04](04-api-parity.md)

Depends on: [S4-01](01-serve-reaches-ready.md)
