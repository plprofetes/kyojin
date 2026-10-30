# A chat completion returns coherent text

- Status: active
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

- 2026-10-30, Qwen3.5-9B EXL3 4.00bpw, ctx 8192, thinking off (`chat_template_kwargs.enable_thinking=false`), `model` must equal the served name `Qwen3.8-Flash-Yamz` (`"x"` gets HTTP 400 `unknown model; expected ...`). Prompt "Reply with one short sentence: what is 2+2?" -> `"2+2 equals 4."`, `finish_reason: stop`, prompt 25 tok in 231 ms (108 tok/s), 8 tokens decoded in 573 ms (**13.97 tok/s**), `speculative: false` **[V: user's curl output]**. First request after READY, 8 tokens: a smoke number, not a benchmark. Request file `logs\windows\req.json` (gitignored). Still open: `/health`, `/v1/models`, streaming, a longer answer.
- 

**Links.** next: [S4-04](04-api-parity.md)

Depends on: [S4-01](01-serve-reaches-ready.md)
