# A chat completion returns coherent text

- Status: todo
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

- 

**Links.** next: [S4-04](04-api-parity.md)

Depends on: [S4-01](01-serve-reaches-ready.md)
