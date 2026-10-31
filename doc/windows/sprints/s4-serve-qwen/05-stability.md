# Stability: sequential, long-context and soak

- Status: active
- Depends: S4-03
- Risk: MEDIUM: leaks, driver resets (TDR) and allocator behaviour on WDDM only show over time.

Sprint: [Sprint 4: serve a small Qwen over HTTP](README.md) · ID S4-05 · priority order 5 in the sprint

**Goal.** No crashes or growth over a short soak.

**Steps.**
1. 10 sequential requests; one prompt near 4K tokens, then near the context limit used at launch.
2. Record GPU and host memory before/after (Task Manager or a probe); look for TDR events in Event Viewer.

**Done when.**
- No crash; memory returns to baseline within tolerance; numbers recorded.

**If it fails.** A TDR/driver reset: record driver version, kernel name from the log, and test with a smaller context.

**Evidence.** (run folders, versions, numbers; fill in)

- 
- 2026-10-31, Qwen3.8-27B EXL3 4.00bpw, `-- --draft-policy mtp`, `-c 8192`, thinking off, `max_tokens` 64, temperature 0: 13 sequential requests, all `finish_reason: stop` **[V: user's output]**. Request 1 (about 4,000-token prompt): 4,023 tokens in 73,993 ms (54 tok/s), decode 15.3 tok/s. Requests 2-10 (identical prompt): 1 new token each (prefix cache hit), 15-24 ms, decode 15.7-16.6 tok/s. Request 11 (about 7,200-token prompt sharing the first 4,023 tokens): 3,183 new tokens in 49,172 ms (65 tok/s), decode 17.1; requests 12-13 cached, decode 17.7-18.2. GPU dedicated memory (Task Manager counter) before 20.18 GiB, after 20.39 GiB (+0.21 GiB). No Display (TDR) events in the System log for the hour **[V]**. Not measured: server process working set (the `Get-Process python` query listed nothing; the process name is not `python` **[?]**).
- Limit of this run: nine of thirteen requests were prefix-cache hits, so they exercised little. Pending: a few requests with distinct long prompts. Prefill figures above are first-use figures (dense-GEMM classes may still tune at first use) **[?]**, not a benchmark.


**Links.** next sprint: [Sprint 5](../s5-harden/README.md)

Depends on: [S4-03](03-chat-completion.md)
