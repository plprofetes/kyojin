# Stability: sequential, long-context and soak

- Status: done
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
- 2026-10-31, second run of the same 13 requests (same prompts, same server settings; whether the server process was restarted in between is not stated **[?]**): all `finish_reason: stop`. Request 1 (4,023 tokens): **7,331 ms (549 tok/s)**, decode 17.3 tok/s; requests 2-10 cached (17-24 ms), decode 16.0-18.0 tok/s; request 11 (3,183 new tokens): **6,375 ms (499 tok/s)**, decode 17.7; requests 12-13 cached **[V: user's output]**. So the first run's 74 s and 49 s prefill were first-use costs (dense-GEMM classes tuning at first use after the 90 s warm-up budget, JIT/cache fills **[I]**), and warm prefill on the 27B + MTP is about 500-550 tok/s at 3-4K tokens.
- 2026-10-31, four requests with distinct prompts (each starts with a different "Document k.", so no prefix-cache hit), 4,207 prompt tokens each, `max_tokens` 64, temperature 0, same server (27B + MTP, 8192 ctx): prefill **7,542 / 8,045 / 7,909 / 7,905 ms = 558 / 523 / 532 / 532 tok/s**, decode 21.1 / 21.0 / 27.9 / 19.1 tok/s, all `finish_reason: stop`, 37 s in total; GPU dedicated memory afterwards **20.58 GiB** (20.18 before the whole soak, 20.39 after the first 13 requests) **[V: user's output]**. Over 17 requests (13 + 4) memory rose 0.40 GiB and stayed far from the 100 GiB ceiling; no crash, no unparsable reply, no Display (TDR) event seen after the first part (not re-queried after this part **[?]**). Not measured: server process host memory (name of the process not `python`). Decode varies 19-28 tok/s with the prompt because MTP acceptance varies.
- Done-when met at this scope (short soak; 17 requests; contexts up to about 7.2K of 8192). Not tested: long idle, hours of load, context filled to the limit with a distinct prompt.


**Links.** next sprint: [Sprint 5](../s5-harden/README.md)

Depends on: [S4-03](03-chat-completion.md)
