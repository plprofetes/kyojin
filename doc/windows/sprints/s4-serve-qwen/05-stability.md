# Stability: sequential, long-context and soak

- Status: todo
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

**Links.** next sprint: [Sprint 5](../s5-harden/README.md)

Depends on: [S4-03](03-chat-completion.md)
