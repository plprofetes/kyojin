# Windows equivalents for the Linux-only memory code

- Status: todo
- Depends: S4-05
- Risk: LOW-MEDIUM: quality of life; matters for large packs and long runs.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-05 · priority order 5 in the sprint

**Goal.** `util/memory.py` and `util/memguard.py` report real host memory on Windows and stop printing Linux-only failures (`swap_in_process` TypeError line).

**Steps.**
1. List what runs at startup on Windows and fails quietly (`swap_in_process` reads `/proc`; `host_memory_available` returns None, which disables the host-memory reserve check).
2. Use `GlobalMemoryStatusEx` (ctypes) for available physical memory; skip the swap read-back on Windows with one log line.
3. Unit tests with injected readers; Linux paths untouched.

**Done when.**
- No Linux-only error text in a Windows start-up log; the reserve check works.

**If it fails.** Keep the quiet degradation if a Windows API is unreliable inside the sandbox of the user's machine.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [deferred.md 1.2](../../deferred.md)

Depends on: [S4-05](../s4-serve-qwen/05-stability.md)
