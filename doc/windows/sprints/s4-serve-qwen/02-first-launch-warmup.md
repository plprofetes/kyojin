# First-launch tuning and warm-up complete and are cached

- Status: todo
- Depends: S4-01
- Risk: MEDIUM: Kyojin tunes dense GEMMs at first launch (about 10 minutes on GLM) and JIT-compiles kernels; Windows file locking and cache paths could break caching.

Sprint: [Sprint 4: serve a small Qwen over HTTP](README.md) · ID S4-02 · priority order 2 in the sprint

**Goal.** Second launch starts fast.

**Steps.**
1. Time the first launch; read the server's progress lines.
2. Launch again; compare. Find where caches are written (`tune_cache`, `kernel_seed`, hip kernel cache).
3. Confirm cache files persist and are reused.

**Done when.**
- First and second launch times recorded.

**If it fails.** If caches are not reused, check path handling and file locks (`file-locked` signature).

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [tune_cache](../../../../exllamav3/util/tune_cache.py) · [kernel_seed](../../../../exllamav3/util/kernel_seed.py) · next: [S4-03](03-chat-completion.md)

Depends on: [S4-01](01-serve-reaches-ready.md)
