# First-launch tuning and warm-up complete and are cached

- Status: done
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
- 2026-10-30/31, Qwen3.5-9B EXL3 4.00bpw, `serve`, startup times as printed by `qserve: READY ... (start-up N s)`, in order: 236 s (cold caches; the HIP JIT chain failed on that run, so it measured Triton/rocBLAS work only), 104 s (hipcc fixed: first run that built the HIP kernels, includes `warm_dense_gemm` 93 s with "tune budget 90 s used"), 99 s (second HIP run), 24 s (run B: HIP kernels switched off by env), **9 s** (defaults again, nothing changed except removing the run-B env vars) **[V: user reports]**. So the second and later launches are fast once the caches (Triton `~\.triton\cache`, `.hsaco` in `%USERPROFILE%\.cache\exllamav3`, the dense-GEMM tune file) are populated. The dense-GEMM tuning needed several launches to complete because it stops after a 90 s budget per launch and continues at the next one **[I]**; where the tune file lives on this machine is not confirmed (see the journal).


**Links.** [tune_cache](../../../../exllamav3/util/tune_cache.py) · [kernel_seed](../../../../exllamav3/util/kernel_seed.py) · next: [S4-03](03-chat-completion.md)

Depends on: [S4-01](01-serve-reaches-ready.md)
