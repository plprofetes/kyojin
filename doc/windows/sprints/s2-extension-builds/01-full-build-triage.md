# Full extension build: triage compile and link errors

- Status: todo
- Depends: S1-02
- Risk: HIGH volume of unknowns: 72 TUs, clang device code, MSVC host code, `moe_mul1.cpp` Windows branch marked UNVERIFIED by its authors.

Sprint: [Sprint 2: the full extension builds and imports](README.md) · ID S2-01 · priority order 1 in the sprint

**Goal.** `build` completes.

**Steps.**
1. `kyojin_win.py -v build --jobs 2` (start low), read `FAILURE.md`, fix, rebuild. Note: on Windows+HIP ninja tracks no header dependencies, so after editing a header delete `build\` (`build --clean`). To see every compile error in one round trip run `ninja -k 0` in the generated `build\temp.*` directory ([review notes](../../review/opus-r1-notes.md)). Near-certain stops, already identified: `hgemm.cu` HOME/2-arg `mkdir`, missing `rocblas` link library, `ext.py` JIT flags.
2. Suspect TUs from the assessment: [moe_mul1.cpp](../../../../exllamav3/exllamav3_ext/cpu/moe_mul1.cpp), [stloader.cpp](../../../../exllamav3/exllamav3_ext/stloader.cpp), [avx*_target](../../../../exllamav3/exllamav3_ext/avx2_target.cpp), [cuda_drv.cpp](../../../../exllamav3/exllamav3_ext/cuda_drv.cpp), [ngram_gather_win.cpp](../../../../exllamav3/exllamav3_ext/ngram_gather_win.cpp), [hgemm.cu](../../../../exllamav3/exllamav3_ext/hgemm.cu), [bindings.cpp](../../../../exllamav3/exllamav3_ext/bindings.cpp).
3. Gate every fix for Windows (`_WIN32`/`__linux__`, `windows` in `setup.py`) so Linux behaviour is unchanged.
4. Log each fix (file, cause, change) in the journal; add a signature for every new error shape.

**Done when.**
- `build` PASS; `.pyd` present; build time recorded.

**If it fails.** If a TU cannot compile: first understand why it is needed for Qwen. Excluding it (via `build_config.py`) is a last resort and must be recorded in [deferred.md](../../deferred.md) with the feature it disables.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [build_config.py](../../../../exllamav3/exllamav3_ext/build_config.py) · [assessment §3-4](../../assessment.md) · next: [S2-02](02-import-and-dll-resolution.md)

Depends on: [S1-02](../s1-risk-spikes/02-hip-extension-spike.md)
