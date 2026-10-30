# util/hip_compiler.py and JIT .hsaco kernels on Windows

- Status: todo
- Depends: S2-02
- Risk: MEDIUM-HIGH. 14 modules call hipcc with Linux assumptions (`--gcc-install-dir`, `/usr/lib/gcc`, `~/.cache`); only the ones the Qwen path actually triggers matter.

Sprint: [Sprint 3: runtime kernels run](README.md) · ID S3-03 · priority order 3 in the sprint

**Goal.** Every JIT kernel the Qwen path needs compiles to `.hsaco` and loads.

**Steps.**
1. Run S3-01 with `EXL3_*` flags as `serve.py` sets them; read the JIT log lines to list kernels actually compiled.
2. Teach [hip_compiler.py](../../../../exllamav3/util/hip_compiler.py) `hipcc.exe`, skip `--gcc-install-dir` on Windows, use a Windows cache dir; fix the module-level `gcc` defaults in the used `*_hip.py` files only.
3. Leave the DS41/CK modules alone (see [deferred.md](../../deferred.md)).

**Done when.**
- List of used JIT kernels in the journal; each compiles and loads on Windows.

**If it fails.** If hipcc cannot produce `.hsaco` standalone (needs MSVC env), run the JIT under the captured VS environment and document it.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [example module](../../../../exllamav3/modules/gr_fn/gr_mix_hip.py) · next: [S3-04](04-gdn-fla-kernels.md)

Depends on: [S2-02](../s2-extension-builds/02-import-and-dll-resolution.md)
