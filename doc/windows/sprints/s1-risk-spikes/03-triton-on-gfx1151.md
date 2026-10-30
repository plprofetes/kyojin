# Spike: Triton kernels run on gfx1151 via triton-windows

- Status: todo
- Depends: S1-01
- Risk: HIGH. 43 modules import triton at module level and Qwen3.5's Gated DeltaNet layers use the vendored FLA Triton kernels. Support is documented for RDNA3.5 but untested with Kyojin's kernels.

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-03 · priority order 3 in the sprint

**Goal.** Show that triton-windows compiles and runs both a trivial kernel and a real Kyojin kernel on this GPU.

**Steps.**
1. `kyojin_win.py check` and read the `triton` probe (vector add).
2. Test a real Kyojin kernel WITHOUT importing `exllamav3` (that would start the 72-file JIT build and fail on Windows, see [review #6](../../review/opus-r1.md)): put `exllamav3\vendor` on `sys.path` and `import fla`; its modules import nothing from exllamav3 at module level. A standalone chunk-kernel-vs-naive-recurrence test is being drafted in review round 2; adding it to the repo needs the user's consent.
3. If it fails, try the other triton-windows pin: `KYOJIN_TRITON_SPEC='triton-windows<3.10'` (3.9.0.post30 includes a VGPR fix for RDNA3.5), then `setup` again.
4. Note which C compiler Triton picked (clang-cl from the SDK vs MSVC) in the journal.

**Done when.**
- Vector add and the chosen Kyojin kernel match the reference; versions recorded.

**If it fails.** Decision gate: if Triton is unusable, Qwen3.5 (GDN layers) cannot run. Options: a different triton-windows build; torch fallbacks for the needed kernels (large work); a Qwen architecture without GDN (a dense full-attention Qwen, support in `serve.py` unverified [?]).

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [triton probe](../../../../scripts/windows/kyojin_win.py) · [vendored FLA kernels](../../../../exllamav3/vendor/fla) · [assessment §4.3](../../assessment.md) · next: [S1-04](04-gpu-memory-ceiling.md)

Depends on: [S1-01](01-preflight-green.md)
