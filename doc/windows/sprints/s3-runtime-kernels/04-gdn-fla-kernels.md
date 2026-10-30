# Gated DeltaNet / FLA Triton kernels run on Windows

- Status: todo
- Depends: S1-03
- Risk: MEDIUM-HIGH (depends on S1-03). The vendored FLA kernels are large Triton programs; compile time and register pressure on RDNA3.5 are unknowns.

Sprint: [Sprint 3: runtime kernels run](README.md) · ID S3-04 · priority order 4 in the sprint

**Goal.** The Qwen3.5 GDN layers run through triton-windows with correct results.

**Steps.**
1. Run S3-01 and note Triton compile time and errors per kernel.
2. If a kernel fails to compile, record kernel name, Triton version and error in the journal; try the other pin (see S1-03).
3. Compare layer outputs against the torch/reference implementation where one exists.

**Done when.**
- GDN layers execute; first-run compile time recorded.

**If it fails.** If unfixable: Qwen3.5 is blocked on Windows; fall back to the decision in S1-03.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [vendor/fla](../../../../exllamav3/vendor/fla) · [gated_delta_net](../../../../exllamav3/modules/gated_delta_net.py) · next: [S3-05](05-numeric-parity.md)

Depends on: [S1-03](../s1-risk-spikes/03-triton-on-gfx1151.md)
