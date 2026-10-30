# Independent review, round 1 (Opus 5.5 subagent), 2026-10-30

Fresh-eyes review of [assessment.md](../assessment.md), the sprint plan and the tooling, done read-only against repo commit `cca1e17` with no Windows machine or GPU. Evidence log with source SHAs and dead ends: [opus-r1-notes.md](opus-r1-notes.md). The reviewer's text is model output; every row below says whether the assistant (author of the assessment) re-checked it. Verdict: **risky but viable**; nothing fundamental found, many near-certain day-one breaks with cheap fixes.

## Findings, with verification

| # | Finding | Checked by author? | Action |
|---|---|---|---|
| 1 | `uv run scripts/...` inside the repo syncs the project (`pyproject.toml` `[tool.uv] package = true`) and would install CUDA/CPU torch and build exllamav3 on the first command | Yes: `pyproject.toml:124-125` | Fixed: all docs and hints now say `uv run --no-project` |
| 2 | torch compiles Windows host `.cpp` with `CXX` (default `cl`) and `.cu` with `hipcc.exe`; AMD builds PyTorch on Windows with `CC=CXX=clang-cl.exe`, `HIP_CLANG_PATH`, `DISTUTILS_USE_SDK=1` | Yes: TheRock `build_prod_wheels.py` @ `682b3407` lines 701-706, 1204 (full SHA needed in the raw URL) | Make `clang-cl` the primary host compiler (S1-02, S2-01, `build`) |
| 3 | `bindings.cpp:39` includes `quant/exl3_kernel_map.cuh`, whose typedef uses `__restrict__` (lines 18-24); cl.exe will not accept it without CUDA's headers | Yes (code); the cl failure itself is inferred | Avoided by clang-cl; no source change unless clang-cl fails |
| 4 | `hgemm.cu:161-164` uses `getenv("HOME")` and 2-argument `mkdir`, not available on Windows | Yes | Windows-guarded fix pending (round 2 patches) |
| 5 | `hgemm.cu` calls rocBLAS directly (`rocblas_create_handle`, solutions API); the Windows link needs `rocblas` as well as `hipblas` | Yes (calls at `hgemm.cu:144,196,200`); `.lib` existence unknown | Add `rocblas` to `setup.py` libraries; S1-01 checks `lib\rocblas.lib` |
| 6 | `ext.py` JIT fallback still passes nvcc-style flags on Windows; any `import exllamav3` before a `.pyd` exists triggers a 72-TU JIT build that fails | Yes (`ext.py:100-108`) | Gate or fail early (patch pending); S1-03 must not import exllamav3 |
| 7 | `tools/qwen/{image_cache,serve_metrics,serve_ping,serve_reqlog}.py` are git symlinks (15 in the repo); Git for Windows defaults `core.symlinks=false` | Yes (`git ls-files -s`, mode 120000) | Clone with `-c core.symlinks=true` (Developer Mode), documented in S1-01 |
| 8 | `startup_health.py:269` calls `os.register_at_fork`, absent on Windows | Yes | Guard (patch pending) |
| 9 | `serve.py` `SERVE_ENV` forces `EXL3_GDN_FUSE/GDN_HF/PLE/DQ/GR_HIP` on, so the HIP JIT path (and `hip_lib.py`) is on the GDN path in serve.py, contradicting the plan | Yes (`serve.py:515,568`) | `hip_lib.py` Windows branch moves earlier; first smoke test avoids serve.py (see 11) |
| 10 | `hip_lib.find_hip_runtime` returns None on Windows; fix with `rocm_sdk.find_libraries("amdhip64")` (the DLL torch already loaded, `amdhip64_7.dll`) | Reviewer-verified name; not re-run | Patch pending |
| 11 | Better first-model ladder: tiny Qwen3.5 (0.8B/2B) with `EXL3_GDN_FUSE=0 EXL3_GDN_HF=0` via a short Model/Cache/Generator script or `examples/generator.py` -> 9B -> serve.py | Plausible; repo names unverified | Adopt; update S1-05, S3-01 |
| 12 | Test Triton without the extension: `vendor/fla` has no module-level exllamav3 imports, so a standalone chunk-kernel test runs in minutes | Partly (reviewer read it) | Round 2 deliverable `fla_standalone_test.py`; needs user consent before it enters the repo |
| 13 | Assessment §5 wrongly said `rocm`, `rocm-bootstrap`, `rocm-sdk-libraries` have no Windows wheels | Reviewer-verified; my grep filtered on "win" | Fixed in assessment §5 |
| 14 | `hip_compiler.py` needs no change for the POC (`which` honours PATHEXT, `--gcc-install-dir` only if the dir exists); S3-03 mostly unnecessary | Partly: `bin/hipcc` `isfile` checks lack `.exe` but the PATH candidate works | Merge S3-02 and S3-03 |
| 15 | ninja on Windows+HIP tracks no header dependencies; "incremental" is misleading after a header edit | Reviewer-verified in `cpp_extension.py:3191-3213` | Noted in S2-01 |
| 16 | TDR (2 s GPU watchdog) can reset the driver on long first launches; mitigations `TdrDelay`/`TdrDdiDelay`, smaller prefill chunk | Inferred | Round 2 deliverable; ask user for admin consent |
| 17 | LLP64: `long` is 32-bit on Windows, silent overflow risk in offsets | Inferred | Round 2 deliverable (audit only the Qwen path) |
| 18 | Paths: use `C:\k`, no spaces/non-ASCII in user name, Defender exclusion | Inferred, standard | In S1-01 |
| 19 | triton-windows RDNA3.5 support "could not be reproduced" | **Disputed**: the reviewer read `main`/`release/*`; the Windows README is on branch `readme`, lines 33-36 list "RDNA3.5 (gfx115x, Strix Halo) \| Yes" | Our statement stands; cited with branch |
| 20 | Delete the failure catalogue, `tasks.py`, `serve`/`check` | **Declined in part**: helpful errors are the user's top requirement; `tasks.py` stays by user decision | Prune the catalogue to evidence-based signatures (round 2) |
| 21 | Assessment §6 said "Not committed"; `kyojin_win.py` is about 1050 lines, not ~300 | Yes | Fixed |
| 22 | Commit subjects should follow repo style (imperative, no prefix) | Yes (`git log`: "Add opt-in ...", "Run the whole test suite ...") | Reword before any MR |

## Alternatives the reviewer ranked

1. Native build with `clang-cl` as host compiler (adopted as primary path).
2. WSL2 + ROCDXG, supported by AMD for Strix Halo with Adrenalin 26.2.2 + ROCm 7.2.1 per AMD's WSL guide; the reviewer is about 60% confident it would work for Kyojin. **Decision (user): not a path.** WSL2 builds Linux binaries, so it cannot produce a Windows build, and its GPU access goes through a busy-polling driver bridge, so its performance is not representative of native Windows, which is the goal. At most a diagnostic: it can show whether a failure is Windows-specific or already present on this torch/ROCm version.
3. AMD HIP SDK installer: version mismatch with pip ROCm 10.1 likely; low value.
4. Prebuilt wheels only after a build works.

## Merge-request advice (summary)

- Windows is new scope for Kyojin (CONTRIBUTING and README say nothing about it, CI is Linux only): open an issue first with a Windows run log.
- Mergeable: the Windows-gated `setup.py`/`ext.py` fixes, `hgemm` portability, `hip_lib` Windows branch, `register_at_fork` guard, one commands-only `doc/windows.md`. Also offer the HIP-on-Windows flag fixes upstream (CONTRIBUTING "Upstream").
- Not for an MR: `doc/windows/` sprint files, journal, assessment, `tasks.py`, `scripts/windows/CLAUDE.md`, the failure catalogue; probably `kyojin_win.py` (a second build path to maintain).
- Split: (1) platform guards, (2) `hgemm`/`bindings` portability, (3) `hip_lib`, (4) docs. Evidence wanted: Linux build plus CPU tests unchanged, a Windows log with hardware and driver versions.

## Round-trip protocol (summary)

One batched session, one zip of `logs\windows\*` plus a pasted `SUMMARY.txt` of at most 80 lines (commit and dirty count; Windows build, driver, VS, uv, Python; torch/HIP/arch/`mem_get_info`; SDK root and `bin\*.exe`, `lib\*.lib` listing; Triton version and compiler; per-step PASS/FAIL with seconds; first error block per failure; de-duplicated `ninja -k 0` errors per TU). Logs over 1 MB are trimmed to head 100, tail 200 and +-10 lines around matches; raw logs stay in the zip. Assistant answers: one copy-paste block, the expected output, then "send back: X". Adopting this needs a small change to `kyojin_win.py` (write `SUMMARY.txt`), which is an edit to an existing script.

## Round 2 (deliverables and what was done)

- Patches: [review/patches/README.md](patches/README.md) (4 applied, 3 saved for S3/S4).
- Hello extension recipe with clang-cl and the `ninja -k 0` recipe: [hello_ext_clangcl.md](hello_ext_clangcl.md). `kyojin_win build` now follows it (`setup.py build_ext --inplace`, then `ninja -k 0` on failure).
- `long` audit: [long_audit.md](long_audit.md): no `long` arithmetic on the Qwen3.5-9B dense path; watch list for MoE/DS41 later.
- TDR: [tdr.md](tdr.md): registry values and detection; the CPU-MoE handoff wait kernels (up to about 30 s) will certainly trip it, the 9B probably not.
- Error catalogue: [error_catalogue.md](error_catalogue.md) was applied to `KNOWN` in `kyojin_win.py` (34 signatures, each with a sample test; dropped `msvc-flag-on-clang`, `triton-compiler`, `device-libs`).
- New finding (verified by the reviewer in torch's code): `build` would have failed at once without `DISTUTILS_USE_SDK=1` after VsDevCmd; fixed.
- Reviewer's concession: the triton-windows `readme` branch (`7b92538`) does list RDNA3.5/Strix Halo. Its compatibility table pairs torch 2.15 with Triton 3.9, so the default pin changed to `triton-windows==3.9.0.post30` (fallback 3.8.0.post29).
- Draft awaiting consent: [drafts/](drafts/README.md) (standalone FLA Triton test).
- Not verified by anyone: exact cl.exe error codes, whether `hipblas.lib`/`rocblas.lib` exist in the devel tree, HIP error text after a TDR, Event ID 4101 (common knowledge).
