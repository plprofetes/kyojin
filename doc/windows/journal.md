# Windows port: session journal

Append-only, newest entry last. One entry per working session: what was done, what was learned (mark **[V]** verified / **[I]** inferred / **[?]** unknown), what is next. Link task IDs. Put run-folder names and measured numbers here or in the task's Evidence section.

## 2026-10-30: assessment, orchestrator, planning (Sprint 0)

- Assessed the port: upstream exllamav3 has Windows builds for CUDA only; Kyojin inherits most Windows-aware host code; ROCm-on-Windows is new ([assessment.md](assessment.md)).
- **[V]** AMD index has Windows wheels for torch `2.15.0a0+rocm10.1.0a20260822` (cp310..cp314), `rocm-sdk-core/devel` `10.1.0a20260822`, `amd-torch-device-gfx1151`; wheel METADATA read over HTTP. torch's Windows `cpp_extension.py` has HIP-on-Windows code (`hipcc.exe`).
- **[V]** `triton-windows` lists RDNA3.5 (gfx115x) as supported (README, releases); the old woct0rdho README is stale. Correction of the first assessment, which called Triton a blocker.
- **[?]** Hugging Face unreachable from the assistant sandbox (proxy 403); model repo name unconfirmed.
- Built `scripts/windows/kyojin_win.py` (`doctor/setup/build/check/serve`), run logs, `FAILURE.md` diagnosis with 20 hypothetical signatures, 25 Linux unit tests. `setup.py` gated for Windows+HIP. None of it has run on Windows.
- Restructured into sprints with fail-fast ordering; added `tasks.py`.
- **Next:** [S1-01](sprints/s1-risk-spikes/01-preflight-green.md): run `doctor` and `setup` on the Windows machine and send the run folders.

## 2026-10-30: simplification

- Decision: no new scripts for the spikes. S1-02 (hand-written 30-line extension under gitignored `logs/windows/scratch`), S1-04 (pasted snippet) and S3-01 (existing `serve.py`, or a local uncommitted edit of `examples/generator.py`) use manual steps that are cleaned up by deleting a folder or `git checkout`.

## 2026-10-30: independent review, round 1

- Opus 5.5 subagent reviewed the assessment read-only ([review/opus-r1.md](review/opus-r1.md)). Verdict: risky but viable. I re-checked its key claims in the code: all confirmed except one it could not reproduce (triton-windows RDNA3.5 support, which is on branch `readme`, so our claim stands).
- Biggest corrections: `uv run` in the repo syncs the project (now `uv run --no-project`); clang-cl should be the host compiler (AMD does the same); `hgemm.cu` needs HOME/mkdir and rocBLAS fixes; serve.py forces HIP JIT and has two Windows crashes (symlinks, `register_at_fork`); ext.py JIT flags; assessment §5 wrongly listed absent wheels.
- Round 2 requested from the reviewer: Windows-gated patches (kept in /tmp, not applied), standalone FLA test, clang-cl hello-extension recipe, `long` audit, TDR notes, exact tool messages for the error catalogue. **Next:** review those, ask the user for consent before adding any new file to the repo.

- User decision: WSL2 is not a fallback path (Linux binaries, busy-poll GPU bridge, wrong performance picture); native Windows build and performance are the goal. WSL2 is a diagnostic at most.

## 2026-10-30: review round 2 applied

- Applied build-critical fixes (unverified on Windows): `setup.py` links hipblas+rocblas; `ext.py` raises a clear ImportError instead of a doomed JIT build; `hgemm.cu` `_mkdir`; `kyojin_win` environment (DISTUTILS_USE_SDK, clang-cl as CC/CXX, HOME, PYTHONUTF8); `build` now `setup.py build_ext --inplace` plus `ninja -k 0` on failure. Saved (not applied) patches for S3/S4 in `review/patches/`.
- `KNOWN` rebuilt from the reviewer's expected tool messages (34 signatures, 26 tests); `doctor` checks the tools/qwen symlinks; the script refuses to run from the repo `.venv`.
- Triton pin now `triton-windows==3.9.0.post30` (fallback `3.8.0.post29`).
- **Needs user consent:** adding `fla_standalone_test.py` (draft in `review/drafts/`). **Next:** S1-01 on the Windows machine.
