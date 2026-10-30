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
