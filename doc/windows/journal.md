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

## 2026-10-30: first Windows run (doctor)

- User ran `doctor` on the Windows machine: everything PASS except the `gpu` probe (my bug, fixed). Key facts recorded in [S1-01](sprints/s1-risk-spikes/01-preflight-green.md): VS 17.14 + SDK 10.0.28000.0, uv 0.13.6, locale is Polish, **Windows sees only 31.6 GB of the 128 GB RAM (about 96 GB presumably reserved for the GPU)**, 6.6 GB free.
- Earlier on the same machine: a `uv run` spawn error (os error 3) because the clone was not on the port branch yet; resolved by checking out `claude/modest-rubin-p69b5g`. Symlinks fixed with Developer Mode + `core.symlinks=true`.
- **Next:** `setup`, then `check`, manual checks (S1-01 step 5, S1-04), hello extension (S1-02), first `build --jobs 1`.

- Windows finding: `uv run --no-project scripts/windows/kyojin_win.py ...` fails on the user's machine (uv 0.13.6, PowerShell) with "Failed to spawn ... os error 3" although the file exists; the explicit form `uv run --no-project python scripts/windows/kyojin_win.py ...` is used everywhere now (plain `python scripts\windows\kyojin_win.py` also works, the script is stdlib-only). Second `doctor`: PASS=9 FAIL=0 INFO=2.

## 2026-10-30: `setup` ran on Windows

- `setup` succeeded end to end (see [S1-01 evidence](sprints/s1-risk-spikes/01-preflight-green.md)); toolchain is real: hipcc 7.16 / clang 23 for `x86_64-pc-windows-msvc`, gfx1151 target, headers present.
- Probes `torch_gpu`/`triton` failed on low memory (OpenBLAS), not on the stack. Fixes: probes cap BLAS threads; PATH for `rocm-sdk test`; dead registry probe removed; `openblas-oom` signature; console heartbeat every 30 s when not `-v` (the user does not want to see all compiler output; run without `-v`, everything is in the run folder).
- User will continue **in the next session**. **Next:** close the program using the GPU/RAM, `git pull`, run `check` (no `-v`), then the manual checks (`dir` for hipblas.lib/rocblas.lib; S1-04 memory snippet), hello extension (S1-02), first `build --jobs 1`.

- `check` under load (other LLM on the GPU): torch probe hit `LLVM ERROR: out of memory` (not a toolchain verdict), Triton probe exposed my `python -c` bug (fixed: probes run from a file; torch probe prints `STEP` markers). Re-run `check` when the GPU/RAM are free.
- Triton PASS on Windows/gfx1151 (vector add, triton-windows 3.9.0.post30): first positive result for the biggest unknown behind S1-03. torch GPU probe still blocked by the other workload (`HIPBLAS_STATUS_ALLOC_FAILED`); `gpu-alloc-failed` signature added.
- Key data: HIP sees 107.87 GiB GPU memory (107.72 free) on Windows, and the `hipblasCreate` ALLOC_FAILED happens despite that: the 'other workload exhausted GPU memory' theory is not supported. Probe extended (elementwise vs BLAS per backend); `gpu-alloc-failed` signature no longer blames the other process. **Next (can run with the other workload up):** `check` again, hello extension (S1-02, tiny compile), single 16 GiB allocation.

## 2026-10-30: paused, waiting for the GPU

- GPU and most RAM are held by another LLM project of the user; `mem_get_info` over-reports free VRAM on Windows. Decision: wait. Nothing else to run until the machine is free.
- Resume with: `git pull`; stop the other workload; `check` (probe now reports the largest allocation that works and matmul per BLAS backend); `dir` for `lib\hipblas.lib`/`lib\rocblas.lib`; hello extension (S1-02); `build --jobs 1`. Send the run folders.
- The other workload is GLM-4.5-Air Q4 (user): confirms that VRAM is held by it and `mem_get_info` over-reports free VRAM on Windows. Kyojin uses `mem_get_info` in 11 files (see S4-01 risk note). Open question for the user: Task Manager > Performance > GPU, Dedicated vs Shared GPU memory in use while it runs.
- Task Manager numbers (user): Dedicated 95.2/95.8 GB used by the other runner, Shared 2.4/15.8. Explains the failed 1 GiB allocation; HIP allocations appear to come from the dedicated pool (usable ceiling about 95 GiB, not the 107.87 GiB torch reports) **[I]**. Option to keep working in parallel: restart the other runner with a smaller context so about 20 GB of dedicated memory stays free.

## 2026-10-30: hardened Python and the Triton test (user consent given)

- Added `scripts/windows/fla_standalone_test.py` (user consented): real vendored GDN chunk kernels vs a naive fp32 recurrence, no extension build needed; never run (no GPU in the authoring sandbox).
- Hardened `setup`: exact uv-managed interpreter `3.12.16` (`KYOJIN_PY`, `--managed-python`), `--no-config` on every uv call, wheel tag derived from major.minor. The existing `.venv-win` already uses 3.12.16 managed; `setup --recreate` rebuilds it under the new rules (optional).
- **Next session:** as in the previous entry, plus run `fla_standalone_test.py` right after `check` (S1-03).

## 2026-10-30: first `build` attempts on Windows

- `build` #1 failed with `[WinError 10106]` (Winsock provider) on `import asyncio` in the child. `capture_vs_env` returned the raw `cmd /c set` output (mixed-case keys), so `sdk_env` added a second `PATH` next to `Path`; now merged onto `os.environ` (`merge_env`) **[I]**; fixed in the next run, cause not separately proven.
- `build` #2 reached the compiler and failed: `clang: error: no such file or directory: '/EHsc'` (and `/MD`, `/wd...`). Cause **[I]**: `.venv-win\Scripts` was not on PATH, torch's `ninja --version` failed and `BuildExtension` fell back to the serial distutils path, which emits `-Xcompiler <msvc flag>` pairs; clang ignores `-Xcompiler` but not its argument. Fix: `build` prepends the venv `Scripts` directory to PATH; signature `no-ninja-fallback`.
- Docs corrected: the earlier [V] claim "clang ignores `-Xcompiler`, so torch's pairs work" (review/opus-r1-notes.md, review/hello_ext_clangcl.md) was wrong for MSVC flags; assessment §Flags no longer lists `-Xcompiler=` for ROCm. Lesson: a "[V] from source" about a flag-ignoring rule did not cover the argument after it.
- **Next:** `git pull`; `build --clean` (watch for ninja `[n/N]` progress); send the run folder on failure.

## 2026-10-30: first successful compile of the HIP sources

- `build` #3 (ninja on PATH): all device TUs compiled; one host TU failed: `cpu/moe_mul1.cpp` with `always_inline function '_mm256_*' requires target feature 'avx'/'f16c'`. Cause **[V from the log, I for the fix]**: the `M1_TARGET_*` macros were gated on `__GNUC__ && __linux__`, so under clang-cl they were empty (the empty branch is right for cl.exe only). Fix: gate on `... || defined(__clang__)` (Linux unchanged: `__GNUC__ && __linux__` still true there). Signature `cpu-target-feature` added.
- Harmless noise seen: `argument unused during compilation: '-Xcompiler'`; the `Assuming 100 MHz realtime clock rate (TSC) for gfx1100/gfx1101` #warning from cuda/std libcxx (device-side chrono only; the only device sleep is `__nanosleep` in the multi-GPU barrier).
- **Next:** `git pull`; `build` (no `--clean`, the other objects are reusable); then the link step: watch for `LNK2019` (rocblas/hipblas) and `LNK2001` (`__builtin_*`, `WaitOnAddress`).

## 2026-10-30: link failure `command 'link.exe' failed: None`; correction of the WinError 10106 diagnosis

- `build` #4 compiled everything, then failed at the setuptools link: `build-env.json` PATH had **no MSVC/Hostx64 entry**, so `link.exe` was not found (setuptools names the bare `link.exe`). Cause **[V]**: `capture_vs_env` passed ONE string with quotes to `cmd /c`; Python's `list2cmdline` turns the inner quotes into `\"`, which cmd does not unescape, so `call "VsDevCmd.bat"` failed and `&& set` never ran: the captured environment was empty. Compile still worked because clang-cl locates VS on its own. Fix: separate argv items (`vsdevcmd_cmd`), unit test on the generated command line, and `build` now stops with a `vs_env` FAIL if INCLUDE/LIB/MSVC-in-PATH are missing instead of continuing silently.
- **Correction:** the `[WinError 10106]` in build #1 was most likely this same empty environment (no `SystemRoot` for the child, so Winsock could not load its providers), not the `Path`/`PATH` duplication I blamed. `merge_env` (copy the parent environment, then apply VsDevCmd on top) masked it and stays as a safety net, but the diagnosis in the build #1 entry above is superseded. Still **[I]** for 10106, **[V]** for the quoting bug.
- **Next:** `git pull`; `build` (compiled objects are reused, so it should go straight to the link). Watch for `LNK2019`/`LNK2001`.

## 2026-10-30: link: `LNK2001 __truncsfhf2`

- `build` #5 compiled and linked everything except one symbol: `stloader_cu.obj : LNK2001 __truncsfhf2` (compiler-rt float->half helper, from the host half of `__float2half_rn` in `inplace_bf16_to_fp16_cpu`) **[I]**. clang-cl links `clang_rt.builtins` by default; setuptools calls `link.exe` directly, so the MSVC CRT is all it sees.
- Found on the user's machine: `_rocm_sdk_devel\lib\llvm\lib\clang\23\lib\windows\clang_rt.builtins-x86_64.lib` (a recursive `-Filter` search of the devel tree missed it, a search from `.venv-win` with `-Include` found it; junction/reparse point **[?]**). Fix: `setup.py` (Windows+HIP only) passes it as `extra_link_args`, raising if absent. Fallback if that is not enough: `-Xarch_host -mf16c` (unverified clang flag). Signature `compiler-rt-builtin` added.
- **Next:** `git pull`; `build` (objects reused). After a green link the script runs the `extension` probe; then `check`.

## 2026-10-30: extension links; first import fails with `DLL load failed`

- `build` green and linked. The `extension` probe failed with `DLL load failed while importing exllamav3_ext`. The probe imported the extension before `torch`; Python 3.8+ does not resolve an extension's dependent DLLs from PATH, and the ROCm/torch DLLs are only loaded once `torch` is imported **[I]**. Probe changed to `import torch` first (S2-02). If it still fails, the next step is `os.add_dll_directory(<sdk>\bin)` and a dependency dump (`dumpbin /dependents`) to name the missing DLL.

## 2026-10-30: `import exllamav3` fails: `cannot import name 'models' from 'tokenizers'`

- The extension imports once torch is loaded (`import torch, exllamav3_ext` OK). `import exllamav3` then failed at `exllamav3/tokenizer/tokenizer.py:4` **[V]**: the venv has **tokenizers 1.0.0-rc.3** (a flat `tokenizers.pyd`, no `models`/`decoders` subpackages) **[V: directory listing]**. `requirements.txt` says `>=0.21.1`; why uv chose the pre-release is not established (the requirements step passes no `--prerelease`) **[?]**. PyPI has 0.23.3 (cp310-abi3 win_amd64 wheel, the version in `uv.lock`).
- Fix: `setup` pins `tokenizers==0.23.3` next to the triton pin (`TOKENIZERS_SPEC`, env `KYOJIN_TOKENIZERS_SPEC`), recorded in `env.json` pins and assessment 7d. Signature `tokenizers-rc`. The repo's own `requirements.txt` is untouched (Linux).
- Immediate repair without `setup`: `uv pip install --no-config --python .venv-win\Scripts\python.exe tokenizers==0.23.3`. Other dependencies may have drifted the same way; compare `pip-freeze.txt` with `uv.lock` (deferred check).
