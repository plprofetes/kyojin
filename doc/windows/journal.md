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

## 2026-10-30: `check` all green (S2-01, S2-02 done)

- After pinning `tokenizers==0.23.3`: `check` `20261030-233000-check` PASS=4 (rocm_sdk, torch_gpu 16 GiB single allocation + matmul on all BLAS backends, triton 3.9.0, extension 135 symbols + `import exllamav3`) **[V]**. The Windows build of the extension and its import work on gfx1151; GPU kernels have not been exercised yet.
- Pinning: the rest of `requirements.txt` is still `>=`; user wants strict pins. Plan: constraints file from a known-good `pip freeze` of this venv (next entry).
- **Next:** pin from the freeze (below); `fla_standalone_test.py` (S1-03); then S3-01 smoke (one real kernel call).

## 2026-10-30: all Python packages pinned (`constraints-win.txt`)

- `uv pip freeze` of the working venv showed two more pre-releases besides tokenizers: **pydantic 2.14.0b1, safetensors 0.9.0rc1** **[V]**; why uv allows pre-releases in the requirements step is not established (`UV_PRERELEASE` is not set here; the user's environment was not inspected) **[?]**. Neither was exercised yet (no model loaded).
- New `scripts/windows/constraints-win.txt` (exact `==`, no pre-releases): the freeze, with pydantic/pydantic-core/safetensors replaced by the `uv.lock` versions. `setup` passes it as `-c` to both `uv pip install` steps; `env.json` records its sha256. Replaces the single `TOKENIZERS_SPEC`. Test: exact versions, no pre-release, every `requirements.txt` package pinned. torch/rocm/triton stay pinned by the constants. `requirements.txt` untouched (Linux).
- The user's current venv still has the two pre-releases until `setup` or the command in the next message is run.

## 2026-10-30: Triton GDN kernels pass on gfx1151 (S1-03 done)

- `fla_standalone_test.py` PASS on the target: o err 2.3e-4, state err 1.1e-3 against the fp32 recurrence; cold 49.9 s, warm 0.7 ms; triton-windows 3.9.0.post30 **[V]**. The biggest unknown behind Qwen3.5 (GDN layers) is therefore cleared at small shapes; `--qwen` (T=4096) not run.
- **Next:** S1-05 (model pack on disk) and S3-01 (end-to-end smoke via `serve`). Expected first failures from the catalogue: `register_at_fork` in `tools/startup_health.py` (patch saved), `libamdhip64.so` in `util/hip_lib.py` (S3-02 patch saved), POSIX-only memory/pagecache code.

## 2026-10-30: constraints file was inconsistent (fixed)

- The user's `uv pip install -c constraints-win.txt pydantic pydantic-core safetensors` failed: stable `pydantic==2.14.0` needs `typing-extensions>=4.16.0`, the freeze had 4.15.0 (matching the beta) **[V: uv error, reproduced here]**. My error: I had checked wheel availability but not that the set resolves. Fix: `typing-extensions==4.16.0` (the `uv.lock` version). Verified with `uv pip compile --python-platform windows --python-version 3.12 -c constraints-win.txt` over `requirements.txt` (minus torch) + `triton-windows==3.9.0.post30`: resolves, every resolved package is pinned, no version differs. The header of the file now records this command (rule: re-run it after every edit).
- Not covered by that check: torch's own dependencies on the AMD index (`sympy`, `networkx`, `mpmath` are pinned but were not part of the compile) **[?]**.

## 2026-10-30: constraints applied on the target; model download started

- The corrected `constraints-win.txt` installed cleanly in the existing venv (`uv pip install -c ... pydantic pydantic-core safetensors typing-extensions`) **[V: user report]**; a follow-up `check` result was not reported. Pre-releases pydantic 2.14.0b1 / safetensors 0.9.0rc1 are gone from the venv.
- Model: `turboderp/Qwen3.5-9B-exl3`, branch **`4.00bpw`** (not `4.0bpw`; the branch exists, given by the user's URL), downloaded with `.venv-win\Scripts\hf.exe download ... --revision 4.00bpw --local-dir C:\models\qwen35-9b-4bpw` (the `hf` CLI ships in the venv via huggingface-hub 2.4.0). File list and size still to be recorded.

## 2026-10-30: first `serve` run: `os.register_at_fork` (S4-01)

- Pack `turboderp/Qwen3.5-9B-exl3` @ `4.00bpw` is on disk: one `model.safetensors` of 7,185,603,988 bytes (6.69 GiB) plus `config.json`, `quantization_config.json`, tokenizer files, `chat_template.jinja`, preprocessor configs; **`generation_config.json` is absent** (the `model_dir` check WARNs; serve falls back to defaults) **[V: dir listing]**.
- `serve` died in 2 s at `tools/startup_health.py:269`: `AttributeError: module 'os' has no attribute 'register_at_fork'` **[V]**, as predicted by the catalogue. Applied the saved patch (`hasattr` guard; `allow_reuse_address = os.name != "nt"` because SO_REUSEADDR on Windows lets a second process bind the same port). Linux: both conditions evaluate as before **[I]**, not run.
- Also: `serve` now sets `PYTHONUTF8=1` for the child (cp1250 locale vs `read_text()` without encoding) instead of applying the UTF-8 patch to `tools/qwen/serve.py`.
- **Next:** `git pull`, fetch `generation_config.json` from the base repo (optional), `serve` again; expect S3-02 (`libamdhip64.so`) next.

## 2026-10-30: serve reaches READY on Windows (S4-01 done)

- After the `startup_health` fix: `qserve: READY on http://127.0.0.1:8000 model=Qwen3.8-Flash-Yamz ctx=8192 sessions=1 speculative=False vision=False (start-up 236 s)` **[V: user report]**. Model loaded, kernels compiled/autotuned (cold Triton cache); no `libamdhip64.so` error reported, so the S3-02 path may not have been hit yet (JIT HIP kernels may only run on the first request) **[?]**.
- **Next:** S4-03: one chat completion with thinking disabled (below), then `/health`, `/v1/models`, streaming. Watch the server console during the first request for JIT-kernel failures (`libamdhip64`, hipcc resolution).

## 2026-10-30: first generated text on Windows (POC goal reached at smoke level)

- `serve` on Windows/gfx1151 answered a chat completion: "2+2 equals 4.", 25 prompt tokens in 231 ms, 8 tokens in 573 ms = 13.97 tok/s, no speculative decoding, Qwen3.5-9B EXL3 4.00bpw **[V: user's curl output]**. The model name in the request must be the served name `Qwen3.8-Flash-Yamz`.
- Path so far: `setup` -> `build` (extension links with `clang_rt.builtins`, ninja on PATH, VsDevCmd env) -> `check` PASS -> pinned requirements -> Triton GDN test PASS -> `startup_health` patch -> `serve` READY (236 s cold) -> completion.
- Not done / unverified: coherence on longer text (numeric parity vs Linux, S3-05), streaming, `/health`, `/v1/models`, multi-turn, speed vs Linux, warm start time, whether JIT HIP kernels were exercised (S3-02/03), Linux regression check of the shared-file edits (deferred.md, S2-03). Decode 14 tok/s is from 8 tokens on a cold server; not a benchmark.

## 2026-10-30: the JIT HIP path does not work (user found it in the log) - hipcc not found

- The user pointed out two things in the `serve` log that I had missed. (1) `EXL3_GDN_HF failed, Triton/HIP chain for the rest of the process: RuntimeError('hipcc not found ...')`; `/health` shows `"hip_compiler": "none found"` **[V]**. My earlier "no `libamdhip64` error, so S3-02 was not needed" was wrong: the JIT kernels failed before they got that far, and I had not read the log. (2) `swap_in_process` reports `TypeError("argument of type 'NoneType' is not iterable")` **[V]**.
- Cause of (1) **[V by reading the code, reproduced in a temp tree]**: `exllamav3/util/hip_compiler.py` tests `<sdk>/bin/hipcc` with `os.path.isfile`; the Windows SDK has `bin\hipcc.exe` (and `hipcc.bat`), no extension-less file; `EXL3_ROCM_SDK` was set correctly by `serve`. The same applies to `Scripts\rocm-sdk.exe`. Fix: `_EXE = ".exe" if os.name == "nt"` and a `_hipcc_in()` helper; Linux names unchanged. Unit test added (`test_windows_sdk_layout_hipcc_exe`; the file needs torch, so its body was run standalone here). Also applied `hip_lib_windows.patch` (S3-02) since the next step after finding hipcc is loading the compiled `.hsaco` through the HIP runtime. Signature `jit-hipcc-missing`. Result with the GDN HF kernel on: **[?]** until the user re-runs `serve` and checks `/health`.
- (2) is the Linux-only `swap_in_process` (`ctypes.CDLL(None)` and `/proc/*`): it never raises by design and `serve` continues; noise only, covered by the `/proc` row in deferred.md 1.2. Not changed.
- Consequence for the earlier smoke result: the 13.97 tok/s answer ran with the Triton/PyTorch chain, not the fused HIP kernels; after this fix speed and numerics can differ.

## 2026-10-30: JIT HIP kernels on: coherent 200-token answer at 30 tok/s

- After the `hipcc.exe` fix: `serve` READY in 104 s (warm-up of dense GEMM tuning took 93 s of it; the 236 s of the cold run included first Triton/HIP compilation), `/health` reports the SDK `hipcc.exe` (clang 23.0.0git, HIP 7.16.26332). The log grep for failures is empty **[V: user report]**.
- Request 2: coherent, correct 200-token answer; prompt 28 tok / 1519 ms, decode 200 tok / 6590 ms = **30.3 tok/s** (earlier fallback-chain run: 13.97 tok/s over 8 tokens) **[V]**. Prefill speed (18 tok/s on 28 tokens) is a first-request figure with first-use costs, not a measurement. No Linux reference on this machine yet.
- Also seen: `warm_dense_gemm: tune budget 90 s used, remaining classes tune at first use` (so later requests may still hit one-time tuning), `swap_in_process` TypeError noise (benign, Linux-only code).
- **Done:** S3-02 (hip_lib + hipcc lookup). **Open:** streaming, a second identical request (warm numbers), the 16-token fixed-prompt evidence for S3-01, numeric parity vs Linux (S3-05), Linux regression check (deferred), memory ceiling sweep before Flash-Next.

## 2026-10-30: warm numbers; `serve` passes extra flags to serve.py

- Same request again (warm, sampling at the card defaults, so different text): prompt 28 tok / 220 ms (127 tok/s), decode 207 tok / 6404 ms = **32.3 tok/s**, coherent answer **[V: user's curl output]**. First request had 30.3 tok/s and 1519 ms prompt (first-use costs). Single runs, temperature 1.0 so lengths differ; for comparisons use `temperature: 0`.
- User wants to test MTP, flags and optimizations on the 9B pack instead of the 90 GB model. `kyojin_win.py serve` now forwards everything after `--` to `tools/qwen/serve.py` (appended last, so it overrides the `--draft-policy off` default): `serve --model <dir> -- --draft-policy mtp`. Environment variables (`EXL3_*`) already pass through from the shell. Plan: greedy request (temperature 0) with `off`, `mtp`, `mix`; speculative decoding is lossless, so identical text is also a check of the verify kernels on Windows **[plan, I]**. Whether the 4.00bpw pack contains the MTP drafter weights is **[?]**.

## 2026-10-30: greedy baseline; this pack has no MTP weights

- Baseline (temperature 0, `--draft-policy off`, 9B 4.00bpw, thinking off, "about 150 words on hash table collisions"): prompt 28 tok / 219 ms (128 tok/s), decode 223 tok / 6749 ms = **33.0 tok/s**, coherent, `finish_reason: stop` **[V: user's curl output]**. Streaming works (`stream: true` from a request file with `curl.exe -N`) **[V: user report]**.
- `serve ... -- --draft-policy mtp` fails at load: `ValueError: Required tensor mtp.pre_fc_norm_hidden.weight not found in any *.safetensors file` (`loader/safetensors.py:727`, via `model.py` load of the `mtp` component) **[V: log]**. Cause: turboderp's `Qwen3.5-9B-exl3` 4.00bpw pack contains no `mtp.*` tensors (the user's header scan found none); Kyojin's converter writes them with `convert.py -mb/--mtp_bits` (doc/convert.md) and the README's MTP numbers are for Qwen3.8-27B packs. `mix` also needs the drafter (MTP + n-gram), so with this pack only `off` works.
- MTP on Windows therefore needs a different pack (a Kyojin/yamz-labs pack that includes the MTP stage, repo not checked **[?]**) or a conversion with `-mb`.

## 2026-10-30: Hugging Face reachable from the sandbox; an MTP pack exists (checked via the HF API)

- The user added `huggingface.co` to the environment's allowed domains; the API now answers (earlier: proxy 403 on CONNECT). Everything below was read from the HF API and raw files **[V]**; third-party content treated as data.
- `HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3` is **not a model**: a sidecar with only the `mtp.*` tensors of the Qwen3.8-27B head (`mtp-3.1bpw/3.7bpw/4.4bpw.safetensors`, 158-224 MiB) that shadows a base pack's own MTP weights (files load alphabetically, later wins). Useless for the Qwen3.5-9B pack; an optional optimisation for 27B packs that already have MTP.
- `turboderp/Qwen3.8-27B-exl3` branch **`4.00bpw`**: 15.73 GiB (`model-0000{1,2}-of-00002.safetensors`, `generation_config.json`, `chat_template.jinja`, tokenizer files), `model.safetensors.index.json` lists **39 `mtp.*` tensors** of 2426 **[V]**. This matches the README's tested config (Qwen3.8-27B 4.0 bpw, MTP + n-gram) and fits the GPU easily. Variants `SC_4.00bpw_H5` / `SC_4.00bpw_H5_V6` (15.26 GiB, quantized vision tower, needs ExLlamaV3 1.4.4 per the card) also have 39 mtp tensors.
- yamz-labs hosts GLM-5.3-Flash, MiMo, Flint, Nanbeige4.3-33B, DeepSeek-V4.1-Flash, **Qwen3.8-Flash-Next-EXL3-Yamz** (size not checked), but no small Qwen pack.
- Plan: download the 27B `4.00bpw` pack in the background; meanwhile S4-02 (second-launch time) and S4-05 (soak) on the 9B pack; then MTP on 27B.

## 2026-10-30: parity plan without a Linux box

- User has no Linux machine for this device. The 9B model card has only a KLD plot (no numbers) **[V: HF README]**, so there is no published figure to compare against. Plan for S3-05/S4-04: A/B on Windows between the default HIP chain and the engine's own fallback (`EXL3_PLE_HIP/DQ_HIP/GR_HIP/GDN_FUSE/GDN_HF=0`), which `serve.py` documents as bitwise-equal paths; compare greedy ids and logprobs (`logprobs`, `top_logprobs`, `return_token_ids`, non-streaming). Limits: it does not validate the stack against another platform or against the unquantized model.
- `~/.cache/exllamav3/*.hsaco` (on Windows `%USERPROFILE%\.cache\exllamav3`) is where the JIT kernels land (`gdn_hf_*`, `gr_mix_*`, `ple_*`, `qsa_pf_*`); its listing is the S3-03 evidence. The default serve log shows only the compiler line.

## 2026-10-30: JIT kernel list (S3-03)

- `%USERPROFILE%\.cache\exllamav3` holds `gdn_hf_gfx1151_bfe9e89a80c14757.hsaco` (28,288 B) and `ple_gfx1151_a080d964d7fd73d4.hsaco` (34,392 B), both 31.10.2026 00:16:21-22, i.e. built by the first `serve` after the `hipcc.exe` fix **[V]**. No `gr_mix_*` or `qsa_pf_*` file for this dense 9B pack **[V]**; why is not established **[?]**.

## 2026-10-31: S3-05 parity A/B identical; startup 99 s vs 24 s; HOME for the tune caches

- A/B (HIP kernels on vs the fallback path, greedy, 48 tokens, ids and logprobs): `choices[0]` **identical** **[V: user's comparison]**. S3-05 done at that scope (see the task file).
- Startup: A 99 s, B 24 s **[V: user report]**. The difference is presumably the dense-GEMM warm-up/tuning (`warm_dense_gemm`: 93 s, "tune budget 90 s used") that only the HIP chain runs **[I]**; B does not tune, A does, and A did not get faster on its second launch (104 s, then 99 s), i.e. the tune cache is not being reused.
- Likely cause **[I, from the code comments]**: the tune file is `$HOME/.cache/exllamav3/dense_gemm_tune.txt` (`hgemm.cu:165-173`, `dense_warmup.py`), `/tmp` when `HOME` is unset, and `serve` did not set `HOME` (only `build` did): on Windows that is `<drive>:\tmp\.cache\...`, where the non-recursive `_mkdir` can fail. Fix: `serve` sets `HOME=%USERPROFILE%` (setdefault). To verify on the next launch: `dense_gemm_tune.txt` appears in `%USERPROFILE%\.cache\exllamav3` and the launch after that is fast.

## 2026-10-31: correction: the tune cache works; startup is 9 s

- The user cleared the run-B variables and started with defaults and nothing else changed: **9 s** startup **[V: user report]**. Series: 236 s, 104 s, 99 s, (B, kernels off) 24 s, 9 s.
- This contradicts the previous entry's explanation (HOME unset, so the tune file is not reused). The tune cache evidently does persist; a more consistent reading **[I]** is that `warm_dense_gemm` stops after 90 s per launch ("remaining classes tune at first use"), so each launch added to the file until every class was covered, then the warm-up became a few seconds, as its docstring says for a populated file. Whether the file sits in `<drive>:\tmp\.cache\exllamav3` (HOME unset) or elsewhere is **[?]**; I did not verify it. The `HOME=%USERPROFILE%` change to `serve` that I pushed is therefore not a fix for a verified problem. It only moves the cache into the user profile (next to the `.hsaco` files); the cost is one more cold tuning period. Decision pending the directory listings requested from the user; revert it if the cache is found working in `C:\tmp`.

## 2026-10-31: tune cache location verified; HOME change reverted

- `C:\tmp\.cache\exllamav3\dense_gemm_tune.txt` (34,928 B, written 31.10.2026 01:00:36) exists and the cache works from there; `%USERPROFILE%\.cache\exllamav3` holds only the two `.hsaco` files **[V: user's listings]**. Cause of the earlier slow launches stays as in the previous entry **[I]**. With `HOME` unset, hgemm.cu/dense_warmup.py use `/tmp`, i.e. `C:\tmp` on this machine, and that works.
- Reverted the `HOME=%USERPROFILE%` default in `serve` (it only relocated a working cache and forced a re-tune). Build env still sets HOME. Moving the tune cache into the user profile is a hygiene item, not done: see deferred.md.
- User: the Qwen3.8-27B 4.00bpw download finished. Next: MTP on it (`serve ... -- --draft-policy mtp`), comparing `speculative: true` vs `false` per request on one server.

## 2026-10-31: MTP works on Windows (Qwen3.8-27B 4.00bpw)

- `turboderp/Qwen3.8-27B-exl3` @ `4.00bpw` (15.73 GiB, 39 `mtp.*` tensors) served with `-- --draft-policy mtp`; same server, `temperature: 0`, 150-word hash-table prompt, `speculative` true (default) vs `false` per request: **text identical (True)**, decode **14.44 tok/s with MTP vs 12.59 tok/s without (+14.7 %)**, `timings.speculative` true **[V: user's PowerShell output]**. Single runs on prose, where MTP acceptance is low (the HermiHg card measures 0.20 to 0.23 acceptance on prose, 0.7 to 0.8 on repetitive/code); `mix` (adds n-gram) and code/repetitive prompts not tried.
- Plain decode is 12.6 tok/s for 15.7 GiB of weights and 33 tok/s for the 6.7 GiB 9B pack: about 200 to 220 GiB/s of weight traffic in both, i.e. memory-bandwidth bound **[I, arithmetic from the timings]**.
- Statuses: S3-01, S3-04, S4-03, S4-04 closed. Open in the POC path: S4-05 (soak), S1-04 (memory ceiling), S2-03 (Linux regression of shared edits), S3-03 (partial: `gr_mix`/`qsa_pf` not exercised), S5 (clean-machine run, CI feasibility, pins, install doc, catalogue prune).

## 2026-10-31: soak, first part (S4-05)

- 13 sequential requests on Qwen3.8-27B 4.00bpw + MTP, up to a 7.2K-token context in an 8192 window: all completed, no TDR event, GPU dedicated memory 20.18 to 20.39 GiB **[V]**. Prompt cache works (identical prompt: 1 new token, about 20 ms). Prefill 54 to 65 tok/s on first use (4,023 and 3,183 tokens) **[V]**, which is slow next to the README's NVIDIA figures but is a cold-ish first-use number on a different GPU, not a conclusion **[?]**.
- Next: distinct-prompt requests to defeat the cache; then S1-04 sweep with the server stopped.

## 2026-10-31: GPU memory sweep (S1-04, first part)

- Idle GPU, `torch.ones` in 1 GiB chunks: OOM at **75 GiB** while the driver reports 32.70 GiB free; single allocations of up to **80 GiB** work (largest tried) **[V]**. torch's 107.87 GiB total is therefore not usable as a budget: a model is thousands of mid-sized tensors, which behaves like the chunk case **[I]**. A 90.7 GB (about 84.5 GiB) pack plus KV cache would not fit if the chunk limit is the real one; the 15.7 GiB 27B pack (20.4 GiB resident) is far below it.
- Next: repeat with 256 MiB and 4 GiB chunks and singles of 88/96/104 GiB to see whether the limit depends on allocation size; record the Task Manager dedicated/shared totals and the Adrenalin/BIOS setting.

## 2026-10-31: S5-04 install doc, S5-05 catalogue marked

- `doc/windows/install.md` written from the journal (prerequisites, install, model + MTP, request quoting, measured numbers, limits, what to send) and linked from the main README and `doc/windows/README.md`. Not validated on a clean machine (S5-01).
- Error catalogue: `OBSERVED` set (10 of 43); unobserved signatures are printed as hypotheses. `winsock-provider` text still names the old causes; revise it when convenient.

## 2026-10-31: correction: GPU memory ceiling is about 100 GiB (S1-04 done)

- Second sweep on the idle GPU: 256 MiB / 1 GiB / 4 GiB chunks reach 101.75 / 101.0 / 100.0 GiB; single block 88 GiB ok, 96 GiB FAIL **[V]**. The first sweep (OOM at 75 GiB, 32.7 GiB reported free) is superseded: probably another consumer held memory then **[?]**. My earlier conclusion that an 85 GiB pack would not fit was based on that run and is withdrawn; `install.md`, S1-04 and `deferred.md` corrected.
- Consequence: Qwen3.8-Flash-Next (84.5 GiB weights) plus KV cache could fit in about 100 GiB, leaving about 15 GiB; not tried. Decode of an 84.5 GiB dense-equivalent read would be bandwidth-bound at a few tok/s unless it is MoE (active experts only) **[I]**.
