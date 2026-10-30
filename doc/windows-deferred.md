# Windows port: deferred items and sprint plan

Companion to [windows-port-assessment.md](windows-port-assessment.md). Target: Windows 11 x64, Strix Halo (gfx1151), Python 3.12, ROCm/torch from AMD's nightly index, `uv`-managed venv `.venv-win`. Goal: **serve a Qwen model through `tools/qwen/serve.py` on Windows.**

Rules for this file: an item moves out of "Deferred" only when a sprint needs it. Every deferral says why and what would trigger it. Update this file in the same commit that changes the status.

Legend: **[V]** verified, **[I]** inferred, **[?]** unknown (see the assessment).

## 1. Deferred (not doing now)

### 1.1 Model and feature scope

| Item | Why deferred | Revisit when |
|---|---|---|
| GLM, MiMo, DeepSeek-V4.1 servers (`tools/glm`, `tools/mimo`, `tools/ds41*`) | POC is Qwen only. GLM/MiMo add MLA/DSA kernels not exercised by Qwen. | Sprint 6, after Qwen serves |
| DS41 SSD expert tier (`modules/expert_store.py` `O_DIRECT`, `engram_io.py` `os.preadv`, `dsv41/*`) | Needs `FILE_FLAG_NO_BUFFERING` / overlapped IO rewrite; experimental even on Linux. | Only if someone wants 100 GB+ models on Windows |
| Qwen3.8-Flash-Next (95 GB) and 27B-class models | Blocked on the Windows GPU-memory ceiling question **[?]**; a small model keeps RAM free for compiling. | Sprint 6 |
| Vision tower (`--no-vision`) | Fewer kernels, less memory. | After text serving is stable |
| Speculative decoding / MTP drafter (`--draft-policy off`) | Adds the drafter model and extra kernels; lossless, so it is purely a speed feature. | Sprint 6 |
| Uncensor presets (`presets/`) | Unrelated to the port. | Never needed for the port |
| Multi-GPU tensor parallel (`model/model_tp*.py`, NCCL/all-reduce `parallel/` sources) | Single APU; `parallel/` is already excluded from the ROCm build (`build_config.py`). | Not planned |

### 1.2 Linux-specific runtime code

| Item | Where | Why deferred | Revisit when |
|---|---|---|---|
| `libamdhip64.so` lookup and `/proc/self/maps` | `util/hip_lib.py` | Needed only once the extension imports; fix against the real DLL name found by the first `doctor` run. | **Sprint 3** (first to be pulled in) |
| `hipcc` resolver (`hipcc` vs `hipcc.exe`, `/opt/rocm`, `--gcc-install-dir`) | `util/hip_compiler.py`, 14 `*_hip.py` JIT modules | Same. `--genco` `.hsaco` output is portable. | Sprint 3 |
| `/proc/meminfo`, `smaps`, `pagemap`, `MADV_POPULATE_READ` | `util/memory.py`, `util/memguard.py`, `serve.py` startup | Memory accounting for tiering/swap-in; not needed for a small model. Must degrade quietly, not crash. | Sprint 4 (guard) / Sprint 6 (real Windows equivalents) |
| `posix_fadvise`, `drop_caches` page-cache control | `util/pagecache.py`, `util/memguard.py` | No Windows equivalent. | Not planned |
| `/sys/class/drm/...mem_info_gtt_used`, `/proc/self/statm` metrics | `tools/glm/serve.py:1233,1237` (check `tools/qwen/serve.py`) | Cosmetic. | Sprint 4 (wrap in try) |
| CPU MoE host path Windows branch (affinity, timers) | `cpu/moe_mul1.cpp` (authors mark it UNVERIFIED) | Only used for CPU-offloaded experts; small dense/MoE-on-GPU test avoids it. | When a MoE model is offloaded |
| `prctl(PR_SET_PDEATHSIG)` | `util/misc.py` | Already platform-guarded. | Not needed |

### 1.3 Tooling, packaging, CI

| Item | Why deferred | Revisit when |
|---|---|---|
| CI job for Windows (`windows_rocm_smoke`) | No point before the local script is green. Runner disk (about 14 GB free on C:) vs 12 GB SDK is a known problem **[?]**. | Sprint 5 |
| Prebuilt wheels / PyPI / releases | POC is source build. | After Sprint 5 |
| `uv.lock` / pinned Windows requirements | First run produces `pip-freeze.txt`; pin from a known-good one. | Sprint 5 |
| sccache / compile cache | Optimisation. | If rebuild time hurts (it will) |
| Stable AMD wheel fallback (`repo.amd.com/rocm/whl/gfx1151/`) | Kyojin is only verified on nightlies; fallback if the pinned nightly disappears. | If `doctor` amd_index fails |
| Index host equivalence (`nightly.repo.amd.com/rocm/whl-next/` vs `rocm.nightlies.amd.com/whl-multi-arch/`) | One constant (`KYOJIN_ROCM_INDEX`); verify when convenient **[?]**. | Sprint 1 |
| Docker / nix / `container.yml` | Linux only. | Not planned |
| Auto-installing Visual Studio / drivers from the script | `doctor` prints exact commands instead; installers need admin and reboot. | After the manual path is stable |
| Python 3.13/3.14, Windows 10, ARM64, other GPUs (gfx110x/gfx120x) | One target only. | Not planned |
| `tools/strix_halo/*.sh` benchmarks, `ck_pagedkv` / `libckpf.so` CK paths | Linux shell scripts; CK falls back to Triton silently. | After serving works, if perf work starts |
| Linux system tuning (IOMMU, DPM level, fan curve) | Linux sysfs. Windows equivalents (Adrenalin tuning, power plan) are untested. | Sprint 6 perf work |
| Perf parity vs Linux, benchmarks | Correctness first. | Sprint 6 |

## 2. Error catalogue (how "helpful errors" evolves)

Every failing step prints and saves (`FAILURE.md` in the run's log folder): command, exit code (Windows crash codes named), the matching **known signature** (cause + fix), the **first** error lines with line numbers into the step log, and the last output. Signatures live in `KNOWN` in `scripts/windows/kyojin_win.py` and are **hypotheses** until observed.

Process after each real failure:
1. Open `FAILURE.md`. If "no known signature matched": add a `Known(...)` entry (regex from the real first-error line, cause, concrete fix) and a sample line in `test_kyojin_win.py::HelpfulErrors.SAMPLES`.
2. If a signature matched but the advice was wrong, fix the advice, not just the code.
3. Record the failure and fix in the log table in section 4.

## 3. Sprint plan: serve a Qwen model on Windows

Effort is dominated by rebuild time (10+ minutes per iteration on Linux, expect more) and by how many unknowns a sprint hits; no calendar estimates. Each sprint ends with evidence: the log folders from the passing run, listed in section 4.

**Sprint 0 (done): assess and tooling.** Assessment, `kyojin_win.py` (`doctor/setup/build/check/serve`), `setup.py` Windows+HIP flags, 25 unit tests on Linux. Evidence: commits on `claude/modest-rubin-p69b5g`.

**Sprint 1: toolchain green.** `doctor` and `setup` pass on the target machine.
- Exit: `doctor` has no FAIL; `torch_gpu` shows `gfx1151` and the visible GPU memory; `triton` vector-add passes; `rocm_sdk` shows `hipcc.exe` and all headers.
- Work: fix whatever the first runs expose (index URL, VS discovery, `rocm-sdk init`, driver). Record the GPU-addressable memory figure (answers the Windows memory-ceiling question **[?]**).
- Gate: if Triton fails on gfx1151, stop and decide between a triton-windows fix, a different pin (`<3.10`), or dropping to torch fallbacks. Do not proceed to Sprint 2 without a decision.

**Sprint 2: extension builds and imports.** `build` produces `exllamav3_ext*.pyd`.
- Exit: `check` shows `extension` PASS (imports; symbols present).
- Work: setuptools/torch/hipcc issues (flags, link libs, include paths, MSVC interplay). Expect `setup.py` and possibly `build_config.py` edits; keep each guarded for Windows so Linux builds are unchanged. Verify `moe_mul1.cpp` compiles under `cl.exe`.
- Gate: Linux build must still pass after every `setup.py` edit (run on a Linux box or CI before merging).

**Sprint 3: runtime kernels run.** A tiny forward pass produces correct numbers.
- Exit: `tests/test_hip_gemv_decode.py` (with `EXL3_TEST_MODEL` set to a small pack) passes, or an equivalent minimal script: model loads, one prefill, one decode step, logits match the Linux reference within tolerance.
- Work: `util/hip_lib.py` and `util/hip_compiler.py` Windows support; JIT `.hsaco` kernels (`hipModuleLoad` via the Windows DLL); Triton FLA/GDN kernels on triton-windows; `~/.cache` paths.
- Gate: numeric mismatch vs Linux means stop and bisect per kernel before continuing.

**Sprint 4: serve a small Qwen.** `tools/qwen/serve.py` answers `/v1/chat/completions`.
- Exit: `kyojin_win.py serve --model <small pack>` prints `READY`, `curl` completion returns coherent text, greedy output matches the Linux run for a fixed prompt (or documented tolerance), 10 sequential requests without crash.
- Work: guard Linux-only metrics/memory code in the server path (`util/memory.py`, startup swap-in, `/sys` reads); pack requirements (`chat_template.jinja`, `generation_config.json`); first-launch kernel tuning time.
- Model: `turboderp/Qwen3.5-9B-exl3` at about 4 bpw is the first candidate **[?]** (repo not yet confirmed from a network that reaches Hugging Face).

**Sprint 5: harden and publish the path.**
- Exit: a second clean machine (or fresh user profile) goes from nothing to serving with the documented commands; `pip-freeze.txt` pinned into a Windows requirements/lock file; `docs/install-windows.md` written; CI `windows_rocm_smoke` compiles on a hosted runner (or documented why not).
- Work: pin versions, trim the error catalogue to real signatures, add the CI job, decide the stable-wheel fallback.

**Sprint 6: scale and speed (separate decision).**
- Candidates: 27B-class dense, Qwen3.8-Flash-Next (needs the memory ceiling answer and real Windows equivalents for the memory-accounting code), speculative decoding, perf comparison with Linux, DS41 only on demand.

## 4. Evidence log (fill as sprints run)

| Date | Sprint | Command | Result | Log folder | Notes / fix |
|---|---|---|---|---|---|
| | 1 | `doctor` | | | |
