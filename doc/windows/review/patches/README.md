# Saved patches (from the independent review, round 2)

Cut against `cca1e17`/`c4a9f87`; `git apply --check` passed on `c4a9f87`. None has been compiled or run on Windows. Why Linux is unchanged is argued in each section below (reviewer's text, abridged to status).

| Patch | Status | Task |
|---|---|---|
| `setup.py` (hipblas+rocblas libraries) | **Applied** (libraries only; the reviewer's CXX/clang-cl block was not applied: compiler choice lives in the environment set by `kyojin_win.py`, keeping the shared file smaller) | S2-01 |
| `exllamav3/ext.py` (clear ImportError instead of a doomed JIT build on Windows+ROCm) | **Applied** | S2-01 |
| `hgemm.cu` (`_mkdir` under `_WIN32`) | **Applied** | S2-01 |
| `kyojin_win` `sdk_env` (DISTUTILS_USE_SDK, CC/CXX clang-cl, HIP_CLANG_PATH) | **Applied** | S2-01 |
| `hip_lib_windows.patch` | Saved, not applied (needed when the first JIT HIP kernel runs) | [S3-02](../../sprints/s3-runtime-kernels/02-hip-lib-windows.md) |
| `tools_startup_health_py.patch` (`register_at_fork`, `allow_reuse_address`) | Saved, not applied | [S4-01](../../sprints/s4-serve-qwen/01-serve-reaches-ready.md) |
| `tools_qwen_serve_py.patch` (UTF-8 `read_text`) | Saved, not applied | S4-01 |

Apply a saved patch with `git apply doc/windows/review/patches/<name>.patch` from the repo root (re-check with `git apply --check` first; line numbers may have moved).

Not patched (reviewer's notes): the 4 `tools/qwen` symlinks (clone with `-c core.symlinks=true`, `doctor` now checks); `util/hip_compiler.py` needs nothing if the SDK `bin` is on PATH or `EXL3_HIPCC` is set; Python `HOME` fallbacks resolve to `\tmp` with HOME unset, so `kyojin_win.py` now sets `HOME=%USERPROFILE%`.
