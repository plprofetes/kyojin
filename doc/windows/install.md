# Install on Windows 11 (Strix Halo, gfx1151): experimental

Native Windows, not WSL2. One target: Windows 11 x64, AMD Ryzen AI Max (gfx1151), current Adrenalin driver, Python 3.12. Everything lives in `.venv-win` inside the clone; nothing is installed into the system Python. Linux remains the primary platform: [install.md](../install.md).

## Status

Verified on one machine (ASUS ProArt PX13, Ryzen AI Max+ 395, 128 GB, Adrenalin 32.0.32015.7029) with dense Qwen packs only, from a build made by the steps below. Not yet repeated from a clean machine; reports are welcome (attach the `logs\windows\...` folder, see the end).

| Works **[V]** | Not verified |
|---|---|
| Build of `exllamav3_ext` (MSVC host, clang-cl, hipcc), about 12 min | MoE packs (Qwen3.5-35B-A3B, Flash-Next, GLM, MiMo, DeepSeek) |
| Qwen3.5-9B and Qwen3.8-27B (EXL3 4.00 bpw) through `tools/qwen/serve.py`, 27B with MTP, identical text at temperature 0 | `tools/glm/serve.py`, `tools/mimo/serve.py` |
| JIT HIP kernels `gdn_hf` and `ple` (bitwise equal to the fallback on one prompt) | JIT kernels `gr_mix`, `qsa_pf` |
| A 13-request soak and a distinct-prompt soak with no TDR | Packs larger than about 85 GiB (see GPU memory below) |

Linux-only server features (swap read-back, page-cache control) degrade quietly. Single runs, not benchmarks: Qwen3.5-9B decodes 33 tok/s, Qwen3.8-27B 12.6 tok/s plain and 14.4 tok/s with MTP on prose.

## Prerequisites

| Need | How |
|---|---|
| Windows 11 x64, gfx1151 GPU, current Adrenalin driver | AMD driver page |
| Git | `winget install Git.Git` |
| uv | `winget install astral-sh.uv`, then open a new terminal |
| Visual Studio 2022 Build Tools with "Desktop development with C++" and a Windows 10/11 SDK | Visual Studio Installer; `doctor` prints the exact missing piece |
| Developer Mode (the repo has git symlinks) | Settings > System > For developers |
| Disk | at least 40 GB free (estimate; `.venv-win` holds the ROCm SDK, about 12 GB, plus torch and the build tree) |
| Free GPU | stop other GPU workloads (an LLM server holding VRAM makes every probe misleading) |

## Install

Clone to a short path without spaces, with symlinks enabled. Use the explicit `uv run --no-project python ...` form: `uv run --no-project scripts\windows\kyojin_win.py` failed with "os error 3" on the test machine, and plain `uv run` inside the repo would install the project (CPU/CUDA torch).

```powershell
git clone -c core.symlinks=true https://github.com/Yamz-Labs/kyojin C:\k
cd C:\k
uv run --no-project python scripts/windows/kyojin_win.py doctor
uv run --no-project python scripts/windows/kyojin_win.py setup
uv run --no-project python scripts/windows/kyojin_win.py build
uv run --no-project python scripts/windows/kyojin_win.py check
```

- `doctor` checks the machine and prints the fix for anything missing. No FAIL is the bar.
- `setup` creates `.venv-win` (an exact uv-managed CPython 3.12.16), installs ROCm torch and the SDK from AMD's nightly index, `triton-windows` and the requirements, all pinned (`scripts/windows/constraints-win.txt`).
- `build` compiles the extension with MSVC (host) and hipcc (device). About 12 minutes for a full compile (4 parallel jobs, 31.6 GB of RAM visible to Windows on the test machine) **[V]**; reruns reuse the objects. Add `--clean` after changing headers: Windows+HIP builds do not track header dependencies.
- `check` must print `PASS=4` (rocm_sdk, torch_gpu, triton, extension) **[V]**.

To build by hand instead of through the driver: open a VS x64 developer shell, export the variables that `sdk_env()` in `scripts/windows/kyojin_win.py` sets (ROCm paths, `PYTORCH_ROCM_ARCH=gfx1151`, `DISTUTILS_USE_SDK=1`, `CC` and `CXX` pointing at the SDK's `clang-cl.exe`, `EXL3_ROCM_SDK`), then run `python setup.py build_ext --inplace` in the repository root.

## Get a model and serve it

Use a pack that includes the MTP head if you want speculative decoding (`turboderp/Qwen3.5-9B-exl3` has none). Branch names use two zeros (`4.00bpw`).

```powershell
.venv-win\Scripts\hf.exe download turboderp/Qwen3.8-27B-exl3 --revision 4.00bpw --local-dir C:\models\qwen38-27b-4bpw
uv run --no-project python scripts/windows/kyojin_win.py serve --model C:\models\qwen38-27b-4bpw -- --draft-policy mtp
```

`serve` runs `tools/qwen/serve.py` with `-c 8192 --no-vision --draft-policy off`; anything after `--` is passed through and overrides those defaults. The first start takes minutes (Triton, HIP kernels and the dense-GEMM tuning are built and cached; the second start took 9 s on the 9B pack **[V]**). Wait for `qserve: READY`; until then the port already answers, but chat requests get `503 {"error": {"message": "Loading model"}}` and `/health` reports loading **[V]**.

Ask it something. PowerShell strips quotes inside `-d '...'`, so put the JSON in a file, quote `@file`, and use the served model id (`/v1/models`):

```powershell
curl.exe -s localhost:8000/v1/models
'{"model":"Qwen3.8-Flash-Yamz","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' | Set-Content -Encoding ascii req.json
curl.exe -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "@req.json"
```

Add `"stream":true` and `curl.exe -N` for streaming. `"speculative":false` turns the MTP drafter off for one request.

## Limits to know

- GPU memory: torch reports 107.87 GiB; on an idle GPU, allocations in 256 MiB, 1 GiB and 4 GiB chunks ran out at about 100 to 101.75 GiB, and a single 88 GiB block worked while 96 GiB failed **[V]**. Plan for about 100 GiB in total, less if another program holds GPU memory. A pack of about 85 GiB plus cache is therefore not ruled out; the 95 GB and 100 GB packs are.
- Windows sees only part of the RAM on the test machine (31.6 GB of 128 GB) **[V]**.

## Environment and caches

The driver sets what each step needs and nothing system-wide; every step logs its environment (secrets redacted) in the run folder.

| Step | Sets |
|---|---|
| `build` | `ROCM_HOME`, `ROCM_PATH`, `HIP_PATH`, `HIPCXX`, `HIP_DEVICE_LIB_PATH`, `PYTORCH_ROCM_ARCH=gfx1151`, `DISTUTILS_USE_SDK=1`, `CC`/`CXX` (the SDK's `clang-cl.exe`), `EXL3_ROCM_SDK`, `EXL3_HIP_DEFINES` (default `EXL3_HIP_STG_PAD`, as on Linux), `MAX_JOBS` (from free RAM, at most 6), `VSLANG=1033`, the VS developer environment |
| `serve` | `PYTHONPATH` (the clone), `PYTHONUTF8=1`, `EXL3_ROCM_SDK` (the runtime JIT finds `hipcc.exe` from it; `ROCM_PATH` and `HIP_PATH` are deliberately not set) |

Override the pins with `KYOJIN_ROCM_INDEX`, `KYOJIN_ROCM_VER`, `KYOJIN_TORCH_VER`, `KYOJIN_TRITON_SPEC` and `KYOJIN_PY`. Everything else is a normal `EXL3_*` switch from [env_vars.md](../env_vars.md).

Per-user caches all live under your profile folder (`%USERPROFILE%`); the cache root is `HOME` if you set it, else `USERPROFILE`:

| What | Where |
|---|---|
| Dense-GEMM tune file | `%USERPROFILE%\.cache\exllamav3\dense_gemm_tune.txt` (or `EXL3_DENSE_GEMM_TUNE_FILE`) |
| JIT HIP kernels (`*.hsaco`) | `%USERPROFILE%\.cache\exllamav3` |
| Triton | `%USERPROFILE%\.triton` (or `TRITON_CACHE_DIR`) |
| Build objects | `build\` in the clone |
| Run logs | `logs\windows\` in the clone (gitignored) |

If you built with an earlier version of this guide, your tune file may be in `<drive>:\tmp\.cache\exllamav3`; move it to the folder above to keep the tuning.

## Moving the pins

The working set is pinned in two places: the constants at the top of `scripts/windows/kyojin_win.py` (torch, ROCm, triton-windows, the AMD index) and `scripts/windows/constraints-win.txt` (every other package, exact `==`, no pre-releases). To move them:

1. Change the constants (or set the `KYOJIN_*` environment overrides to try first) and edit the constraints file.
2. Prove the constraints resolve for BOTH install steps (Windows, Python 3.12): the two `uv pip compile ...` commands in the header of `constraints-win.txt`. A pin that no longer resolves fails here, before a 9 GB download.
3. `setup --recreate`, then `build --clean`, `check`, and one served completion.
4. Compare the installed set with the file: every constraints line must appear in `uv pip freeze --python .venv-win\Scripts\python.exe`.
5. Update the pinned versions in `kyojin_win.py` and `constraints-win.txt` together, and state them in the PR.

If a pinned nightly wheel disappears from AMD's index, `doctor` (amd_index) says so; move the pin to a wheel that exists on the index and rerun the verification above.

## When something fails

Every `kyojin_win.py` command writes `logs\windows\<timestamp>-<command>\`. Send that whole folder (`FAILURE.md` first, then `run.log` and the numbered step logs). Secrets in the environment snapshot are redacted. Do not paste screenshots of the console only: the folder has the full output.

Run the driver's own tests anywhere (no GPU, no Windows): `python -m unittest scripts/windows/test_kyojin_win.py`.
