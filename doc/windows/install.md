# Install on Windows 11 (Strix Halo, gfx1151)

Native Windows, not WSL2. One target: Windows 11 x64, AMD Ryzen AI Max (gfx1151), latest Adrenalin driver, Python 3.12. Everything lives in `.venv-win` inside the clone; nothing is installed into the system Python.

Status: written from the first successful run (2026-10-30/31) and **not yet repeated from a clean machine** ([S5-01](sprints/s5-harden/01-clean-machine-run.md)). Claims marked **[V]** were measured on the target machine; the rest is expected behaviour.

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

Clone to a short path without spaces, with symlinks enabled. Use the explicit `uv run --no-project python ...` form: `uv run --no-project scripts\windows\kyojin_win.py` failed with "os error 3" on the target **[V]**, and plain `uv run` inside the repo would install the project (CPU/CUDA torch).

```powershell
git clone -c core.symlinks=true https://github.com/plprofetes/kyojin C:\k
cd C:\k
git checkout claude/modest-rubin-p69b5g   # until the Windows branch is merged
uv run --no-project python scripts/windows/kyojin_win.py doctor
uv run --no-project python scripts/windows/kyojin_win.py setup
uv run --no-project python scripts/windows/kyojin_win.py build
uv run --no-project python scripts/windows/kyojin_win.py check
```

- `doctor` checks the machine and prints the fix for anything missing. No FAIL is the bar.
- `setup` creates `.venv-win` (exact uv-managed CPython 3.12.16), installs ROCm torch and the SDK from AMD's nightly index, `triton-windows`, and the requirements, all pinned (`scripts/windows/constraints-win.txt`).
- `build` compiles the CUDA/HIP extension with MSVC (host) and hipcc (device). About 12 minutes for a full compile on the target (4 parallel jobs, 31.6 GB RAM visible to Windows) **[V]**; reruns reuse the objects. Add `--clean` after changing headers: Windows+HIP builds do not track header dependencies.
- `check` must print `PASS=4` (rocm_sdk, torch_gpu, triton, extension) **[V]**.

## Get a model and serve it

Use a pack that includes the MTP head if you want speculative decoding; turboderp's `Qwen3.5-9B-exl3` has none. Branch names use two zeros (`4.00bpw`).

```powershell
.venv-win\Scripts\hf.exe download turboderp/Qwen3.8-27B-exl3 --revision 4.00bpw --local-dir C:\models\qwen38-27b-4bpw
uv run --no-project python scripts/windows/kyojin_win.py serve --model C:\models\qwen38-27b-4bpw -- --draft-policy mtp
```

`serve` runs `tools/qwen/serve.py` with `-c 8192 --no-vision --draft-policy off`; anything after `--` is passed through and overrides those defaults. The first start takes minutes (Triton, HIP kernels and the dense-GEMM tuning are built and cached; the second start took 9 s on the 9B pack **[V]**). Wait for `qserve: READY`.

Ask it something. PowerShell strips quotes inside `-d '...'`, so put the JSON in a file, quote `@file`, and use the served model id (`/v1/models`):

```powershell
curl.exe -s localhost:8000/v1/models
'{"model":"Qwen3.8-Flash-Yamz","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' | Set-Content -Encoding ascii req.json
curl.exe -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "@req.json"
```

Add `"stream":true` and `curl.exe -N` for streaming. `"speculative":false` turns the MTP drafter off for one request.

## What it measured

On the target (single runs, not benchmarks): Qwen3.5-9B 4.00bpw decodes 33 tok/s; Qwen3.8-27B 4.00bpw decodes 12.6 tok/s plain and 14.4 tok/s with MTP on prose, with identical text at temperature 0 **[V]**.

## Limits to know

- GPU memory: torch reports 107.87 GiB; on an idle GPU, allocations in 256 MiB, 1 GiB and 4 GiB chunks ran out at about 100 to 101.75 GiB, and a single 88 GiB block worked while 96 GiB failed **[V]**. Plan for about 100 GiB in total, less if another program holds GPU memory (a first sweep stopped at 75 GiB, probably for that reason **[?]**). A pack of about 85 GiB plus cache is therefore not ruled out.
- Windows sees about 31.6 GB of the 128 GB RAM on the target **[V]**.
- Linux-only server features degrade quietly (swap read-back, page-cache control).
- The HIP kernels gdn_hf and ple are built and match the fallback path bitwise on one prompt **[V]**; others (gr_mix, qsa_pf) were not exercised.

## Moving the pins

The working set is pinned in two places: the constants at the top of `scripts/windows/kyojin_win.py` (torch, ROCm, triton-windows, the AMD index) and `scripts/windows/constraints-win.txt` (every other package, exact `==`, no pre-releases). To move them:

1. Change the constants (or set the `KYOJIN_*` environment overrides to try first) and edit the constraints file.
2. Prove the constraints resolve together for Windows and Python 3.12: the `uv pip compile ...` command in the header of `constraints-win.txt`. A pin that no longer resolves fails here, before a 9 GB download.
3. `setup --recreate`, then `build --clean`, `check`, and one served completion.
4. Compare the installed set with the file: every constraints line must appear in `uv pip freeze --python .venv-win\Scripts\python.exe`.
5. Update the version table in [assessment.md](assessment.md) section 7d and add a [journal.md](journal.md) entry.

If a pinned nightly wheel disappears from AMD's index, `doctor` (amd_index) says so; the stable-index fallback is in [deferred.md](deferred.md).

## When something fails

Every `kyojin_win.py` command writes `logs\windows\<timestamp>-<command>\`. Send that whole folder (`FAILURE.md` first, then `run.log` and the numbered step logs). Secrets in the environment snapshot are redacted. Do not paste screenshots of the console only: the folder has the full output.

More: [journal.md](journal.md) (what happened, in order), [deferred.md](deferred.md) (what is not done), [assessment.md](assessment.md) (background and sources).
