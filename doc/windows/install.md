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

`serve` runs `tools/qwen/serve.py` with `-c 8192 --no-vision --draft-policy off`; anything after `--` is passed through and overrides those defaults. The first start takes minutes (Triton, HIP kernels and the dense-GEMM tuning are built and cached; the second start took 9 s on the 9B pack **[V]**). Wait for `qserve: READY`; until then the port already answers, but chat requests get `503 {"error": {"message": "Loading model"}}` and `/health` reports loading **[V]**.

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
2. Prove the constraints resolve for BOTH install steps (Windows, Python 3.12): the two `uv pip compile ...` commands in the header of `constraints-win.txt` (step 1 resolves torch's dependencies from AMD's index only, step 2 the rest from PyPI). A pin that no longer resolves fails here, before a 9 GB download.
3. `setup --recreate`, then `build --clean`, `check`, and one served completion.
4. Compare the installed set with the file: every constraints line must appear in `uv pip freeze --python .venv-win\Scripts\python.exe`.
5. Update the version table in [assessment.md](assessment.md) section 7d and add a [journal.md](journal.md) entry.

If a pinned nightly wheel disappears from AMD's index, `doctor` (amd_index) says so; the stable-index fallback is in [deferred.md](deferred.md).

## Text encoding on the target (Polish locale)

The target runs with the ANSI code page **1250** and the OEM code page **852** (culture `pl-PL`); no UTF-8 override is set **[V, registry and `python -c "import locale"`]**. It is a property of the machine, not of one session, so every new shell, script and agent session meets it. What goes wrong, and the rule for each:

| Symptom | Cause | Rule |
|---|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode character '✅'` (or an arrow, `≥`, an emoji) | Python writes redirected or piped stdout/stderr (`> file`, `\| tee`, a tool's capture) in cp1250. Model answers and our own reports contain characters outside it. Measured: redirected `sys.stdout.encoding` is `cp1250` **[V]**. | Run with UTF-8 mode: `python -X utf8 ...` or `PYTHONUTF8=1`. For files you write, pass `encoding="utf-8"`. |
| The fix "did nothing" with `PYTHONUTF8=1` or `PYTHONIOENCODING=utf-8` | `python -I` (isolated mode) **ignores every `PYTHON*` variable** **[V: `-I` printed utf8_mode 0 and cp1250 with the variable set]**. | With `-I` use the flag: `python -X utf8 -I script.py` gives utf8_mode 1 and utf-8 **[V]**. Or drop `-I`. |
| `read_text()` / `open()` mis-decodes or raises on a UTF-8 file | Without `encoding=`, Python uses cp1250 | `PYTHONUTF8=1` covers every call without touching code; `kyojin_win.py serve` and `build`, `scripts/windows/serve-lan.ps1` and `tools/sweep.py` already set it |
| `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbe` from a `_readerthread` above the `bench.py` table | In UTF-8 mode, `bench.py` decodes its helper commands (`subprocess.run(..., text=True)`) as UTF-8, but a Windows helper prints cp852 or cp1250 bytes **[I: the exact helper is not identified]**. Harmless: the table is still correct. | Fixed on `pr/windows-core` (commit `5451806`: `encoding="utf-8", errors="replace"`), not on this branch. Any new `subprocess` call: give `encoding` and `errors="replace"` explicitly |
| `chcp` shows 65001 in one terminal and 852 in another | The console code page is per console and only affects what the console displays; it does not change what Python uses for files and pipes | Do not rely on `chcp`; set the Python side as above |

For a script that must be safe either way, start it with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` instead of relying on the caller.

Not changed by this document (the operator's call, each has side effects): the Windows setting "Beta: Use Unicode UTF-8 for worldwide language support" (makes the ANSI page 65001 for the whole machine and can break old programs that expect cp1250), or a user-level `PYTHONUTF8=1` environment variable (fixes Python only, applies to every Python program including ones that decode with a legacy page on purpose).

## When something fails

Every `kyojin_win.py` command writes `logs\windows\<timestamp>-<command>\`. Send that whole folder (`FAILURE.md` first, then `run.log` and the numbered step logs). Secrets in the environment snapshot are redacted. Do not paste screenshots of the console only: the folder has the full output.

More: [journal.md](journal.md) (what happened, in order), [deferred.md](deferred.md) (what is not done), [assessment.md](assessment.md) (background and sources).
