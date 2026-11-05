# Switch sweeps (per model)

What each server switch and build flag is worth on a real model, measured on two standard queries. One report per model and
draft policy lives in this folder: `<model>-<policy>.md` (for example `qwen38-27b-4bpw-off.md`).

The inventory of every switch the code reads (default, read site, server value, nearest comment) is
[../../knobs.md](../../knobs.md). The hand-written explanations are in [../../env_vars.md](../../env_vars.md).

## Method (`tools/sweep.py`)

- **Fresh server per measurement:** `tools/qwen/serve.py --no-vision --draft-policy off`, with the arm's environment on top of the server's own `SERVE_ENV`.
- **Warm-up, then measure:** one warm-up request per prompt size with other text and 8 new tokens, because the first use of a size tunes kernels. Then the measured requests: a ~4K and a ~16K token prompt of fixed real text (`exllamav3/conversion/standard_cal_data`), greedy, 256 new tokens.
- **Metrics:** prefill and decode speed come from the server's own `qserve: request` line. The reply text is kept, so a switch that changes the output shows as `same: no`.
- **Order and noise:** arms run in a shuffled order per round (2 rounds by default). The baseline (no override) runs first, in the middle and last of every round, so the noise band in the report is measured.
- **Default arms:**
  - every `SERVE_ENV` entry reverted to the engine default written at its read site
  - a few decode-GEMV switches (`EXL3_DEC_*`)
  - five build flags (`EXL3_HIP_DEFINES`): one `kyojin_win.py build --clean` each, about 12 minutes. The default build is restored afterwards, even after an interruption; the next start restores it if needed.
- **Excluded by default:**
  - Arms that only act on speculative verification are skipped with `--draft-policy off`. Run again with `--draft-policy mtp` for those.
  - `--skip REGEX` drops more, e.g. `MOE` for a dense model.
- **Results and resume:** results are written as they come in (`results.jsonl`). Running the same command again skips the runs that succeeded and retries the failed ones. After 3 failed runs in a row the sweep stops, which usually means a driver reset: check Event Viewer, Display 4101.

## Running it overnight (Windows target)

Before starting:
- Pull the branch.
- Make sure the installed extension is the default build (last built with `EXL3_HIP_DEFINES` unset).
- Turn off sleep on AC: `powercfg /change standby-timeout-ac 0`.

From the repo root:

```
.venv-win\Scripts\python.exe tools\sweep.py run --model C:\models\qwen38-27b-4bpw --skip MOE --dry-run
.venv-win\Scripts\python.exe tools\sweep.py run --model C:\models\qwen38-27b-4bpw --skip MOE
```

- **Duration:** about 3 minutes per measurement on the 27B, roughly 2 h per round of env arms plus about 1.5 h for the build arms. `--max-hours` (default 10) stops starting new runs.
- **Output:** goes to `logs\sweep\<date>-<model>-off\`: `report.md`, `results.jsonl`, `server\*.log`, `builds\*.log`, `sweep.log`.
- **Publishing:** copy the report here:

```
.venv-win\Scripts\python.exe tools\sweep.py report logs\sweep\<dir> --md doc\benchmarks\sweeps\qwen38-27b-4bpw-off.md
```

- **MoE model:** the same command with `--model C:\exl3-probe\models\Qwen3-30B-A3B-exl3-4.0bpw` and without `--skip MOE`.
- **Custom arms:** `--arms FILE.json`, e.g. `[{"name": "x", "env": {"EXL3_DEC_KBS": "6"}}, {"name": "y", "build": "EXL3_DEC_PF=1"}]`.

## Reading a report

- Cells are the % change of the median against the baseline median; positive means faster.
- **Bold** means the change is larger than the baseline's own half-range and larger than 1 %. Anything else is noise.
- `same: no` means the switch changes the greedy reply. That is expected for numerics switches (k-split, fused paths) and a red flag for switches documented as bit-identical.
- The `EXL3_DEC_KBS` / `EXL3_DEC_MIN_BLOCKS` arms are global (every dense decode GEMV), so they show a direction, not a per-shape optimum. See [S6-08's plan](../../windows/sprints/s6-scale-and-speed/plan-08-decode-gemv.md).
