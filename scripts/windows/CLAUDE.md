# Windows port: working conventions

Scope: everything under `scripts/windows/` and `doc/windows/`, plus the small Windows-gated edits the port needs elsewhere (`setup.py`, later `exllamav3/util/hip_*.py`). Start at [doc/windows/README.md](../../doc/windows/README.md), then run `python scripts/windows/tasks.py next`.

## Assumptions (one target, do not widen)

- Windows 11 x64, AMD Strix Halo (gfx1151), latest Adrenalin driver, 128 GB machine, Python 3.12.
- Everything lives in a `uv`-managed venv `.venv-win` in the repo; nothing is installed into the system Python or PATH.
- ROCm torch, `rocm-sdk` and `triton-windows` come from AMD's nightly index and PyPI, pinned in one place (`kyojin_win.py` constants, env overrides `KYOJIN_*`).
- Build chain: MSVC (`cl.exe`) for host C++, `hipcc.exe` for device code, driven by setuptools + torch `BuildExtension` + ninja. No CMake, no new build system.
- POC goal: serve a small Qwen EXL3 pack with `tools/qwen/serve.py`. Linux is the product; Windows changes must not alter Linux behaviour.
- The assistant's sandbox is Linux with no GPU and no Windows. Nothing here is verified on Windows until a run folder from the real machine says so.

## Simplest solution first (read before writing anything)

- Prefer the smallest thing that works: prose over tooling, an existing tool over a new script, an edit to an existing file over a new file, a few lines over a framework. `tasks.py` is already more machinery than the job needs; it stays, but it is not a precedent.
- **New scripts, tools, subcommands, test harnesses or generators need the user's explicit consent before they are created.** Propose in one or two sentences (what it does, why prose or an existing tool will not do) and wait for a yes. This includes helpers, spike programs, one-off generators and new `kyojin_win.py` subcommands. Editing an existing script to fix a bug or extend an existing step is fine.
- Not covered by the consent rule: documentation, task files, journal entries, and changes to existing files that the current task requires.
- When in doubt, do less and ask. A shorter answer or a manual step the user performs is better than an automation nobody asked for.

## Expectations

- **Simple, happy path first.** Make the golden path work before handling variations. No retries, fallbacks, version matrices, installers or abstractions "for later". If something is not needed for the next task, add it to [deferred.md](../../doc/windows/deferred.md) instead of building it.
- **Fail fast.** Do the hardest, least certain thing first (that is how sprint tasks are ordered). Stop and report at a gate rather than working around it.
- **Reuse, do not reinvent.** Prefer documented commands from AMD (TheRock), triton-windows, uv, Microsoft tools. Do not solve solved problems.
- **Helpful errors are a feature.** Every failure must say what failed, the likely cause, and the exact next command. Verbose logs are always written; the console is quiet unless `-v`.
- **Gate Windows code.** Edits to shared files sit behind `sys.platform` / `os.name` / `windows` conditions. Keep each edit minimal and review the diff against Linux.

## Wording

- The reader is an engineer. Be terse and factual; no hype, no filler, no cheerleading.
- Mark claims: **[V]** verified (source or run), **[I]** inferred, **[?]** unknown. Never present an inference as a verified fact.
- Messages to the user: cause, then the exact command to run. Hints contain real commands and paths, not advice.
- Report outcomes as they are: failing tests, skipped steps and unverified code are stated plainly.

## Sources and licences (mandatory)

- Cite every source you read or reuse: file path (and line) for code, URL for docs. Put licence findings in [assessment.md](../../doc/windows/assessment.md) section 7c.
- Do not copy third-party code. If a reuse is ever necessary, record the licence, the origin and the copied lines in 7c in the same change, and keep the notice the licence requires. Unlicensed material (for example the community gist) is ideas only.
- Extracted third-party files stay outside the repo (scratch directories).

## Conventions established so far

- **Run folders:** each `kyojin_win.py` command writes `logs/windows/<timestamp>-<command>/` (gitignored): `run.log` (DEBUG), `NN-<step>.log` per subprocess, `env.json`/`build-env.json` (secrets redacted), `report.json`, `FAILURE.md` (when something failed), `pip-freeze.txt` (after setup). Ask for that folder when debugging.
- **Checks** return `Check(name, status, summary, details, hint)` with status `PASS/WARN/FAIL/SKIP/INFO`; probes never raise; probes that can crash natively run in a subprocess.
- **Error catalogue:** `KNOWN` signatures in `kyojin_win.py` are hypotheses. A new real failure gets a signature plus a sample line in `test_kyojin_win.py` (`HelpfulErrors.SAMPLES`; the test enforces one sample per signature).
- **Tasks** (kept in prose; the tracker script is optional): one file per task under `doc/windows/sprints/s<N>-<slug>/<NN>-<slug>.md`; `NN` is the priority order (hardest first); header lines `- Status:`, `- Depends:`, `- Risk:`. Editing the `- Status:` line by hand is fine; `tasks.py sync` only refreshes the tables and may be skipped. Every task has Goal, Steps, Done when, If it fails, Evidence, Links; documents cross-link with relative links (`tasks.py check` verifies them).
- **Journal:** append one entry per session to [journal.md](../../doc/windows/journal.md); never rewrite old entries.
- **Reproducibility:** pin every dependency to an exact version or commit SHA, never a floating tag or `latest`, and report what was used. Every run records the repo commit, the number of uncommitted paths and the pins in `env.json` and `pip-freeze.txt`. When a doc or journal entry cites a source (repo, wheel index, tool), give the version, tag or SHA that was read, not just the URL. The current set is in [assessment.md](../../doc/windows/assessment.md) section 7d; update it whenever a pin moves.
- **Pins and URLs:** constants at the top of `kyojin_win.py`; the AMD index host is a single constant.
- **Branch/commits:** work on the session branch given in the task (currently `claude/modest-rubin-p69b5g`), commit with the required trailer lines, push there. No pull request unless asked.

## File locations

| What | Where |
|---|---|
| Orchestrator (`doctor/setup/build/check/serve`) | `scripts/windows/kyojin_win.py` |
| Orchestrator tests (stdlib unittest, no GPU) | `scripts/windows/test_kyojin_win.py`, `scripts/windows/test_tasks.py` |
| Task tracker | `scripts/windows/tasks.py` |
| Docs index, assessment, deferred items, journal | `doc/windows/README.md`, `assessment.md`, `deferred.md`, `journal.md` |
| Sprints and tasks | `doc/windows/sprints/` |
| Windows-gated build flags | `setup.py` (repo root) |
| Run logs, venv | `logs/windows/`, `.venv-win/` (both gitignored) |
| Linux reference for the build environment | `tools/strix_halo/env.sh`, `rebuild.sh`, `build.sh` |

## Commands

```bash
python scripts/windows/tasks.py next     # what to do now
python scripts/windows/tasks.py check    # docs consistent
python scripts/windows/test_kyojin_win.py && python scripts/windows/test_tasks.py
# on the Windows machine
uv run --no-project scripts/windows/kyojin_win.py -v doctor    # then setup, build, check, serve --model <dir>
```

## Definition of done for a change

Tests above pass (`tasks.py check` if you used the tracker); Linux behaviour of shared files is unchanged; the task's Evidence and the journal are updated; claims are marked V/I/?; sources and licences are recorded; the change is committed and pushed.
