# Get a small Qwen EXL3 pack on disk (user, parallel)

- Status: todo
- Depends: -
- Risk: LOW difficulty; ordered last because it can happen any time. The candidate repo name is unconfirmed (Hugging Face was unreachable from the assistant's sandbox).

Sprint: [Sprint 1: risk spikes (fail fast)](README.md) · ID S1-05 · priority order 5 in the sprint

**Goal.** Have a small EXL3 pack plus the files `tools/qwen/serve.py` needs.

**Steps.**
1. Confirm `turboderp/Qwen3.5-9B-exl3` exists and pick the ~4.0 bpw branch (turboderp repos use one branch per bitrate); `hf download turboderp/Qwen3.5-9B-exl3 --revision 4.0bpw --local-dir C:\models\qwen35-9b-4bpw` (branch name to confirm).
2. Ensure `chat_template.jinja` and `generation_config.json` are in the folder; copy them from the base `Qwen/Qwen3.5-9B` repo if absent.
3. `kyojin_win.py serve --model <dir>` prints the `model_dir` check; it should PASS.
4. Record the exact repo, branch, size and file list in the journal.

**Done when.**
- `model_dir` check PASS; path and file list in the journal.

**If it fails.** If the repo or branch does not exist, pick another small Qwen EXL3 (search `exl3` on Hugging Face) and update [deferred.md](../../deferred.md) and the assessment §6.3.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [serve command](../../../../tools/qwen/SERVE.md) · [assessment §6.3](../../assessment.md) · next sprint: [Sprint 2](../s2-extension-builds/README.md)
