# End-to-end smoke: load the pack, prefill, decode 16 tokens

- Status: todo
- Depends: S2-02, S1-05
- Risk: HIGHEST in this sprint: integrates everything and will surface failures in code that has never run on Windows.

Sprint: [Sprint 3: runtime kernels run](README.md) · ID S3-01 · priority order 1 in the sprint

**Goal.** Reach the first generated token and enumerate what breaks.

**Steps.**
1. Ask for consent before adding a script; the simplest route is the user running the existing `tools/qwen/serve.py` or a short snippet. If approved, write `scripts/windows/smoke_model.py`: `Config.from_directory`, `Model`, `Cache`, `Generator`/`Job`, greedy 16 tokens; add `kyojin_win.py smoke --model`.
2. Run with `-v`; for each distinct failure, open or extend S3-02 / S3-03 / S3-04 or add a new task file here.
3. Keep a table in the journal: failure, root cause, fix, file.

**Done when.**
- 16 tokens generated from a fixed prompt; output saved in the run folder.

**If it fails.** If the failure is in an area outside S3-02..04 (for example the loader reading safetensors, `stloader.cpp` file IO), add a task rather than patching ad hoc.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [exllamav3 package](../../../../exllamav3) · [qwen3_5 architecture](../../../../exllamav3/architecture/qwen3_5.py) · next: [S3-02](02-hip-lib-windows.md)

Depends on: [S2-02](../s2-extension-builds/02-import-and-dll-resolution.md), [S1-05](../s1-risk-spikes/05-model-pack-on-disk.md)
