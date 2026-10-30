# serve.py starts and reaches READY

- Status: todo
- Depends: S3-01
- Risk: MEDIUM-HIGH: the server pulls in memory management, metrics and tier code written for Linux.

Sprint: [Sprint 4: serve a small Qwen over HTTP](README.md) · ID S4-01 · priority order 1 in the sprint

**Goal.** READY line printed.

**Steps.**
0. Saved patches: [startup_health](../../review/patches/tools_startup_health_py.patch) (`register_at_fork`, port reuse) and [serve.py UTF-8](../../review/patches/tools_qwen_serve_py.patch). `tools/qwen` symlinks must be real files (`doctor` symlinks check).
1. `kyojin_win.py -v serve --model <dir>`; fix start-up failures (Linux-only `/proc`, `/sys`, `os.sysconf`, `~/cache` paths, `swap_in_process`).
2. Guard, do not delete, Linux behaviour; each guard is `sys.platform`/`os.name` based.
3. Check the path of `--slot-save-path` and other defaults under `~`.

**Done when.**
- READY printed; `/health` and `/v1/models` respond.

**If it fails.** Anything in the server path that needs real Windows work (not a guard) becomes a task.

**Evidence.** (run folders, versions, numbers; fill in)

- 

**Links.** [qwen serve.py](../../../../tools/qwen/serve.py) · [SERVE.md](../../../../tools/qwen/SERVE.md) · [memory util](../../../../exllamav3/util/memory.py) · next: [S4-02](02-first-launch-warmup.md)

Depends on: [S3-01](../s3-runtime-kernels/01-end-to-end-smoke.md)

- Known risk from the Windows data (S1-04): Kyojin sizes memory with `torch.cuda.mem_get_info()` (11 files incl. `tools/qwen/serve.py:2265` free-VRAM guard `MIN_FREE_MIB`, `exllamav3/model/model.py`, `model_init.py`, `util/memory.py`, `loader/safetensors.py`). On Windows that call over-reports free VRAM while another process holds it (107.55 GiB reported, 1 GiB allocation failed), so auto-sizing and the free-VRAM guard can mislead; expect OOMs at load/first prefill if the GPU is shared. Run with the GPU otherwise idle for the POC and size `-c`/model explicitly.
