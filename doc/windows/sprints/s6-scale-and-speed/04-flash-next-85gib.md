# Run Qwen3.8-Flash-Next on Windows: smoke test, step up, n-gram placement, GPU dedicated memory

- Status: todo
- Depends: S1-04
- Risk: MEDIUM-HIGH: first run of the Flash-Next kernels (n-gram gather, PLE, grouped MoE decode with 640-wide experts, more JIT kernels) on Windows; the host RAM Windows shows is only 31.6 GB; the best trade between GPU-dedicated and host memory is unknown.

Sprint: [Sprint 6: ship it and scale](README.md) · ID S6-04 · priority order 4 in the sprint (the file name keeps the old "85 GiB" wording so links stay valid)

**Marking:** **[V]** measured, **[I]** inferred from code or docs, **[?]** unknown.

## Goal

Know, with numbers, what runs of Qwen3.8-Flash-Next on this machine and with which memory split: (1) does the model class load and generate at all on Windows, (2) how big a quant is practical, (3) n-gram table on SSD (default) against in RAM, (4) what lowering the GPU-dedicated memory buys and costs. Output: a short table in the Evidence log and a recommended setting.

## What we know before starting

- **The pack sizes are mostly the n-gram table** [V, HF file lists of `turboderp/Qwen3.8-Flash-Next-exl3`, all branches `mul1` codebook]:

  | Branch | Total | `ngram_embedding.safetensors` | Rest (weights) |
  |---|---|---|---|
  | `1.66bpw_h4_ng3` | 46.6 GiB | not split | not split |
  | `2.05bpw_h4_ng4` | 58.4 GiB | not split | not split |
  | `2.18bpw_h6_ng8` | 84.3 GiB | 48.3 GiB | 36.0 GiB |
  | `3.05bpw_h5_ng5` | 79.1 GiB | 30.4 GiB | 48.8 GiB |
  | `4.05bpw_h6_ng6` | 99.9 GiB | 36.4 GiB | 63.6 GiB |
  | `2.68`, `3.18`, `3.68`, `4.18`, `5.05`, `6.05` | 91.5 to 129.4 GiB | not split | not split |

  The earlier "does not fit" reading used the totals. By the README and `doc/env_vars.md` the n-gram table is **streamed from disk by default** and does not occupy GPU memory, so the GPU-resident part is the "rest" column **[I: not yet seen on Windows]**.
- **GPU ceiling:** torch reports 107.87 GiB; about 100 GiB can be allocated on an idle GPU [V, [S1-04](../s1-risk-spikes/04-gpu-memory-ceiling.md)]. Windows shows 31.6 GB of the 128 GB RAM, which is about 128 minus a roughly 96 GB GPU-dedicated setting **[I]**.
- **n-gram placement code** (`exllamav3/modules/ngram_embedding.py`, `doc/env_vars.md`): default is disk streaming (run-coalesced positioned reads; "overlapped `ReadFile` at high queue depth on Windows", `EXL3_NGRAM_IO_THREADS` default 128, look-ahead `EXL3_NGRAM_WARM_TOKENS` 8192 into the page cache). RAM mode is opt-in: `EXL3_NGRAM_STREAM=0` or the server flag `--ngram-ram`. RAM mode wants **table + 48 GiB margin** of available host memory; where host memory cannot be read (`avail` is `None`, which is the case on Windows until [S6-05](05-windows-host-memory.md)) no warning is printed, so the margin has to be checked by hand. The load log prints `[ngram] placement: disk|ram (reason); table N GiB, ...`.
- **Which MoE decode path Flash-Next takes:** hidden 2560, expert width 640, top-10, `mul1`, so not the fused `exl3_dec_moe` (needs multiples of 512) but the grouped HIP path (any 128-aligned width, `mul1`, top-k 6/8/10) [I, from `block_sparse_mlp.py`]. Mixed-bitrate layers (fractional bpw such as 2.05) need the mixed-K branch of the grouped path, with its own gates **[?]**. This is the case behind the README's Linux decode figures.
- **Windows kernels not yet exercised:** `gr_mix`, `qsa_pf`, `kda_fused_h`, DS41 kernels (see [deferred.md](../../deferred.md)); `gdn_hf` and `ple` have run.
- **Qwen3.5-35B-A3B already runs here** with the fused path: about 2700 tok/s prefill, 47 to 55 tok/s decode [V, [S6-09](09-qwen3-30b-a3b-moe-decode.md)]. So the MoE machinery works on Windows; this task tests the Flash-Next specifics.

## Plan

**Measurement rules** (learned in S6-08/S6-09): `tools/bench.py` against **127.0.0.1** (the Windows `localhost` delay), hands-off machine, no stray `EXL3_*`/`KYOJIN_*` variables in the window (`Get-ChildItem Env:`), interleave when comparing, at least 4 runs, first run after a start is warm-up, state the power state. Decode and prefill are read from `bench.py` and, if in doubt, the server's `qserve: request` lines (`prefill_s`, `generate_s`). Optional timeline: `KYOJIN_REQLOG_DIR` (an existing directory!).

### Phase 0: prepare (30 minutes)

1. Disk: 58 GiB for the smoke pack now, 79 GiB for phase 2, about 100 GiB if phase 2 goes to 4.05 (`C:\models` free space; the models are the only thing that matters here).
2. Record the baseline once: Windows-visible RAM (`(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB`), the Adrenalin "Variable Graphics Memory" value, driver version, `torch.cuda.mem_get_info()` and the 1 GiB allocation sweep of S1-04. Stop other GPU users.
3. Know the storage: which drive holds the model folder, its model, and whether it is the system drive (the n-gram gather is random reads; `EXL3_NGRAM_IO_THREADS` is tuned for NVMe).

### Phase 1: smoke test on a small quant (the point is "does it run", not quality)

```powershell
cd C:\Projects\kyojin
.venv-win\Scripts\hf.exe download turboderp/Qwen3.8-Flash-Next-exl3 --revision 2.05bpw_h4_ng4 --local-dir C:\models\flash-next-2p05
dir C:\models\flash-next-2p05
```
Check the file list for `config.json`, tokenizer files, `chat_template.jinja` (extract from `tokenizer_config.json` if missing, as for the 30B), `generation_config.json`, `ngram_embedding.safetensors`, and `mtp.*` tensors. Then serve with the driver's defaults (disk streaming, `-c 8192`):
```powershell
uv run --no-project python scripts/windows/kyojin_win.py serve --model C:\models\flash-next-2p05
```
Watch: the `[ngram] placement:` line, `qserve: vram` lines, Task Manager GPU "Dedicated GPU memory" and "Shared GPU memory", host RAM, disk activity during the first requests. Then, one short greedy request (install guide curl with `127.0.0.1`), then `tools/bench.py --max-tokens 256 --reps 3` twice. If the first start takes minutes, that is JIT/tuning; note which kernels build (`EXL3_LOG_JIT=1` logs them).

Record: loaded or failing step, GPU dedicated/shared at idle and during a request, host RAM, load time, prefill and decode tok/s, text sanity. If it fails: run folder (`logs\windows\<ts>-serve\`), first error, which kernel; **do not patch loader code ad hoc**, add a task. A fallback for a build/JIT failure of a single kernel is its `EXL3_*=0` switch, recorded as a finding.

Pass criterion: loads, generates coherent text, no TDR (Event Viewer > Display > 4101), decode in the tens of tok/s (the 35B-A3B reference is 47 to 55; this model is bigger).

### Phase 2: step up the quant

`3.05bpw_h5_ng5` (79 GiB total, about 49 GiB of weights), then `4.05bpw_h6_ng6` (99.9 GiB total, about 64 GiB of weights) if disk streaming holds. Repeat the Phase 1 measurements. Stop when GPU memory plus KV no longer fits, or decode drops below usable, and write down the margin. `2.18bpw_h6_ng8` (the original S6-04 target, 36 GiB of weights, 48 GiB table) is the largest-table case: the best test of the SSD side.

### Phase 3: n-gram table on SSD against in RAM

Per quant from phase 2 (start with `3.05bpw_h5_ng5`): same prompts, three settings, interleaved:

- **SSD stream (default)**, cold (first request after a start, page cache empty: reboot or the standby-list flush tool RamMap) and warm.
- **RAM:** `$env:EXL3_NGRAM_STREAM = "0"` (works on this branch and on 1.8.9; the server flag `--ngram-ram` exists only on 1.8.9, not on this 1.8.3-based branch). Needs table + 48 GiB of host memory; on the current 31.6 GB it cannot hold a 30 GiB table, so this phase is blocked until Phase 4 raises host RAM. Note the placement line and `MemAvailable` printed at load.
- Metrics: TTFT and prefill tok/s at 2K, 8K and 32K prompts (the gather is batched per 4096-token chunk), decode tok/s, and the extra time of the first request after a cold start. `EXL3_NGRAM_IO_THREADS` (32 / 128 / default) and `EXL3_NGRAM_WARM_TOKENS` as secondary knobs only if SSD streaming is visibly the bottleneck.

### Phase 4: lower the GPU-dedicated memory

Why: host RAM has to hold the page cache for streaming, or the whole table in RAM mode, plus the Python process and pinned staging; 31.6 GB is tight. GPU-dedicated memory is a setting of the AMD driver ("Variable Graphics Memory" in Adrenalin, or the UMA frame buffer size in the BIOS) **[I: exact name and location to be confirmed on the machine]**; it needs a reboot. Procedure, one setting at a time (for example 96, 64, 48 GB of dedicated memory):

1. Write down the current value so it can be restored. Change it, reboot.
2. Record Windows-visible RAM, `torch.cuda.mem_get_info()` and the S1-04 allocation sweep. **Open question [?]:** whether the usable GPU memory (about 100 GiB today) falls one-for-one with the dedicated setting, or whether Windows "shared GPU memory" keeps the ceiling high (Windows can map system memory for the GPU up to a fraction of RAM). This single measurement decides whether lowering the setting is cheap.
3. Run the smoke pack and the phase 2 pack in both n-gram modes. Record decode/prefill and any change in the dense models (27B, 35B-A3B) as a regression check: their numbers today are 12.7 to 14.4 tok/s and 47 to 55 tok/s.
4. Pick the setting with the best Flash-Next result that leaves the machine usable; write the rule of thumb into [install.md](../../install.md) (and `doc/windows/install.md` on the PR branch only after the setting is confirmed by a second person).

### Phase 5: write up

Evidence log table (quant, n-gram mode, dedicated setting, GPU/host memory, load time, prefill, decode, notes), a go/no-go on "Flash-Next as the Windows MoE target", and follow-up tasks for whatever failed (kernels, S6-05 host memory API, MTP/drafter).

## Done when

- A quant of Flash-Next generates on Windows with GPU and host memory at load and idle recorded, or the failing step and error are named with the run folder.
- The SSD against RAM comparison and the effect of the dedicated-memory setting are in the Evidence log with numbers, or the blocking reason is written down.
- The largest quant that fits, with its margin, and the recommended memory split are recorded.

## If it fails

- A single JIT/build failure of a kernel: record the kernel and error, try its switch off, add a task; do not edit loaders.
- GPU OOM at load: smaller quant first; the S1-04 sweep tells how much was really available.
- Streaming too slow on this SSD: try RAM mode after Phase 4 raises host RAM; otherwise the lower-`ng` quants (`ng3`, `ng4`) have smaller tables.
- Lowering the dedicated memory makes other things worse (27B, 35B): restore the value from step 1 and record the numbers.

## Original steps (before the pack sizes were split, 2026-10-31; superseded by the plan above)

1. Check disk (90.7 GB download) and stop other GPU users; run the S1-04 sweep first if anything changed.
2. `hf.exe download turboderp/Qwen3.8-Flash-Next-exl3 --revision 2.18bpw_h6_ng8 --local-dir C:\models\...`; check the file list for `mtp.*` tensors and `ngram_embedding.safetensors`; also look at yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz (size not checked).
3. `serve --model <dir> -c 8192` (small context first); watch GPU memory (`Dedicated Usage`) and the host RAM during the load.
4. One greedy request; then the soak commands from S4-05 with a short context.

## Evidence

(run folders, versions, numbers; fill in)

-

**Links.** [S1-04 ceiling](../s1-risk-spikes/04-gpu-memory-ceiling.md) · [S6-05 host memory](05-windows-host-memory.md) · [S6-09 (35B-A3B and the MoE dispatch)](09-qwen3-30b-a3b-moe-decode.md) · [deferred.md](../../deferred.md) · [env_vars.md](../../../env_vars.md) (`EXL3_NGRAM_*`)

Depends on: [S1-04](../s1-risk-spikes/04-gpu-memory-ceiling.md)
