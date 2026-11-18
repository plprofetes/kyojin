# Qwen3.8-Flash-Next, EXL3 4.05 bpw (h6, ng6): first run on Windows

Date: 2026-11-18. Machine: Windows 11, AMD Radeon 8060S (gfx1151), 128 GB unified memory, GPU-dedicated pool 96 GB, host RAM visible to Windows 31.6 GB. Branch `rebase/upstream-main-1.8.26` (commit `ca133e8`), default build and knobs.

## Pack

- Source: `turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6`, in `C:\models\flash-next-4p05` (100.1 GiB, 27 files, sizes checked against the Hub listing). **[V]**
- `ngram_embedding.safetensors` is 36.4 GiB (not counted in the weights); the 9 weight shards are about 64 GiB. The index has 303,121 tensors, 6,203 of them `mtp.*`, so `--draft-policy mtp` has weights to load (not run here). **[V]**
- Served name is `Qwen3.8-Flash-Yamz` whatever the pack (a fixed label in `serve.py`). **[V]**

## Method

`kyojin_win.py serve --model C:/models/flash-next-4p05 --port 8000 -- --draft-policy off --no-vision` (ctx 8192, n-gram table streamed from disk by default), then twice in a row `tools/bench.py --base http://127.0.0.1:8000/v1 --reps 3 --max-tokens 256 --corpus doc/knobs.md --json` against `127.0.0.1`. The corpus is the same fixed file as the 2026-11-17 122B run (2680-token prompt), so prefill is comparable. The server had already run its start-up warm-up before pass 1. No stray `EXL3_*`/`KYOJIN_*` variables were set. Power state was not recorded.

## Results

| | Pass 1 | Pass 2 |
|---|---|---|
| Prefill, ~2680 tokens (median of 3) | 1163.8 tok/s | 1123.6 tok/s |
| Decode, prose / chat / code (median of 3) | 19.7 / 19.7 / 19.7 tok/s | 19.9 / 19.9 / 19.9 tok/s |

- Server start-up (load + warm-up): 159 s. **[V]**
- First request after start (prompt 2569, cold kernels and page cache): 6.03 s, 426 tok/s; the next ones 1156 to 1226 tok/s. **[V]**
- Decode in the server's own log: 255 tokens in about 12.9 s (`generate_s` 12.83 to 12.93). **[V]**
- Load log: `[ngram] placement: disk (default: disk streaming); table 36.4 GiB, host MemAvailable ? GiB`. Windows cannot report host memory yet, hence the `?` (S6-05). **[V]**
- No display driver reset (Event Viewer, Display 4101) in the two hours around the run. **[V]**
- Sanity: one greedy chat request gave a correct two-sentence MoE explanation and the right working for 17 * 23 = 391 in the reasoning text; the reply stopped at the 200-token cap. **[V]**

## Reading

- **Decode about 19.8 tok/s** on this 4.05 bpw pack, against 17.3 to 17.5 for the 122B-A10B heretic 4.06 bpw on the same build and corpus **[V]**. The two decode figures agree to 1 % between passes. The identical prose / chat / code values are what `bench.py` prints from medians of 3 runs; the server log shows the same 12.8 to 12.9 s for every request, so decode speed here does not depend on the prompt type **[I]**.
- **Prefill about 1120 to 1160 tok/s**, a little above the 122B (1029) **[V]**. The 3 % gap between the two passes is within what the journal records as normal variation on this machine; the figure is "about 1100 to 1150".
- **The n-gram table on SSD did not slow decode or steady-state prefill** at these prompt sizes **[V]**. Only the first request after a start paid for it (cold page cache, kernel tuning): 6 s against 2.2 s **[I, the two causes are not separated]**.
- **Fit:** about 64 GiB of weights are resident in the 96 GB pool, with room left for context 8192 **[V that it loaded and ran; GPU memory in use was not recorded]**.
- This is the first run of the Flash-Next kernels on Windows (grouped MoE decode with 640-wide experts, n-gram gather, PLE); they loaded and produced coherent text with no kernel switched off **[V]**.
- The `[ngram]` line, the `swapped out during the load` line (`TypeError: argument of type 'NoneType' is not iterable`, all counters 0) and the `bench.py` cp1250 `UnicodeDecodeError` above the table are known Windows gaps in the host-memory API and the bench reader, not failures of this run **[I for the swap line: not investigated]**.

## Not run

- Speculative decoding (`--draft-policy mtp`), n-gram table in RAM (`EXL3_NGRAM_STREAM=0`, blocked by the 31.6 GB host RAM), cold-cache runs after a reboot, prompts of 8K and 32K, context above 8192, vision, quality or perplexity, lower GPU-dedicated memory, the smaller packs of S6-04 phase 1 and 2.
- One server start; no repeat start to check start-up variation.

Raw output: `logs/bench-flashnext-4p05-off-run{1,2}.md`, server log `logs/serve-flash-next-4p05-off.log` (gitignored).

## Addendum: MTP drafting and long prompts (same day)

### MTP drafting (`--draft-policy mtp`, ctx 8192, no vision)

Same method and corpus as above; start-up 138 s, `speculative=True`, drafter 2 s, two-stage draft head. Two `bench.py` passes (medians of 3):

| | Draft off (pass 2) | MTP pass 1 | MTP pass 2 |
|---|---|---|---|
| Decode, prose | 19.9 tok/s | 17.8 | 19.3 |
| Decode, chat | 19.9 tok/s | 23.3 | 23.0 |
| Decode, code | 19.9 tok/s | 33.3 | 32.4 |
| Prefill, ~2680 tokens | 1123.6 tok/s | 545.3 | 1037.5 |

- **MTP helps where output is predictable:** code +63 % (32.4 to 33.3 against 19.9), chat +16 to +17 % **[V, two passes agree within 3 %]**. **Prose is a wash or a loss** (17.8 to 19.3 against 19.9, up to -11 %) **[V]**: the verify step costs more than the accepted drafts return. A policy that keeps MTP on for code only is not available as a switch here; `--draft-policy` is per server **[I]**.
- Per-request server lines (generate time for 256 tokens): 8.0 s, 14.6 s, 12.4 s, 9.0 s for four last requests, i.e. 17.5 to 31.9 tok/s **[V]**. The server log prints no acceptance rate, so none is reported **[?]**.
- **Prefill pass 1 was 545 tok/s**, half of every other figure (pass 2: 1037.5, close to draft off). The first pass after this start paid for something once (kernel tuning or a cold page cache); the draft-off server did not show it in its first pass **[V that it happened, ? why]**. Steady-state prefill is not slowed by MTP beyond about 8 % (1037 against 1124), which is within the passes' own spread **[I]**.
- Plain-against-speculative output exactness was not checked (S6-10).

### Long prompts (draft off, ctx 40960, no vision)

Start-up 67 s (warm page cache). Prompts are random slices of `doc/**/*.md` with a unique salt, so the prefix cache never hit; `max_tokens` 1, greedy. The slice sizes came out off target because Markdown with code and tables is 2.19 tokens per word; the prompt tokens are the real counts. Script: three requests per size, wall time of the whole request; the server's `prefill_s` agrees within 0.2 s.

| Prompt tokens | Prefill |
|---|---|
| 6890 / 7155 / 8320 | 1180 / 1184 / 1222 tok/s |
| 26748 / 28160 / 33251 | 1272 / 1281 / 1277 tok/s |

- **Prefill does not slow down with length** (it even rises from about 1190 to about 1280 tok/s, as fixed per-chunk costs spread over more tokens) **[V]**. A 33K-token prompt takes 25 s **[V]**.
- **The SSD-streamed n-gram table is not a bottleneck up to 33K tokens** with a warm page cache **[V]**. A cold-cache run after a reboot was not done, so the first-request cost of 6 s seen in the main run stays unexplained **[?]**.
- Context 40960 loaded with no memory trouble; no display driver reset (Event Viewer 4101) in the two hours around these runs **[V]**. Decode speed at long context was not measured.

Raw output: `logs/bench-flashnext-4p05-mtp-run{1,2}.md`, `logs/longprompt-flashnext-4p05-off.txt`, server logs `logs/serve-flash-next-4p05-{mtp,off-ctx40k}.log` (gitignored).
