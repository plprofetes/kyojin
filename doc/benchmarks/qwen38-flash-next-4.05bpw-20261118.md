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

## Addendum 2: default drafting (`mix`), thinking, effort, budget, LAN runner (same day)

Server: `scripts/windows/serve-flash-next.ps1` (new; bearer token, `0.0.0.0`, ctx 131072, `--draft-policy mix`, thinking default off, effort `medium`, serve.py's built-in chat template). Start-up 160 s; auth checked: no token and a wrong token give 401, the right one 200, `GET /health` 200 without a token, the LAN address answers **[V]**.

**Correction to the sections above:** the plain (`off`) and `mtp` servers were started without `--no-thinking`, so thinking was on by default and the 256 benchmark tokens were mostly reasoning text. The figures stand for thinking-mode text; `mix` below was run both ways.

`tools/bench.py` (through a scratch wrapper that adds the bearer header; `bench.py` has no auth option), corpus `doc/knobs.md`, two passes per setting, medians of 3:

| | Thinking off, pass 1 / 2 | Thinking on, pass 1 / 2 |
|---|---|---|
| Decode, prose | 29.9 / 29.7 | 31.3 / 29.4 |
| Decode, chat | 32.6 / 32.3 | 33.3 / 32.2 |
| Decode, code | 41.2 / 41.1 | 39.6 / 37.7 |
| Prefill, ~2680 tokens | 546.0 / 1094.4 | 1070.3 / 1027.0 |

- **`mix` beats `off` (19.9) and `mtp` (17.8 to 33.3) on every prompt type** **[V]**: +50 % prose, +62 % chat, +90 to +107 % code against plain. The first pass after the start again showed half the prefill speed (546) in the thinking-off series **[V, cause unknown]**.

Effort and thinking-budget probe (greedy off; temperature 1.0 / top-p 0.95 / top-k 20, seed 1, one request per cell, `mix`), completion tokens (reasoning characters):

| Prompt | off | low | medium | xhigh | medium + budget 512 |
|---|---|---|---|---|---|
| math (train catch-up) | 624 (0) | 327 (314) | 539 (495) | 262 (575) | 539 (495) |
| count letters | 229 (0) | 378 (615) | 371 (587) | 358 (1096) | 371 (587) |
| code (fib mod, O(log n)) | 322 (0) | 873 (1449) | 927 (1667) | 938 (2405) | 798 (1328) |

All 12 answers on the three easy prompts were correct (code checked by running it) **[V]**. Budget 2048 changed nothing on them (the reasoning was under it). The hard prompt (`n*n+n+41` composite for 1 <= n <= 100, true answer 14), `max_tokens` 12000: off, low, medium and xhigh each used all 12,000 tokens (286 to 360 s) with no answer; `thinking_budget` 256 and 1024 answered after 1,072 and 1,361 tokens and were wrong (4 and 36) **[V]**.

- **Effort is a soft instruction**, not a limit: the reasoning length moved with it on the code prompt (1.0 : 1.15 : 1.66 characters, low : medium : xhigh) and not clearly on the others, from one seed each **[V that it is noisy, I that it is a prompt hint]**. **`thinking_budget` is a hard cap that works** (798 against 927 tokens at 512; a hard stop at 256) **[V]**, and a cap that is too small makes the answer wrong rather than short **[V]**.
- Decode in the probe: 42 to 45 tok/s on short answers, 33 to 38 tok/s in the long reasoning runs **[V]**.
- Not run: `presence_penalty` and the other penalties (the code does not read them; not probed here), a per-switch knob sweep, more than one seed, a larger set of hard prompts.

Raw output: `logs/bench-flashnext-4p05-mix-think{off,on}-run{1,2}.md`, `logs/effort-probe-mix.{txt,json}`, `logs/effort-probe-hard-mix.{txt,json}`, server log `logs/serve-flash-next-runner-mix.log` (gitignored). Model card: [../model-cards/qwen38-flash-next-4.05bpw.md](../model-cards/qwen38-flash-next-4.05bpw.md).
