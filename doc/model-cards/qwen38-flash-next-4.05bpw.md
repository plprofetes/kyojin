---
license: other
license_name: qwen-community-1.0
library_name: exllamav3
pipeline_tag: text-generation
tags: [exl3, quantized, qwen, moe, rocm, strix-halo, gfx1151, windows]
---
<!-- Draft for sharing. Every number is from doc/benchmarks/qwen38-flash-next-4.05bpw-20261118.md (run folder logs/, 2026-11-18). Cells marked "not measured" are empty on purpose. Fill the two <...> links before publishing. -->

# Qwen3.8-Flash-Next, EXL3 4.05 bpw: running it on one 128 GB Windows machine

**Size:** 125B parameters, 6B active per token (the upstream figures; the 51B n-gram table and 4B MTP head come on top). The Hugging Face sidebar count for an EXL3 pack is not the model's: the site counts packed 16-bit words.

**Hardware:** one AMD Strix Halo machine (Ryzen AI Max+ 395, 128 GB unified memory, Radeon 8060S, gfx1151) with a 96 GB GPU-dedicated pool, Windows 11. This is the only machine it has been tested on.

This card is a run guide and measurement report for the EXL3 quantisation of [Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next) published by turboderp: [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), branch `4.05bpw_h6_ng6`. The weights are unchanged; we did not requantise them. It runs on the Kyojin engine (built on ExLlamaV3), not on stock ExLlamaV3. Not affiliated with Qwen.

## Highlights

- **It fits and it is quick.** About 64 GiB of weights sit in the GPU pool; the 36 GiB n-gram table stays on the SSD and is read as needed. Decode **30 to 41 tokens per second** with the server's default drafting (`mix`), 20 without drafting.
- **Long prompts do not slow it down.** Prompt processing stays at **1180 to 1280 tokens per second** from 7K to 33K tokens: a 33K-token prompt is ready in 25 seconds.
- **Faster than the 122B on the same machine.** Qwen3.5-122B-A10B (4.06 bpw) gives 17.3 to 17.5 tok/s decode (no drafting) and about 1030 tok/s prefill with the same engine and test text.
- **Context:** the model supports 262,144 tokens. The server starts with a 131,072-token context; the longest prompt we ran was 33K tokens.
- **Thinking you can steer:** thinking on or off per request, three effort levels, and a hard token cap (`thinking_budget`). See "Recommended settings".
- **Feedback wanted.** Tried it on your own Strix Halo or Ryzen AI Max machine? Tell us your tokens per second and your hardware: <issues link>.

## How to run it (Windows)

Needs about 100 GiB of free disk for the pack and the Kyojin Windows setup from the engine's `doc/windows/install.md`.

```powershell
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6 --local-dir C:\models\flash-next-4p05
cd <your kyojin checkout>
.\scripts\windows\serve-flash-next.ps1                      # LAN server, 128K context, bearer token, drafting on
.\scripts\windows\serve-flash-next.ps1 -Token my-secret-1234 -OpenFirewall   # your own token; firewall rule once, elevated shell
```

`serve-flash-next.ps1` listens on `0.0.0.0:8000`, needs `Authorization: Bearer <token>` on every route except `GET /health`, and prints the base URLs. The token is `-Token` if you pass one, else the first line of `%USERPROFILE%\.kyojin\flash-next-token.txt` (a new `qwen-flash-next-<8 hex>` is generated there on the first run). It is short on purpose: use it on a network you trust, or pass a longer `-Token`. Plain HTTP only.

```bash
curl http://<ip>:8000/v1/chat/completions -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Hello"}]}'
```

- The model name in requests is `qwen3.8-flash-next` (`-ModelId` changes it). Any other name gets 400. One request is decoded at a time; the rest queue.
- Runner flags: `-Thinking` (thinking on by default), `-Effort low|medium|xhigh`, `-DraftPolicy off|mix|mtp` (default `mix`), `-Ctx`, `-MaxReply`, `-Port`, `-Model`.
- To run without the script: `kyojin_win.py serve --model C:\models\flash-next-4p05 --port 8000 --ctx 131072 -- --draft-policy mix` (no token, listens on `127.0.0.1` only). Use `127.0.0.1`, not `localhost`: on Windows `localhost` adds a delay.
- **First start:** the server compiles and tunes kernels, then answers: 160 s to ready with a 128K context, 67 to 159 s on other starts. The port opens only when the model is loaded; `GET /health` shows progress.
- Image input was **not tested** on this model (all runs used `--no-vision`; the runner starts the same way).

## Speed

One machine, Windows 11, default engine settings, greedy decoding, text only. Prompt for the standard figures: 2680 tokens of fixed text (`doc/knobs.md` in the engine repo), 256 new tokens, median of 3, two passes. Tokens per second.

| Decode | No drafting (`off`) | MTP only (`mtp`) | Default (`mix`: MTP + n-gram lookup) |
|---|---|---|---|
| Code | 19.9 | 32.4 to 33.3 | **37.7 to 41.2** |
| Chat | 19.9 | 23.0 to 23.3 | **32.2 to 33.3** |
| Prose | 19.9 | 17.8 to 19.3 | **29.4 to 31.3** |
| Prompt processing, 2680 tokens | 1124 to 1164 | 1037 (second pass) | 1027 to 1094 |

- **Use `mix` (the runner's default).** It is the fastest of the three on every prompt type, and it did not lose speed on free prose, where MTP alone did. Drafting does not change the output: every drafted token is verified.
- The decode figures for `off` and `mtp` were taken with the server's default of thinking on, so the 256 tokens were mostly reasoning text; `mix` was measured with thinking on and off, and the two agree within 9 % (the ranges above cover both).
- Decode drops to 33 to 38 tok/s in long reasoning (thousands of tokens) from about 42 in short answers **[V, effort probe]**.
- The first benchmark pass after a server start measured about 545 tokens per second of prompt processing, half the steady 1030 to 1160, with MTP and with `mix`; later passes were normal. We do not know why. The first request after a start takes longer (6 s against 2.2 s for the same 2.6K-token prompt).

Prompt processing by prompt length (plain decoding, context 40,960, prefix cache never hit, page cache warm):

| Prompt tokens | Tokens per second |
|---|---|
| 6,890 to 8,320 | 1180 to 1222 |
| 26,748 to 33,251 | 1272 to 1281 |

## Memory

- GPU pool 96 GB; the weights take about 64 GiB of it. Exact GPU memory in use was **not recorded**.
- Windows shows 31.6 GB of host RAM on this machine (the rest is the GPU pool). The n-gram table is streamed from disk and was not loaded into RAM; the in-RAM mode was **not tested** and needs more host memory than this machine shows.
- Run it from an SSD. We did not test a slower disk, and we did not test with a cold page cache after a reboot.

## What was and was not tested

| Tested | Not tested |
|---|---|
| Loads and generates coherent text on Windows (a short chat answered correctly) | **Quality:** no perplexity, no task scores, no comparison with the unquantised model |
| Decoding speed with no drafting, MTP and `mix`; prompt processing up to 33K tokens; the LAN runner with bearer auth; thinking on/off, three efforts, a thinking budget (one seed each) | Context above 128K; a prompt above 33K; decode speed at long context |
| No display-driver reset in the test window | Image input; tool calls; other hardware (NVIDIA, other AMD GPUs, less than 128 GB) |
| | That drafted and plain decoding give identical tokens; the drafting acceptance rate (not logged); a per-switch knob sweep |

## Recommended settings

Per request unless marked "server". **[Qwen]** = the model authors' recommendation (their README); **[V]** = measured here; **[I]** = inferred, not tested.

**Sampling**

| Use | temperature | top_p | top_k | min_p |
|---|---|---|---|---|
| Thinking on (default for hard tasks) **[Qwen]** | 1.0 | 0.95 | 20 | 0 |
| Thinking off, chat **[Qwen]** | 0.7 | 0.8 | 20 | 0 |
| Code edits, patches, anything that must repeat | 0 (greedy) | | | |

- The server's own defaults are 1.0 / 0.95 / 20 / 0 (the pack's `generation_config.json`). Qwen also suggests `presence_penalty` 1.5 when thinking is off; **this server ignores `presence_penalty`, `frequency_penalty` and `repetition_penalty`** **[I: no code reads them; not probed on this model]**. A repetition loop is stopped by the server's loop guard instead (`finish_reason: "length"`, `stop_reason: "loop_guard"`).
- Greedy output is repeatable but is not guaranteed to be the same with and without drafting at the last digit **[?: not checked on this model]**.

**Thinking, effort, budget**

| Setting | How | Recommendation |
|---|---|---|
| Thinking on / off | `"enable_thinking": true/false` or `"chat_template_kwargs": {"enable_thinking": ...}`; the runner defaults to off (`-Thinking` flips it); the bare server defaults to on | **Off for chat, tool calls and quick edits; on for maths, planning, debugging.** Off answered the easy test prompts correctly in 230 to 620 tokens **[V]**. |
| Effort | `"reasoning_effort": "low" \| "medium" \| "xhigh"` (`high` = `xhigh`, `none` = thinking off); server default `medium` (`-Effort`), Qwen's own default is `xhigh` | **`medium`.** Effort is an instruction in the prompt, not a hard limit: on three easy prompts the whole reply was 260 to 940 tokens at every level, and on the code prompt the reasoning grew with the level (1,449 / 1,667 / 2,405 characters at low / medium / xhigh) but not on the others **[V, one seed]**. Use `low` to save time, `xhigh` for hard problems, and `thinking_budget` when you need a limit. |
| Thinking budget | `"thinking_budget": <tokens>` | **1,024 to 4,096 for interactive use.** It is a hard cap: 512 shortened the code answer from 927 to 798 tokens, and a cap of 256 made the model answer after about 760 characters of reasoning **[V]**. Too small a cap gives a wrong answer sooner: on a 100-case counting problem (right answer 14), caps of 256 and 1,024 answered 4 and 36 **[V]**. Leave the budget unset for hard problems and set `max_tokens` high instead. |
| Reply length | `"max_tokens"` (server default 16,384 in the runner) | Set it at least **2,048 above `thinking_budget`**; if the budget uses all of `max_tokens` the reply comes back empty with `finish_reason: "length"`. On the counting problem, thinking at every effort level and plain answering all used the full 12,000 tokens without an answer, so give hard problems 16K or more and expect minutes (about 5 minutes at 12,000 tokens) **[V]**. Qwen suggests up to 262,144 tokens of reasoning and 131,072 of answer for agent tasks **[Qwen]**. |

Reasoning comes back in `reasoning_content`, the answer in `content`. Each test above ran once with a fixed seed; treat the token counts as examples.

**Server settings** (runner flags or `serve.py` flags; the defaults are the measured configuration, none was tuned here)

| Setting | Default in the runner | Notes |
|---|---|---|
| `-DraftPolicy` | `mix` | Fastest measured; `off` for plain decoding; `mtp` is slower than `mix` everywhere. Changing it needs a restart. |
| `-Ctx` | 131,072 | Starts and serves with 128K (about 3 GiB of KV cache). Longest prompt run: 33K. Up to 262,144 is the model's limit **[?: not tried]**. A bigger context costs memory, not speed. |
| `-MaxReply` | 16,384 | Reply cap when a request sends no `max_tokens`. |
| `-Effort` / `-Thinking` | `medium` / off | See above. |
| `--cache-bits 8` (serve.py) | 0 (16-bit KV) | About 40 % smaller KV pages; output differs slightly from the 16-bit cache **[I, README of the engine; not tried here]**. Only needed beyond 128K. |
| `--ngram-ram` or `EXL3_NGRAM_STREAM=0` | off (table streamed from SSD) | Loads the 36 GiB table into RAM. Not usable on this machine (31.6 GB of host RAM). Streaming cost nothing measurable up to 33K tokens, so leave it **[V]**. |
| `EXL3_NGRAM_IO_THREADS` / `EXL3_NGRAM_WARM_TOKENS` | 128 / 8192 | Only for a slow disk **[I]**; not changed. |
| `--sessions` | 1 | One request at a time. More sessions share the GPU and can change greedy output; not tested on this model. |
| `KYOJIN_SSE_PING_S` | 15 (set by the runner) | Keep-alive comment every 15 s in streams, so a long prefill does not drop the connection. Set the client's read timeout to several minutes. |
| `--slot-save-path` | `~\.kyojin\slots-flash-next` | Where saved prompt-cache slots go (`POST /slots/0?action=save`). |

The engine's full list is `doc/knobs.md` and `doc/env_vars.md`; a per-switch sweep (`tools/sweep.py`) has not been run on this model.

## Licence and credits

Qwen Community License 1.0 (copy shipped with the pack as `LICENSE`). Read it before commercial use: it asks for the model name to be shown in very large products and requires a separate licence from Qwen for hosted model-as-a-service and AI coding or office assistants. Base model Qwen3.8-Flash-Next by the Qwen team. EXL3 quantisation by turboderp. Engine: Kyojin (<link>), built on ExLlamaV3 by turboderp; Windows support and these measurements by <your name or org>. Provided as is, without warranty.
