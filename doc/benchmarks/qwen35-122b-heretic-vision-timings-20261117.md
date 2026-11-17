# Qwen3.5-122B-A10B heretic-v2 (EXL3 4.06 bpw), image input: timings and correctness on kyojin 1.8.26

Date: 2026-11-17. Machine: Windows 11, AMD Radeon 8060S (gfx1151), 128 GB unified memory, GPU-dedicated pool 96 GB. Branch `rebase/upstream-main-1.8.26` (94 local commits on upstream `b2f0d5f`, version 1.8.26), extension rebuilt clean with `kyojin_win.py build --clean` (12.9 min). Text-only speed of the same build: [benchmark in the 1.8.15 report](qwen35-122b-heretic-4.06bpw-20261110.md), 17.3 to 17.5 tok/s decode (unchanged).

## Setup

- Pack: `C:\models\qwen35-122b-heretic-4.06bpw`. Vision tower loaded (`vision=True`, start-up 158 s, warm-up 96 s).
- Server: `tools/qwen/serve.py --model ... --port 8000 -c 8192 --draft-policy off`, started directly because `kyojin_win.py serve` hard-codes `--no-vision`. No `EXL3_*` variables.
- Client: OpenAI-style `/v1/chat/completions`, base64 PNG `image_url`, `temperature 0`, `127.0.0.1`.
- Seven images drawn by script with known content: shapes with a text label (320x200 and 1280x720), a bar chart (800x500), an invoice page (1024x768 and 1920x1080), a 6x6 grid of coloured tiles (768x768 and 2048x2048; truth red 18, green 7, blue 11).
- Three modes: **full thinking** (default, `max_tokens` 1200 to 1500), **thinking off** (`enable_thinking: false`), **thinking capped** (`thinking_budget: 200`: after 200 thinking tokens the server closes the thought and the model answers).
- Each image was changed by one or two pixels between rounds so the server's image cache could not serve it from an earlier round.

## Result: wall time and correctness per mode

Wall time of one request, first time the server sees the image. "gen" counts every generated token, thinking included.

| Image | Full thinking | Thinking off | Thinking capped at 200 |
|---|---|---|---|
| shapes 320x200 | 24.8 s, 422 gen, correct | 7.1 s, 115 gen, correct | 15.2 s, 265 gen, correct |
| bar chart 800x500 | 19.6 s, 313 gen, correct | 10.6 s, 157 gen, correct | 17.0 s, 274 gen, correct |
| invoice 1024x768 | 35.8 s, 589 gen, exact | 7.2 s, 97 gen, exact | 18.6 s, 299 gen, exact |
| tiles 768x768 | 62.2 s, 1079 gen, 18/7/11 correct | 28.9 s, 486 gen, 18/7/11 correct | 22.3 s, 364 gen, 18/7/11 correct |
| shapes 1280x720 | 29.2 s, 426 gen, correct | 8.0 s, 106 gen, correct | 17.5 s, 273 gen, correct |
| invoice 1920x1080 | 52.4 s, 717 gen, exact | 9.9 s, 97 gen, exact | 20.8 s, 301 gen, exact |
| tiles 2048x2048 | 80.2 s, 910 gen, 18/7/11 correct | 35.4 s, 486 gen, 18/7/11 correct | 22.3 s, 257 gen, **wrong**: 17/9/10 (first try 16/10/10) |

**[V]** for every cell; correctness compared by hand with what was drawn. Raw answers are in the JSON files named at the end.

## Reading

- **Thinking off is as correct as full thinking on these seven images and 2 to 7 times faster** (7 to 11 s for reading and describing; 29 to 35 s for the counting task). Even with thinking off, the model reasons in its visible answer when asked to count (row by row, 486 tokens) and gets it right.
- **A 200-token thinking cap is the worst of the three on the hard case.** It was right on six images and wrong on the 2048x2048 tile count in both tries (the count is dense: 36 tiles in 4096 vision tokens). It also was not faster than thinking off on the easy images (15 to 21 s against 7 to 11 s), because the model still writes a long formatted answer after its short thought **[V]**.
- **Decode speed is 17 to 18 tok/s in all modes**, equal to text-only, so only the number of generated tokens decides the wall time **[V]**.
- **Suggestion for an agent loop [I, from these seven images only]:** thinking off for reading and describing; full thinking, or thinking off with a request to count step by step, for counting and other exact tasks. Do not use the 200-token cap for dense images.
- Not covered: photographs, several images per request, long image-heavy conversations, real browser or phone screenshots, GGUF or Flash-Next.

## Cost of the image itself

The server prints the vision tower time per new image (`qserve: image WxH -> N tokens, tower M ms`). Steady cost the first time an image is seen is tower plus prompt prefill (both from the server log, thinking-off runs):

| Image | Vision tokens | Tower | Prefill | Tower + prefill |
|---|---|---|---|---|
| 320x200 | 77 | 0.04 s | 0.43 s | 0.5 s |
| 800x500 | 400 | 0.13 s | 1.25 s | 1.4 s |
| 1024x768 | 768 | 0.26 s | 1.37 s | 1.6 s |
| 768x768 | 576 | 0.18 s | 1.21 s | 1.4 s |
| 1280x720 | 880 | 0.33 s | 1.62 s | 1.9 s |
| 1920x1080 | 2040 | 1.04 s | 3.44 s | 4.5 s |
| 2048x2048 | 4096 | 3.34 s | 3.94 s | 7.3 s |

- **This corrects my first draft.** It credited the gap "total minus prefill minus decode" to image encoding (1 to 15 s). That gap is mostly something else: in the first round the 2048x2048 request spent 14.6 s outside prefill and decode, and its prefill ran at 300 tok/s (13.8 s) against 1049 tok/s (3.9 s) in the later thinking-off round for the same size. So the first request of a new image size class in a server's life also pays a one-time cost of about 10 s (the tuning of new row classes that `doc/EXL3_CONV_GEMM.md` describes), and the steady figures are the ones in the table **[I: the cause is inferred from the repeat, not traced]**.
- **A repeated image costs nothing:** same bytes means the embedding is reused (no tower run) and the KV prefix matches in 256-token blocks (second request of the same image: 256 to 4096 tokens cached) **[V]**.

## Image cache check: no stale answers

The cache keys an image by the SHA-256 of its bytes and gives every freshly built embedding new token ids from a counter that only goes up (`tools/strix_halo/image_cache.py`, `exllamav3/tokenizer/mm_embedding.py`), so a different image cannot match old KV pages. Checked with two images of the same 640x400 size and different text, thinking off:

| Request | Answer | cached tokens |
|---|---|---|
| A, "ALPHA 9031" | `ALPHA 901` | 0 |
| B, same size, "CHARLIE 7714" | `CHARLIE 714` | 0 |
| A again | `ALPHA 901` | 256 |

Each answer matches its own image, B did not inherit A's text, and the repeat of A was cached **[V]**. Both answers dropped one character from the text ("9031" read as "901", "7714" as "714") **[V]**; with thinking on, the same kind of label was read exactly in the first round. Whether thinking off, the terse prompt ("answer with the text only") or the font size causes this is open **[?]**.

- **A pitfall in the capped mode:** with `thinking_budget` the server cancels the job at the budget and restarts it from its own prompt plus the kept thinking, and the usage block of that second job reports the whole prompt as cached (`prompt_tokens_details.cached_tokens` equal to `prompt_tokens`). That is not a cross-request cache hit **[I from the server's own docstring and the logs]**; ignore `cached` for budgeted requests.

## Notes on the method

- The first attempt at the thinking-off arm stopped after two images: printing a result containing a check mark raised `UnicodeEncodeError` (cp1250 on redirected output), and the environment-variable fix had no effect because `python -I` ignores `PYTHON*` variables. See "Text encoding on the target" in [../windows/install.md](../windows/install.md). The numbers above come from reruns with `python -X utf8 -I`.
- One run per image and mode; temperature 0, so repeats should mostly match, except where the cap changes the answer (the 2048x2048 count differed between its two tries).

Raw data (scratchpad, not in the repository): `image-suite.json` (full thinking, two runs per image), `r3-off.json` (thinking off), `r2-b200.json` (capped), `cache-test2.log`, and the server log `serve-122b-vision.log`.
