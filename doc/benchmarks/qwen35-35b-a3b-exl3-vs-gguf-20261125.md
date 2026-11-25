# Qwen3.5-35B-A3B on Windows (gfx1151): EXL3 4.00bpw vs GGUF UD-Q4_K_XL, upstream 1.9.3+44

Date 2026-11-25. Build: branch `rebase/upstream-main-1.9.3-44` (upstream `116a883`, 1.9.3 plus 44 commits) plus the GGUF edits of commit `03b9b01` (see "Changes needed"). Machine: Strix Halo gfx1151, dynamic GPU memory mode, one server at a time.

## Result: controlled run (same context, same prefill chunks, interleaved)

Eight server starts, one `tools/bench.py --base http://127.0.0.1:8000/v1 --corpus doc/knobs.md --reps 3` each (corpus ~2680 tokens, temperature 0, drafting off), `--ctx 16384` for every start, `EXL3_PREFILL_CHUNK` set explicitly. Round 1 order: EXL3-4096, GGUF-8192, EXL3-8192, GGUF-4096; round 2 the reverse. Each cell is the median of 3 inside one start; the two numbers are round 1 / round 2 **[V]**.

| Arm | Prefill tok/s | Decode tok/s (prose) | Decode chat / code |
|---|---|---|---|
| EXL3, chunk 4096 (its default) | 2524.4 / 2476.1 | 48.3 / 46.4 | 48.2 / 46.6, 48.2 / 46.5 |
| EXL3, chunk 8192 | 2457.9 / 2489.1 | 46.3 / 45.8 | 46.4 / 46.0, 46.3 / 45.7 |
| GGUF, chunk 4096 | 1797.2 / 1803.9 | 41.1 / 41.5 | 41.3 / 41.4, 41.2 / 41.3 |
| GGUF, chunk 8192 (its default) | 1818.1 / 1768.0 | 42.1 / 41.2 | 42.1 / 41.0, 42.2 / 41.0 |

- **Prefill:** GGUF about 1800 against EXL3 about 2490 tok/s, that is **-28 %**. The chunk (4096 or 8192) changes neither format by more than the round-to-round spread (up to 3 %) **[V]**.
- **Decode:** GGUF 41.0 to 42.1 against EXL3 45.7 to 48.3 tok/s, about **-11 %**. Prose, chat and code agree within 0.3 tok/s inside every start **[V]**. The GGUF reads 14 % more bytes per token (20.71 against 18.1 GiB on disk), so the gap is of the size the weight bytes alone would give **[I]**.
- **Start-up** (server process to `/health` ok, file cache warm): GGUF 30 to 31 s for both chunks; EXL3 20 s at chunk 4096 but 112 to 121 s at chunk 8192, in both rounds. Cause not looked at **[?]**.
- **Spread between starts of the same arm:** prefill up to 3 %, decode up to 4 % (EXL3-4096, 48.3 against 46.4). The format gaps are several times larger.
- **Earlier single run, the same day** (EXL3 at `--ctx 8192`, GGUF at `--ctx 16384` with its default chunk, one start each): EXL3 prefill 2727.8, decode 43.1-43.3; GGUF prefill 1735.8, decode 42.6-42.8. That run suggested equal decode; it is superseded: its EXL3 decode was 7 to 10 % below all four EXL3 starts here, while its prefill was 10 % above them. One start can land that far off, so a single start per format is not enough for this model **[V]**.
- Not compared: output quality (the GGUF only answered a short question coherently, thinking on), MTP / mix drafting (the GGUF has no MTP head here), long context, cold cache, vision, GPU memory use.

## How each was served

| | EXL3 | GGUF |
|---|---|---|
| Source | `turboderp/Qwen3.5-35B-A3B-exl3` `4.00bpw`, `C:\models\qwen35-35b-a3b-4bpw` | `unsloth/Qwen3.5-35B-A3B-GGUF` `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`, `C:\models\qwen35-35b-a3b-ud-q4kxl-gguf` (22 241 950 336 bytes, equal to the Hub listing) |
| Command | `EXL3_PREFILL_CHUNK=<n> kyojin_win.py serve --model <dir> --port 8000 --ctx 16384` | `EXL3_GGUF_GENERIC=1 EXL3_PREFILL_CHUNK=<n> kyojin_win.py serve --model <dir> --port 8000 --ctx 16384` |
| Context | 16384 here (8192 in the earlier single run) | 16384: at 8192 the warm-up job needs 33 pages and 32 exist, so the server exits |

GGUF directory (the loader wants a config directory, not a bare file): `loader/gguf_config.write_config_dir()` wrote `config.json`, `chat_template.jinja`, the tokenizer files and the `model.gguf` link from the GGUF header; `generation_config.json`, `preprocessor_config.json` and `video_preprocessor_config.json` were copied from the EXL3 pack (the model class opens the preprocessor file even with `--no-vision`). A copied EXL3 `config.json` does not work: the loader then looks for safetensors **[V]**.

## Changes needed to serve this GGUF on Windows (commit `03b9b01`)

1. `exllamav3/loader/gguf_collection.py`: `os.pread` does not exist on Windows (`AttributeError` while reading the embedding), and `os.open(..., O_RDONLY)` is text mode there. Now `O_BINARY` and a seek-and-read under a lock on `nt`; other platforms still use `os.pread` **[V]**.
2. `exllamav3/architecture/minicpm_v47.py`, `_fp_override`: this unsloth file stores `ssm_alpha` / `ssm_beta` (and most non-expert weights) as Q8_0, which the helper rejected (`AssertionError: blk.0.ssm_beta.weight: type 8`). Added a Q8_0 branch next to the existing embedding override **[V]**.
3. `EXL3_GGUF_GENERIC=1` is also needed: `Linear.load_gguf_` lets an in-memory override win over the stored blocks only under that flag, and without it the Q8_0 `in_proj_a/b` layers fail with `expected U8 payload, got float32`. Whether the flag should be unnecessary for `qwen35moe` was not looked at **[?]**.

Not run: `tests/` (no `pytest` in `.venv-win`), a Linux check of the two edits.
