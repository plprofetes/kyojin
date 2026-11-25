# Qwen3.5-35B-A3B on Windows (gfx1151): EXL3 4.00bpw vs GGUF UD-Q4_K_XL, upstream 1.9.3+44

Date 2026-11-25. Build: branch `rebase/upstream-main-1.9.3-44` (upstream `116a883`, 1.9.3 plus 44 commits) with two uncommitted edits for the GGUF path (see "Changes needed"). Machine: Strix Halo gfx1151, dynamic GPU memory mode, one server at a time.

## Result

`tools/bench.py --base http://127.0.0.1:8000/v1 --corpus doc/knobs.md --reps 3`, median of 3, temperature 0, drafting off, one server start per format **[V]**.

| | EXL3 `4.00bpw` | GGUF `UD-Q4_K_XL` | GGUF vs EXL3 |
|---|---|---|---|
| Weights on disk | 18.1 GiB | 20.71 GiB | +14 % |
| Prefill, ~2680 tokens | 2727.8 tok/s | 1735.8 tok/s | -36 % |
| Decode, prose, 128 tokens | 43.2 tok/s | 42.8 tok/s | -1 % |
| Decode, chat | 43.3 tok/s | 42.7 tok/s | -1 % |
| Decode, code | 43.1 tok/s | 42.6 tok/s | -1 % |
| Start-up (server process to ready) | not timed | about 30 s (model load 15 s) | |

- Decode is the same within 1 % in this run, although the GGUF reads 14 % more bytes per token. Why is not determined; a shared limit other than weight bandwidth is possible **[?]**.
- GGUF prefill is a third slower. The two runs do not share a prefill chunk: `serve.py` sets `EXL3_PREFILL_CHUNK=8192` for GGUF directories and keeps 4096 for EXL3 (its comment says 8192 helps GGUF and hurts EXL3), so this is each format at its default, not a controlled comparison **[V]**.
- Not compared: output quality (only that the GGUF answers coherently, thinking on), MTP / mix drafting (the GGUF has no MTP head here), long context, cold cache, vision, repeat runs.

## How each was served

| | EXL3 | GGUF |
|---|---|---|
| Source | `turboderp/Qwen3.5-35B-A3B-exl3` `4.00bpw`, `C:\models\qwen35-35b-a3b-4bpw` | `unsloth/Qwen3.5-35B-A3B-GGUF` `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`, `C:\models\qwen35-35b-a3b-ud-q4kxl-gguf` (22 241 950 336 bytes, equal to the Hub listing) |
| Command | `kyojin_win.py serve --model <dir> --port 8000 --ctx 8192` | `EXL3_GGUF_GENERIC=1 kyojin_win.py serve --model <dir> --port 8000 --ctx 16384` |
| Context | 8192 | 16384: at 8192 the warm-up job needs 33 pages and 32 exist, so the server exits |

GGUF directory (the loader wants a config directory, not a bare file): `loader/gguf_config.write_config_dir()` wrote `config.json`, `chat_template.jinja`, the tokenizer files and the `model.gguf` link from the GGUF header; `generation_config.json`, `preprocessor_config.json` and `video_preprocessor_config.json` were copied from the EXL3 pack (the model class opens the preprocessor file even with `--no-vision`). A copied EXL3 `config.json` does not work: the loader then looks for safetensors **[V]**.

## Changes needed to serve this GGUF on Windows (uncommitted)

1. `exllamav3/loader/gguf_collection.py`: `os.pread` does not exist on Windows (`AttributeError` while reading the embedding), and `os.open(..., O_RDONLY)` is text mode there. Now `O_BINARY` and a seek-and-read under a lock on `nt`; other platforms still use `os.pread` **[V]**.
2. `exllamav3/architecture/minicpm_v47.py`, `_fp_override`: this unsloth file stores `ssm_alpha` / `ssm_beta` (and most non-expert weights) as Q8_0, which the helper rejected (`AssertionError: blk.0.ssm_beta.weight: type 8`). Added a Q8_0 branch next to the existing embedding override **[V]**.
3. `EXL3_GGUF_GENERIC=1` is also needed: `Linear.load_gguf_` lets an in-memory override win over the stored blocks only under that flag, and without it the Q8_0 `in_proj_a/b` layers fail with `expected U8 payload, got float32`. Whether the flag should be unnecessary for `qwen35moe` was not looked at **[?]**.

Not run: `tests/` (no `pytest` in `.venv-win`), a Linux check of the two edits.
