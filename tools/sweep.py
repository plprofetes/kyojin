#!/usr/bin/env python3
"""Sweep runtime switches (env vars) and build flags against the real server on two standard queries.

Every measurement starts a fresh tools/qwen/serve.py with one arm's env on top of the server's own defaults (SERVE_ENV),
warms it up, sends a ~4K and a ~16K token prompt (greedy, fixed text), reads prefill/decode speed from the server's
`qserve: request` log line, keeps the reply text (to see if the arm changes the output), and stops the server.
The baseline (no override) runs at the start, middle and end of every round, so the noise is measured.
Results go to OUT/results.jsonl as they come in; running the same command again skips the runs that succeeded.

  .venv-win\\Scripts\\python.exe tools\\sweep.py run --model C:\\models\\qwen38-27b-4bpw --skip MOE
  .venv-win\\Scripts\\python.exe tools\\sweep.py run --model ... --dry-run      # list the arms only
  .venv-win\\Scripts\\python.exe tools\\sweep.py report logs\\sweep\\<dir> [--md doc\\benchmarks\\sweeps\\<model>.md]

Arms: each SERVE_ENV entry reverted to the engine default at its read site, a few exl3_dec switches, and build flags
(one `kyojin_win.py build --clean` each, ~12 min; the default build is restored at the end). --arms FILE replaces them:
[{"name": "x", "env": {"K": "V"}}, {"name": "y", "build": "EXL3_DEC_PF=1"}]. Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVE = REPO / "tools" / "qwen" / "serve.py"
WIN = os.name == "nt"
KNOB = re.compile(r"^(EXL3|QWSPEC|QSERVE|MPW|KYO)_")
KEEP = {"EXL3_ROCM_SDK", "EXL3_ROCM_DEV_INCLUDE", "EXL3_HIP_DEFINES"}   # plumbing, not switches
DEFAULT_DEFINES = "EXL3_HIP_STG_PAD"   # kyojin_win.py DEFAULT_HIP_DEFINES
VERIFY_ONLY = re.compile(r"VERIFY|GEMV_R_|MTP|DRAFT|QWSPEC|ROW_SPLIT|MOE_R8")   # no effect with --draft-policy off

EXTRA_ARMS = [   # exl3_dec (doc/windows/sprints/s6-scale-and-speed/plan-08-decode-gemv.md); KBS/MIN_BLOCKS are global
    {"name": "dec_min_blocks_16", "env": {"EXL3_DEC_MIN_BLOCKS": "16"}},
    {"name": "dec_min_blocks_64", "env": {"EXL3_DEC_MIN_BLOCKS": "64"}},
    {"name": "dec_min_blocks_192", "env": {"EXL3_DEC_MIN_BLOCKS": "192"}},
    {"name": "dec_kbs_4", "env": {"EXL3_DEC_KBS": "4"}},
    {"name": "dec_kbs_8", "env": {"EXL3_DEC_KBS": "8"}},
    {"name": "dec_kbs_10", "env": {"EXL3_DEC_KBS": "10"}},
    {"name": "dec_qkv_unfused", "env": {"EXL3_DEC_QKV": "0"}},
    {"name": "dec_off", "env": {"EXL3_DEC": "0"}},
    {"name": "build_dec_prod2", "build": "EXL3_DEC_PROD=2"},
    {"name": "build_dec_pf1", "build": "EXL3_DEC_PF=1"},
    {"name": "build_dec_prod2_pf1", "build": "EXL3_DEC_PROD=2 EXL3_DEC_PF=1"},
    {"name": "build_dec_sb2", "build": "EXL3_DEC_SB=2"},
    {"name": "build_dec_sb8", "build": "EXL3_DEC_SB=8"},
]


def log(out: Path, msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(out / "sweep.log", "a", encoding="utf8") as f:
        f.write(line + "\n")


def default_arms() -> list[dict]:
    """SERVE_ENV entries reverted to the literal default at their read site (tools/knobs.py), plus EXTRA_ARMS."""
    spec = importlib.util.spec_from_file_location("knobs", REPO / "tools" / "knobs.py")
    knobs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(knobs)
    inv = knobs.scan()
    arms = []
    for name, sval in knobs._serve_env().items():
        dflt = None
        for raw in inv.get(name, {}).get("defaults", {}):   # most common first; take the first plain literal
            m = re.fullmatch(r"""["']([^"']*)["']|(-?\d+)""", raw.strip())
            if m:
                dflt = m.group(1) if m.group(1) is not None else m.group(2)
                break
        if dflt is not None and dflt != sval:
            arms.append({"name": f"revert_{name}", "env": {name: dflt}})
    return arms + EXTRA_ARMS


def prompt(txt: str, depth: int, salt: int) -> str:
    """~depth tokens of real text at an offset fixed by (depth, salt): the same prompt in every run, another one for warm-up."""
    n = int(depth * 4.2)
    off = (104729 * (salt + 1) + 7919 * depth) % (len(txt) - n - 1)
    return txt[off:off + n] + "\n\nWrite a long, detailed continuation of the text above (at least 300 words)."


def post(url: str, body: dict | None = None, timeout: float = 3600):
    req = urllib.request.Request(url, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf8"))


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()


def measure(a, out: Path, env_over: dict, tag: str, txt: str) -> dict:
    """One fresh server: load, warm up every depth, measure every depth, stop."""
    env = {k: v for k, v in os.environ.items() if not KNOB.match(k) or k in KEEP}
    env.update(PYTHONPATH=str(REPO), PYTHONUTF8="1", QSERVE_LOG_VRAM="1", **env_over)
    logf = out / "server" / f"{tag.replace('|', '_')}.log"
    logf.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, "-u", str(SERVE), "--model", a.model, "--port", str(a.port),
           "-c", str(a.ctx), "--no-vision", "--draft-policy", a.draft_policy]
    rec = {"tag": tag, "env": env_over, "status": "ok", "requests": []}
    t0 = time.time()
    with open(logf, "w", encoding="utf8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{a.port}"
        while True:
            if proc.poll() is not None or time.time() - t0 > 1200:
                raise RuntimeError(f"server did not come up (exit {proc.poll()})")
            try:
                model_id = post(url + "/v1/models", timeout=10)["data"][0]["id"]
                post(url + "/health", timeout=10)
                break
            except Exception:  # noqa: BLE001 - still loading
                time.sleep(3)
        rec["load_s"] = round(time.time() - t0, 1)
        for salt, max_tokens in ((1, 8), (0, a.max_tokens)):   # warm-up pass (first use of a size tunes), then the measured pass
            for d in a.depths:
                size = logf.stat().st_size
                r = post(url + "/v1/chat/completions", {"model": model_id, "max_tokens": max_tokens, "temperature": 0,
                         "enable_thinking": False, "messages": [{"role": "user", "content": prompt(txt, d, salt)}]})
                if salt:
                    continue
                for _ in range(20):   # the stats line follows the reply
                    time.sleep(0.5)
                    with open(logf, encoding="utf8", errors="replace") as f:
                        f.seek(size)
                        lines = [l for l in f.read().splitlines() if "qserve: request" in l]
                    if lines:
                        break
                line = lines[-1]
                s = dict(re.findall(r"(\w+)=([\d.]+)", line))
                text = r["choices"][0]["message"].get("content") or ""
                rec["requests"].append({
                    "depth": d, "prompt": int(s["prompt"]), "cached": int(s["cached"]), "new": int(s["new"]),
                    "prefill_tps": round(int(s["prompt"]) / float(s["prefill_s"]), 1) if float(s["prefill_s"]) else None,
                    "decode_tps": round(int(s["new"]) / float(s["generate_s"]), 2) if float(s["generate_s"]) else None,
                    "sha1": hashlib.sha1(text.encode()).hexdigest(), "text": text})
    except Exception as e:  # noqa: BLE001 - recorded, the sweep goes on
        rec.update(status="failed", error=f"{type(e).__name__}: {e}",
                   log_tail=logf.read_text(encoding="utf8", errors="replace").splitlines()[-20:])
    finally:
        stop(proc)
        m = re.findall(r"vram .*?peak=(\d+)", logf.read_text(encoding="utf8", errors="replace"))
        rec["vram_peak_mib"] = int(m[-1]) if m else None
        rec["wall_s"] = round(time.time() - t0, 1)
        time.sleep(5)
    return rec


def build(out: Path, defines: str) -> bool:
    """Rebuild the extension with EXL3_HIP_DEFINES=defines (Windows: kyojin_win.py build --clean). A marker file says a
    non-default build is installed, so an interrupted sweep restores the default on its next start."""
    marker = out / "NON_DEFAULT_BUILD"
    if defines != DEFAULT_DEFINES:
        marker.write_text(defines, encoding="utf8")
    log(out, f"build EXL3_HIP_DEFINES={defines!r} (about 12 min)")
    logf = out / "builds" / f"{int(time.time())}.log"
    logf.parent.mkdir(exist_ok=True)
    with open(logf, "w", encoding="utf8", errors="replace") as lf:
        rc = subprocess.run([sys.executable, str(REPO / "scripts" / "windows" / "kyojin_win.py"), "build", "--clean"],
                            cwd=REPO, env={**os.environ, "EXL3_HIP_DEFINES": defines}, stdout=lf, stderr=subprocess.STDOUT).returncode
    log(out, f"build {'ok' if rc == 0 else f'FAILED (exit {rc}), see {logf}'}")
    if rc == 0 and defines == DEFAULT_DEFINES and marker.exists():
        marker.unlink()
    return rc == 0


def cmd_run(a) -> int:
    a.depths = [int(x) for x in a.depths.split(",")]
    a.ctx = a.ctx or (max(a.depths) + a.max_tokens + 2048 + 1023) // 1024 * 1024
    out = Path(a.out or REPO / "logs" / "sweep" / f"{dt.date.today():%Y%m%d}-{Path(a.model).name}-{a.draft_policy}")
    out.mkdir(parents=True, exist_ok=True)
    arms = json.loads(Path(a.arms).read_text(encoding="utf8")) if a.arms else default_arms()
    if a.skip:
        arms = [x for x in arms if not re.search(a.skip, x["name"])]
    if a.draft_policy == "off":
        arms = [x for x in arms if not VERIFY_ONLY.search(x["name"])]
    if not WIN:
        arms = [x for x in arms if "build" not in x]   # the build step is the Windows one
    for x in arms:
        print(f"  {x['name']:34} {x.get('build') or x.get('env')}")
    if a.dry_run:
        return 0
    meta = out / "meta.json"
    if not meta.exists():
        git = lambda *c: subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True).stdout.strip()   # noqa: E731
        meta.write_text(json.dumps({"model": a.model, "date": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "commit": git("rev-parse", "HEAD"),
                                    "dirty": len(git("status", "--porcelain").splitlines()), "depths": a.depths,
                                    "max_tokens": a.max_tokens, "ctx": a.ctx, "draft_policy": a.draft_policy,
                                    "rounds": a.rounds, "defines": DEFAULT_DEFINES}, indent=1), encoding="utf8")
    res = out / "results.jsonl"
    done = {r["tag"] for r in map(json.loads, res.read_text(encoding="utf8").splitlines()) if r["status"] == "ok"} if res.exists() else set()   # failed runs are retried
    txt = "".join((REPO / "exllamav3" / "conversion" / "standard_cal_data" / n).read_text(encoding="utf8")
                  for n in ("wiki.utf8", "c4.utf8", "technical.utf8", "code.utf8"))
    deadline = time.time() + a.max_hours * 3600
    fails = 0
    if (out / "NON_DEFAULT_BUILD").exists() and not build(out, DEFAULT_DEFINES):
        log(out, "STOP: could not restore the default build")
        return 2

    def run(name: str, env_over: dict, rnd: int, pos: int, defines: str = "") -> bool:
        nonlocal fails
        tag = f"{name}|r{rnd}|p{pos}"
        if tag in done:
            return True
        if time.time() > deadline or fails >= 3:
            log(out, "STOP: " + ("deadline" if fails < 3 else "3 failed runs in a row (driver/TDR? see server/*.log)")
                + "; run the same command again to continue")
            return False
        log(out, f"r{rnd} {name} {env_over}")
        rec = measure(a, out, env_over, tag, txt)
        rec.update(arm=name, round=rnd, build=defines)
        fails = fails + 1 if rec["status"] != "ok" else 0
        with open(res, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
        log(out, f"  {rec['status']} {rec.get('error', '')} " + " ".join(f"{q['depth']}: {q['prefill_tps']}/{q['decode_tps']}" for q in rec["requests"]))
        return True

    env_arms = [x for x in arms if "build" not in x]
    ok = True
    for rnd in range(1, a.rounds + 1):   # shuffled per round; baseline first, middle, last
        order = env_arms[:]
        random.Random(rnd).shuffle(order)
        h = len(order) // 2
        seq = [{"name": "baseline", "env": {}}] + order[:h] + [{"name": "baseline", "env": {}}] + order[h:] + [{"name": "baseline", "env": {}}]
        for pos, x in enumerate(seq):
            ok = ok and run(x["name"], x["env"], rnd, pos)
    try:
        for x in [x for x in arms if "build" in x] if ok else []:
            if all(f"{x['name']}|r{r}|p0" in done for r in range(1, a.rounds + 1)):
                continue
            if time.time() > deadline - 1800 or not build(out, f"{DEFAULT_DEFINES} {x['build']}"):
                continue
            for rnd in range(1, a.rounds + 1):
                run(x["name"], {}, rnd, 0, x["build"])
    finally:
        if (out / "NON_DEFAULT_BUILD").exists() and not build(out, DEFAULT_DEFINES):
            log(out, "WARNING: the default build is NOT installed; run kyojin_win.py build --clean with EXL3_HIP_DEFINES unset")
    report(out)
    log(out, f"done: {out / 'report.md'}")
    return 0


def report(out: Path, md: Path | None = None) -> None:
    meta = json.loads((out / "meta.json").read_text(encoding="utf8"))
    recs = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
    depths = meta["depths"]

    def vals(rs, d, m):
        return [q[m] for r in rs if r["status"] == "ok" for q in r["requests"] if q["depth"] == d and q[m]]

    base = [r for r in recs if r["arm"] == "baseline"]
    ref = {d: next((q["sha1"] for r in base if r["status"] == "ok" for q in r["requests"] if q["depth"] == d), None) for d in depths}
    det = all(q["sha1"] == ref[q["depth"]] for r in base if r["status"] == "ok" for q in r["requests"])
    bmed = {(d, m): statistics.median(vals(base, d, m)) if vals(base, d, m) else None for d in depths for m in ("prefill_tps", "decode_tps")}
    noise = {k: (100 * (max(vals(base, *k)) - min(vals(base, *k))) / 2 / v if v else 0) for k, v in bmed.items()}
    L = [f"# Switch sweep: {Path(meta['model']).name}, draft policy {meta['draft_policy']}", "",
         f"{meta['date']}, commit `{meta['commit'][:12]}` ({meta['dirty']} dirty paths), prompts ~{' / ~'.join(map(str, depths))} "
         f"tokens, greedy, {meta['max_tokens']} new tokens, fresh server per run, {meta['rounds']} rounds, default build "
         f"`EXL3_HIP_DEFINES={meta['defines']}`. Baseline = the server's own SERVE_ENV. Baseline replies "
         f"{'identical across runs' if det else '**differ across runs** (so `same` below is not meaningful)'}.", "",
         "| depth | baseline prefill tok/s | baseline decode tok/s | noise (half-range) prefill / decode |", "|---|---|---|---|"]
    for d in depths:
        L.append(f"| {d} | {bmed[(d, 'prefill_tps')]} | {bmed[(d, 'decode_tps')]} | ±{noise[(d, 'prefill_tps')]:.1f}% / ±{noise[(d, 'decode_tps')]:.1f}% |")
    L += ["", "Arms: % change of the median vs baseline (+ = faster); **bold** = outside the baseline noise and > 1 %; "
          "`same` = reply identical to the baseline.", "",
          "| arm | override | runs |" + "".join(f" prefill {d} | decode {d} | same {d} |" for d in depths),
          "|---|---|---|" + "---|---|---|" * len(depths)]
    rows = []
    for name in dict.fromkeys(r["arm"] for r in recs if r["arm"] != "baseline"):
        rs = [r for r in recs if r["arm"] == name]
        cells, worst = "", 0.0
        for d in depths:
            for m in ("prefill_tps", "decode_tps"):
                v = vals(rs, d, m)
                p = 100 * (statistics.median(v) - bmed[(d, m)]) / bmed[(d, m)] if v and bmed[(d, m)] else None
                worst = max(worst, abs(p or 0)) if m == "decode_tps" else worst
                cells += "  |" if p is None else (f" **{p:+.1f}** |" if abs(p) > max(noise[(d, m)], 1) else f" {p:+.1f} |")
            same = [q["sha1"] == ref[d] for r in rs if r["status"] == "ok" for q in r["requests"] if q["depth"] == d]
            cells += f" {'' if not same else 'yes' if all(same) else '**no**'} |"
        ok = sum(r["status"] == "ok" for r in rs)
        over = " ".join(f"`{k}={v}`" for k, v in rs[0]["env"].items()) or f"build `{rs[0].get('build', '')}`"
        rows.append((-worst, f"| {name} | {over} | {ok}/{len(rs)} |{cells}"))
    L += [r for _, r in sorted(rows)]
    L += ["", "Raw: `results.jsonl` (reply texts included), `server/*.log`, `builds/*.log`, `sweep.log`.", ""]
    (md or out / "report.md").write_text("\n".join(L), encoding="utf8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--out", help="output dir; reuse it to resume (default logs/sweep/<date>-<model>-<policy>)")
    r.add_argument("--depths", default="4096,16384")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--ctx", type=int, default=0)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--draft-policy", default="off", choices=("off", "mtp", "mix"))
    r.add_argument("--arms", help="JSON file replacing the default arms")
    r.add_argument("--skip", help="regex: skip arms whose name matches (e.g. MOE for a dense model, ^build to skip rebuilds)")
    r.add_argument("--max-hours", type=float, default=10)
    r.add_argument("--port", type=int, default=8011)
    r.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.add_argument("--md", help="also write the markdown here, e.g. doc/benchmarks/sweeps/<model>.md")
    a = ap.parse_args()
    if a.cmd == "report":
        report(Path(a.dir), Path(a.md) if a.md else None)
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
