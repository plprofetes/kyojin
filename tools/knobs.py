#!/usr/bin/env python3
"""Inventory of the engine's switches: environment variables and compile-time build flags.

  python tools/knobs.py scan  [--json OUT]       inventory as JSON (stdout or OUT)
  python tools/knobs.py doc   [--out doc/knobs.md]   regenerate the generated reference page
  python tools/knobs.py check                      CI gate: every switch read in code is mentioned in doc/env_vars.md,
                                                   or listed in doc/knobs_undocumented.txt (switches that predate the gate)
  python tools/knobs.py check --update-baseline    rewrite that list (only ever shrink it by documenting; see doc/knobs.md)

Deliberately naive (stdlib, regex, no imports of the engine):
  - a switch is a quoted string literal with a known prefix (EXL3_, QWSPEC_, QSERVE_, MPW_, KYO_) anywhere in the scanned
    code, or a build flag `#if !defined(X)` / `#ifndef X` / `defined(X)` with prefix EXL3_ or QWG_ in the C++/HIP sources;
  - "documented" means the name appears as a whole word anywhere in doc/env_vars.md.
A false positive (a literal that is not really a switch) is fixed by documenting it or by adding it to the baseline with a reason.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_DOC = REPO / "doc" / "env_vars.md"
BASELINE = REPO / "doc" / "knobs_undocumented.txt"
GEN_DOC = REPO / "doc" / "knobs.md"
SERVE = REPO / "tools" / "qwen" / "serve.py"

PY_GLOBS = ("exllamav3/**/*.py", "tools/qwen/*.py", "setup.py")
C_GLOBS = ("exllamav3/exllamav3_ext/**/*.cu", "exllamav3/exllamav3_ext/**/*.cuh", "exllamav3/exllamav3_ext/**/*.cpp",
           "exllamav3/exllamav3_ext/**/*.h")
SKIP_PARTS = ("/build/", "_hip", ".hip")
ENV_PREFIX = r"(?:EXL3|QWSPEC|QSERVE|MPW|KYO)_[A-Z0-9_]+"
BUILD_PREFIX = r"(?:EXL3|QWG)_[A-Z0-9_]+"
RE_LIT = re.compile(r"""["'](%s)["']""" % ENV_PREFIX)
RE_COND = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif)\b(.*)$", re.M)
RE_BUILD_NAME = re.compile(r"\b(%s)\b" % BUILD_PREFIX)
RE_DEFINE = re.compile(r"^\s*#\s*define\s+(%s)\b" % BUILD_PREFIX, re.M)
RE_GETENV = re.compile(r"""(?:getenv|env_int|env_flag|env_bool|env_float|environ\.get|environ\.setdefault)\s*\(\s*["'](%s)["']"""
                       r"""(?:\s*,\s*([^)\n]{1,60}))?""" % ENV_PREFIX)
RE_SUBSCRIPT = re.compile(r"""environ\s*\[\s*["'](%s)["']\s*\]""" % ENV_PREFIX)
RE_DEFINE_DEFAULT = re.compile(r"^\s*#\s*define\s+(%s)\s+(\S+)" % BUILD_PREFIX, re.M)

# Code areas, for grouping in the generated page (first match on the path wins)
AREAS = (
    ("quant/exl3_dec", "decode GEMV (exl3_dec)"), ("exllamav3_ext/quant", "quantized kernels"),
    ("exllamav3_ext/", "other kernels (C++/HIP)"), ("generator/", "generator / scheduling"),
    ("modules/quant", "quantized linear"), ("modules/", "modules"), ("model/", "model / loader"),
    ("tools/qwen", "Qwen server"), ("util/", "utilities"), ("", "other"),
)


def _files(globs):
    out = []
    for g in globs:
        for p in REPO.glob(g):
            s = p.as_posix()
            if p.is_file() and not any(x in s for x in SKIP_PARTS):
                out.append(p)
    return sorted(set(out))


def _comment_near(lines, i, name):
    """Nearest comment text on the line itself or in the comment block just above (max 4 lines), stripped of markers."""
    def text(l):
        m = re.search(r"(#|//)\s?(.*)$", l)
        return m.group(2).strip() if m and not l.strip().startswith(("#include", "#if", "#endif", "#define", "#else", "#elif", "#pragma")) else ""
    own = text(lines[i])
    above = []
    j = i - 1
    while j >= 0 and len(above) < 4:
        t = lines[j].strip()
        if t.startswith(("#", "//")) and not t.startswith(("#include", "#if", "#endif", "#define", "#else", "#elif", "#pragma")):
            above.insert(0, text(lines[j]))
            j -= 1
        else:
            break
    s = " ".join(x for x in above + [own] if x)
    return re.sub(r"\s+", " ", s)[:300]


def _area(sites) -> str:
    """Area of the first engine site (exllamav3/...), else of the first site."""
    paths = [x.rsplit(":", 1)[0] for x in sites]
    p = next((x for x in paths if x.startswith("exllamav3/")), paths[0] if paths else "")
    return next(a for pat, a in AREAS if pat in p)


def _internal_build_constants() -> set[str]:
    """Names #defined somewhere WITHOUT an `#if !defined(NAME)` / `#ifndef NAME` guard on the line before: constants and
    header guards, not switches. A name that is only ever tested (never defined) is an opt-in flag (set by -D)."""
    out = set()
    for p in _files(C_GLOBS):
        try:
            lines = p.read_text(encoding="utf8", errors="replace").splitlines()
        except OSError:
            continue
        prev = ""
        for l in lines:
            m = RE_DEFINE.match(l)
            if m:
                name = m.group(1)
                guarded = re.match(r"\s*#\s*(?:(?:el)?if\s+!\s*defined\s*\(?\s*%s\b|ifndef\s+%s\b)" % (name, name), prev)
                if not guarded:
                    out.add(name)
            if l.strip():
                prev = l
    return out


def _serve_env():
    """SERVE_ENV of tools/qwen/serve.py as {name: value} (parsed, not imported)."""
    import ast
    try:
        tree = ast.parse(SERVE.read_text(encoding="utf8"))
    except (OSError, SyntaxError):
        return {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SERVE_ENV" for t in n.targets):
            try:
                return {k: v for k, v in ast.literal_eval(n.value)}
            except ValueError:
                return {}
    return {}


def scan() -> dict:
    knobs: dict[str, dict] = {}

    def k(name, kind):
        d = knobs.setdefault(name, {"name": name, "kinds": set(), "sites": [], "defaults": Counter(), "comment": "", "area": ""})
        d["kinds"].add(kind)
        return d

    internal_consts = _internal_build_constants()
    for p in _files(PY_GLOBS) + _files(C_GLOBS):
        rel = p.relative_to(REPO).as_posix()
        is_c = p.suffix in (".cu", ".cuh", ".cpp", ".h")
        try:
            src = p.read_text(encoding="utf8", errors="replace")
        except OSError:
            continue
        lines = src.splitlines()
        reads = {}
        for m in RE_GETENV.finditer(src):
            reads.setdefault(m.group(1), []).append((m.group(2) or "").strip().rstrip(",").strip())
        for m in RE_SUBSCRIPT.finditer(src):
            reads.setdefault(m.group(1), [])
        for i, line in enumerate(lines):
            for m in RE_LIT.finditer(line):
                name = m.group(1)
                d = k(name, "env (C++)" if is_c else "env (Python)")
                d["sites"].append(f"{rel}:{i + 1}")
                if not d["comment"]:
                    d["comment"] = _comment_near(lines, i, name)
                if not d["area"]:
                    d["area"] = next(a for pat, a in AREAS if pat in rel)
        for name, dflts in reads.items():
            if name in knobs:
                for x in dflts:
                    knobs[name]["defaults"][x or "<unset>"] += 1
        if is_c:
            defaults = {m.group(1): m.group(2) for m in RE_DEFINE_DEFAULT.finditer(src)}
            for m in RE_COND.finditer(src):
              for nm in RE_BUILD_NAME.finditer(m.group(1)):
                name = nm.group(1)
                if name in internal_consts:
                    continue
                i = src.count("\n", 0, m.start())
                d = k(name, "build flag")
                d["sites"].append(f"{rel}:{i + 1}")
                if name in defaults:
                    d["defaults"][defaults[name]] += 1
                if not d["comment"]:
                    d["comment"] = _comment_near(lines, i, name)
                if not d["area"]:
                    d["area"] = next(a for pat, a in AREAS if pat in rel)

    doc = ENV_DOC.read_text(encoding="utf8") if ENV_DOC.is_file() else ""
    headed = set(re.findall(r"^#+ .*?`(%s)`" % ENV_PREFIX, doc, re.M)) | set(re.findall(r"`(%s)`" % ENV_PREFIX, "\n".join(
        l for l in doc.splitlines() if l.startswith("#"))))
    serve_env = _serve_env()
    out = {}
    for name, d in sorted(knobs.items()):
        out[name] = {
            "name": name,
            "kinds": sorted(d["kinds"]),
            "area": _area(d["sites"]),
            "sites": d["sites"][:12],
            "n_sites": len(d["sites"]),
            "defaults": dict(d["defaults"].most_common()),
            "serve_env": serve_env.get(name),
            "documented": bool(re.search(r"\b%s\b" % re.escape(name), doc)),
            "doc_heading": name in headed,
            "comment": d["comment"],
        }
    return out


def read_baseline() -> set[str]:
    if not BASELINE.is_file():
        return set()
    return {l.split("#")[0].strip() for l in BASELINE.read_text(encoding="utf8").splitlines() if l.split("#")[0].strip()}


def cmd_check(args) -> int:
    inv = scan()
    undocumented = {n for n, d in inv.items() if not d["documented"]}
    if args.update_baseline:
        BASELINE.write_text("# Switches read in code but not mentioned in doc/env_vars.md when the knobs gate was added.\n"
                            "# Generated by `python tools/knobs.py check --update-baseline`. Shrink it by documenting; never add to it\n"
                            "# by hand except for a false positive (write the reason after a #).\n"
                            + "".join(f"{n}\n" for n in sorted(undocumented)), encoding="utf8")
        print(f"baseline: {len(undocumented)} undocumented switches written to {BASELINE.relative_to(REPO)}")
        return 0
    base = read_baseline()
    new = sorted(undocumented - base)
    stale = sorted(base & {n for n, d in inv.items() if d["documented"]})
    print(f"knobs: {len(inv)} switches, {len(inv) - len(undocumented)} mentioned in doc/env_vars.md, "
          f"{len(undocumented & base)} grandfathered, {len(new)} new and undocumented")
    if stale:
        print(f"note: {len(stale)} baseline entries are documented now; run `python tools/knobs.py check --update-baseline` "
              f"to drop them: {', '.join(stale[:10])}{' ...' if len(stale) > 10 else ''}")
    if new:
        print("\nFAIL: switches read in code but not mentioned in doc/env_vars.md:")
        for n in new:
            d = inv[n]
            print(f"  {n}  ({', '.join(d['kinds'])}; {d['sites'][0]})")
        print("\nFix: add a `### `NAME` (default: ...)` entry to doc/env_vars.md saying what it does, its default and when to "
              "change it. Then run: python tools/knobs.py doc")
        return 1
    return 0


def _md_cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def cmd_doc(args) -> int:
    inv = scan()
    by_area = defaultdict(list)
    for d in inv.values():
        by_area[d["area"] or "other"].append(d)
    n_doc = sum(d["documented"] for d in inv.values())
    lines = [
        "# Switch inventory (generated)",
        "",
        "Generated by `python tools/knobs.py doc` from the source tree; do not edit by hand. Hand-written explanations live in "
        "[env_vars.md](env_vars.md); this page lists **every** environment variable and build flag the code reads, where, its "
        "default as written at the read site, whether the Qwen server sets it (`SERVE_ENV` in `tools/qwen/serve.py`), and the "
        "nearest code comment as a first-draft description.",
        "",
        f"- {len(inv)} switches; {n_doc} mentioned in env_vars.md; "
        f"{sum(1 for d in inv.values() if d['serve_env'] is not None)} set by the Qwen server; "
        f"{sum(1 for d in inv.values() if 'build flag' in d['kinds'])} build flags (set through `EXL3_HIP_DEFINES` at build time).",
        "- CI (`.github/workflows/knobs.yml`) fails when a new switch is not mentioned in env_vars.md "
        "(`python tools/knobs.py check`). Switches that predate the gate are listed in "
        "[knobs_undocumented.txt](knobs_undocumented.txt).",
        "- The project's git history starts at a squashed snapshot, so commit messages do not explain the older switches; the "
        "comment column is the best source until a switch is written up in env_vars.md.",
        "- Measured effect of switches on real models: [benchmarks/sweeps/](benchmarks/sweeps/README.md).",
        "",
    ]
    for area in sorted(by_area):
        lines += [f"## {area}", "", "| switch | kind | default at read site | server | doc | first site | comment |", "|---|---|---|---|---|---|---|"]
        for d in sorted(by_area[area], key=lambda x: x["name"]):
            dfl = ", ".join(f"`{k}`" for k in list(d["defaults"])[:3]) or ""
            lines.append("| `{}` | {} | {} | {} | {} | `{}` | {} |".format(
                d["name"], ", ".join(d["kinds"]), _md_cell(dfl), "" if d["serve_env"] is None else f"`{d['serve_env']}`",
                "yes" if d["doc_heading"] else ("mention" if d["documented"] else "**no**"), d["sites"][0] if d["sites"] else "",
                _md_cell(d["comment"])))
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf8")
    print(f"wrote {args.out}: {len(inv)} switches")
    return 0


def cmd_scan(args) -> int:
    s = json.dumps(scan(), indent=1)
    if args.json:
        Path(args.json).write_text(s, encoding="utf8")
    else:
        print(s)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--json")
    d = sub.add_parser("doc")
    d.add_argument("--out", default=str(GEN_DOC))
    c = sub.add_parser("check")
    c.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args(argv)
    return {"scan": cmd_scan, "doc": cmd_doc, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
