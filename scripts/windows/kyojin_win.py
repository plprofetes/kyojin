#!/usr/bin/env python3
"""Windows 11 / gfx1151 build orchestrator for Kyojin (POC, happy path, verbose by design).

    uv run scripts/windows/kyojin_win.py doctor     # read-only preflight + probes
    uv run scripts/windows/kyojin_win.py setup      # venv + ROCm torch + triton-windows + rocm-sdk init
    uv run scripts/windows/kyojin_win.py build      # compile exllamav3_ext (editable install)
    uv run scripts/windows/kyojin_win.py check      # import / GPU / Triton / extension smoke tests
    uv run scripts/windows/kyojin_win.py serve --model <pack-dir>

Every run writes logs/windows/<timestamp>-<command>/ (gitignored): run.log (DEBUG, everything), one NN-<step>.log per
subprocess with its full combined output, env.json (redacted environment snapshot), report.json (every check with its
details). Attach that directory when something fails. Stdlib only; no state outside the repo except uv's own caches.

Prior art this file follows (ideas and documented command patterns only; no third-party code is copied):
  * AMD TheRock RELEASES.md (MIT): `pip install --index-url <index> "torch[device-gfx1151]"`, `rocm-sdk init|test|path`.
  * triton-lang/triton-windows (MIT-style): `pip install "triton-windows<3.9"`, clang-cl found from the ROCm wheel.
  * Kyojin tools/strix_halo/env.sh and rebuild.sh (MIT): build-time ROCm variables, EXL3_HIP_DEFINES default.
  * Microsoft vswhere / VsDevCmd.bat documented usage for locating MSVC and capturing its environment.
See doc/windows/assessment.md section 7c for the licence notes.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import datetime as dt
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV = REPO / ".venv-win"
LOG_ROOT = REPO / "logs" / "windows"

# --- pins: one place. Verified on the AMD index from a Linux box (wheel METADATA read over HTTP range requests):
# torch 2.15.0a0+rocm10.1.0a20260822 (cp310..cp314 win_amd64) requires rocm[libraries]==10.1.0a20260822 and, via the
# device-gfx1151 extra, amd-torch-device-gfx1151/gfx115x == the same version. rocm-sdk-core/devel win_amd64 wheels exist
# at 10.1.0a20260822. NOT yet verified on a real Windows box: that is what `doctor`/`setup`/`check` are for.
AMD_INDEX = os.environ.get("KYOJIN_ROCM_INDEX", "https://rocm.nightlies.amd.com/whl-multi-arch/")
PYPI_INDEX = "https://pypi.org/simple"
ROCM_VER = os.environ.get("KYOJIN_ROCM_VER", "10.1.0a20260822")
TORCH_VER = os.environ.get("KYOJIN_TORCH_VER", "2.15.0a0+rocm10.1.0a20260822")
TRITON_SPEC = os.environ.get("KYOJIN_TRITON_SPEC", "triton-windows<3.9")   # Kyojin's Linux pin is triton 3.8.0
GFX = "gfx1151"
PY_VER = "3.12"
MIN_FREE_GB = 40          # my estimate (SDK devel ~12 GB on Linux + torch + build tree); tune after the first run
GB_PER_JOB = 4            # my estimate for hipcc; Kyojin's Linux default is MAX_JOBS=6 ("unbounded ninja OOMs the box")
DEFAULT_HIP_DEFINES = "EXL3_HIP_STG_PAD"   # same default as tools/strix_halo/rebuild.sh

log = logging.getLogger("kyojin_win")

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"


# ----------------------------------------------------------------------------------------------------------------------
# logging / reporting
# ----------------------------------------------------------------------------------------------------------------------

SECRET_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|CREDENTIAL|PAT$|^HF_)", re.I)


def redact_env(env: dict[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if SECRET_RE.search(k) else v) for k, v in env.items()}


NTSTATUS = {
    0xC0000005: "ACCESS_VIOLATION (native crash)",
    0xC0000135: "DLL_NOT_FOUND (a required DLL is missing from PATH)",
    0xC0000139: "ENTRYPOINT_NOT_FOUND (DLL version mismatch)",
    0xC000001D: "ILLEGAL_INSTRUCTION",
    0xC0000409: "STACK_BUFFER_OVERRUN / fail-fast",
    0xC0000374: "HEAP_CORRUPTION",
    0xC00000FD: "STACK_OVERFLOW",
}


def describe_exit(code: int | None) -> str:
    if code is None:
        return "no exit code (timeout or not started)"
    u = code & 0xFFFFFFFF
    name = NTSTATUS.get(u)
    return f"{code} (0x{u:08X}{', ' + name if name else ''})"


@dataclasses.dataclass
class Check:
    name: str
    status: str
    summary: str = ""
    details: dict = dataclasses.field(default_factory=dict)
    hint: str = ""


class Run:
    """One invocation: log directory, handlers, step counter, collected checks."""

    def __init__(self, command: str, verbose: bool, root: Path = LOG_ROOT):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"{stamp}-{command}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.command = command
        self.verbose = verbose
        self.step = 0
        self.checks: list[Check] = []
        log.handlers.clear()
        log.setLevel(logging.DEBUG)
        log.propagate = False
        fh = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s", "%H:%M:%S"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(fh)
        log.addHandler(ch)
        self._fh, self._ch = fh, ch
        log.info("kyojin_win %s -> %s", command, self.dir)
        self.snapshot_env()

    def close(self):
        for h in (self._fh, self._ch):
            h.flush()
            log.removeHandler(h)
        self._fh.close()

    def snapshot_env(self, env: dict[str, str] | None = None, name: str = "env.json"):
        env = dict(os.environ if env is None else env)
        def git(*args):
            try:
                return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=20).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return "unknown"
        sha, dirty = git("rev-parse", "HEAD"), len(git("status", "--porcelain").splitlines())
        log.info("repo commit %s (%d uncommitted paths)", sha[:12] or "unknown", dirty)
        data = {
            "repo_commit": sha, "repo_uncommitted_paths": dirty,
            "pins": {"amd_index": AMD_INDEX, "rocm": ROCM_VER, "torch": TORCH_VER, "triton": TRITON_SPEC, "python": PY_VER},
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "env": redact_env(env),
            "PATH_entries": env.get("PATH", env.get("Path", "")).split(os.pathsep),
        }
        (self.dir / name).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        lvl = {PASS: logging.INFO, INFO: logging.INFO, WARN: logging.WARNING, FAIL: logging.ERROR, SKIP: logging.INFO}[c.status]
        log.log(lvl, "[%-4s] %-22s %s", c.status, c.name, c.summary)
        log.debug("        details: %s", json.dumps(c.details, indent=2, default=str, sort_keys=True))
        if c.hint and c.status in (FAIL, WARN):
            log.log(lvl, "         -> %s", c.hint)
        return c

    def fail_step(self, r: "Result", title: str, cmd: list[str] | None = None, default_hint: str = "") -> Check:
        """A failed subprocess step: diagnose, extract root-cause lines, print and persist a shareable FAILURE.md entry."""
        text = r.out + "\n" + r.error
        known = diagnose(text)
        errors = extract_errors(r.out, first_line_no=2)       # step log line 1 is the "$ command" header
        block = render_failure(title, cmd or [r.name], r.code, r.seconds, r.logfile.name, known, errors, r.tail(12))
        for line in block.splitlines():
            log.error(line)
        with open(self.dir / "FAILURE.md", "a", encoding="utf-8") as f:
            f.write("```text\n" + block + "\n```\n\n")
        hint = (known[0].fix if known else default_hint) or f"see {r.logfile.name}"
        return Check(r.name, FAIL, f"{title}: {describe_exit(r.code)}" + (f" [{known[0].id}]" if known else ""),
                     {"log": r.logfile.name, "known": [k.id for k in known], "first_errors": errors, "tail": r.tail(20)}, hint)

    def write_report(self):
        data = [dataclasses.asdict(c) for c in self.checks]
        (self.dir / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def summary(self) -> int:
        self.write_report()
        counts = {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, FAIL, SKIP, INFO)}
        log.info("")
        log.info("summary: %s", "  ".join(f"{k}={v}" for k, v in counts.items()))
        fails = [c for c in self.checks if c.status == FAIL]
        for c in fails:
            log.info("  FAIL %s: %s", c.name, c.summary)
            if c.hint:
                log.info("       next: %s", c.hint)
        log.info("logs: %s  (attach this folder when reporting an issue)", self.dir)
        return 1 if fails else 0

    # -- subprocess -------------------------------------------------------------------------------------------------

    def run(self, cmd: list[str], name: str, env: dict[str, str] | None = None, cwd: Path | None = None,
            timeout: float | None = None, stream: bool | None = None) -> "Result":
        """Run cmd, tee combined stdout+stderr to NN-<name>.log (always) and the console (when streaming)."""
        self.step += 1
        logfile = self.dir / f"{self.step:02d}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.log"
        stream = self.verbose if stream is None else stream
        eff = dict(os.environ if env is None else env)
        delta = {k: v for k, v in eff.items() if os.environ.get(k) != v}
        log.debug("RUN %s: %s", name, " ".join(_q(c) for c in cmd))
        log.debug("    cwd=%s timeout=%s log=%s", cwd or os.getcwd(), timeout, logfile.name)
        if delta:
            log.debug("    env delta vs parent: %s", json.dumps(redact_env(delta), sort_keys=True))
        t0 = time.monotonic()
        out: list[str] = []
        code: int | None = None
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=eff, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            log.error("    could not start %s: %s", cmd[0], e)
            logfile.write_text(f"could not start: {e}\n", encoding="utf-8")
            return Result(name, None, "", logfile, 0.0, error=str(e))
        timer = None
        if timeout:
            timer = threading.Timer(timeout, proc.kill)
            timer.start()
        with open(logfile, "w", encoding="utf-8") as lf:
            lf.write(f"$ {' '.join(_q(c) for c in cmd)}\n")
            assert proc.stdout is not None
            for line in proc.stdout:
                out.append(line)
                lf.write(line)
                if stream:
                    log.debug("    | %s", line.rstrip("\n"))
            proc.stdout.close()
            code = proc.wait()
            if timer:
                timer.cancel()
            lf.write(f"\n[exit {describe_exit(code)} after {time.monotonic() - t0:.1f}s]\n")
        dur = time.monotonic() - t0
        log.debug("    exit=%s in %.1fs", describe_exit(code), dur)
        return Result(name, code, "".join(out), logfile, dur)


@dataclasses.dataclass
class Result:
    name: str
    code: int | None
    out: str
    logfile: Path
    seconds: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    def tail(self, n: int = 25) -> str:
        return "\n".join(self.out.splitlines()[-n:])


def _q(s: str) -> str:
    return f'"{s}"' if " " in s else s


# ----------------------------------------------------------------------------------------------------------------------
# pure helpers (unit-tested)
# ----------------------------------------------------------------------------------------------------------------------

def parse_set_output(text: str) -> dict[str, str]:
    """Parse `cmd /c set` output (NAME=value lines) into a dict; skips lines without '=' and the empty-name ones."""
    env = {}
    for line in text.splitlines():
        k, sep, v = line.partition("=")
        if sep and k and not k.startswith("="):
            env[k] = v
    return env


def env_diff(before: dict[str, str], after: dict[str, str]) -> dict[str, tuple]:
    ci_before = {k.upper(): (k, v) for k, v in before.items()}
    out = {}
    for k, v in after.items():
        old = ci_before.get(k.upper())
        if old is None:
            out[k] = (None, v)
        elif old[1] != v:
            out[k] = (old[1], v)
    return out


def pick_vs(vswhere_json: str) -> dict | None:
    """Newest VS instance that has the MSVC x64 tools (vswhere already filtered by -requires; take the first)."""
    try:
        data = json.loads(vswhere_json or "[]")
    except json.JSONDecodeError:
        return None
    return data[0] if data else None


def compute_jobs(cpu_count: int, avail_gb: float, requested: int | None = None) -> int:
    if requested:
        return max(1, requested)
    return max(1, min(6, cpu_count, int(avail_gb // GB_PER_JOB)))


def parse_index_wheels(html: str, package: str, version: str, py_tag: str = "cp312") -> list[str]:
    """Wheel file names for package==version, win_amd64, py_tag, from a simple-index HTML page."""
    names = re.findall(r'href="([^"#]+)', html)
    ver = version.replace("+", "%2B")
    out = []
    for n in names:
        base = n.rsplit("/", 1)[-1]
        b = base.replace("%2B", "+")
        if version in b and "win_amd64" in b and (py_tag in b or "py3-none" in b) and package.replace("-", "_") in b.lower().replace("-", "_"):
            out.append(b)
    return sorted(set(out))


def sdk_env(root: Path, base: dict[str, str]) -> dict[str, str]:
    """Build-time ROCm variables, mirroring tools/strix_halo/env.sh (`build` mode) for a Windows SDK layout."""
    env = dict(base)
    env["ROCM_HOME"] = env["ROCM_PATH"] = env["HIP_PATH"] = str(root)
    env["HIPCXX"] = str(root / "bin" / "hipcc.exe")
    sep = os.pathsep
    env["PATH"] = sep.join([str(root / "bin"), str(root / "lib" / "llvm" / "bin"), str(root / "llvm" / "bin"),
                            env.get("PATH", env.get("Path", ""))])
    for cand in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode"):
        if cand.is_dir():
            env["HIP_DEVICE_LIB_PATH"] = str(cand)
            break
    env["PYTORCH_ROCM_ARCH"] = GFX
    env["EXL3_ROCM_SDK"] = str(root)
    env.setdefault("EXL3_HIP_DEFINES", DEFAULT_HIP_DEFINES)
    return env


def serve_command(py: str, model: str, port: int, ctx: int) -> list[str]:
    return [py, "-u", str(REPO / "tools" / "qwen" / "serve.py"), "--model", model, "--port", str(port),
            "-c", str(ctx), "--no-vision", "--draft-policy", "off"]


# ----------------------------------------------------------------------------------------------------------------------
# helpful errors: failure signatures, first-error extraction, FAILURE.md
# ----------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Known:
    id: str
    pattern: str        # regex, searched case-insensitively over the failing step's output
    title: str
    cause: str
    fix: str


# HYPOTHESES from how these toolchains usually fail, not yet observed on this stack. When a real failure matches
# nothing here, add its signature after the first run (see doc/windows/deferred.md, "error catalogue").
KNOWN: tuple[Known, ...] = (
    Known("no-msvc", r"Microsoft Visual C\+\+ 1\d\.\d or greater is required|Unable to find vcvarsall|cl\.exe.*not (found|recognized)|'cl' is not recognized",
          "MSVC compiler not found",
          "Visual Studio Build Tools (C++ workload) is missing or its environment was not loaded.",
          "Install Build Tools with the 'Desktop development with C++' workload, then run `doctor` (visual_studio must PASS)."),
    Known("missing-win-headers", r"C1083: Cannot open include file: '(windows\.h|type_traits|stdio\.h|vcruntime\.h)'|'(windows\.h|vcruntime\.h)' file not found",
          "Windows SDK / MSVC headers not visible to the compiler",
          "INCLUDE/LIB are not set: the Visual Studio developer environment was not applied to this compile.",
          "Run via this script's `build` (it applies VsDevCmd.bat). Check build-env.json for INCLUDE and LIB. Install the Windows 11 SDK component if `doctor` shows none."),
    Known("missing-hip-headers", r"fatal error: '(hip/[^']+|hipblas/[^']+|hipsparse/[^']+|thrust/[^']+|rocblas/[^']+)' file not found",
          "ROCm SDK headers missing",
          "The rocm-sdk-devel tree is not expanded, or ROCM_HOME points at the wrong directory.",
          "Run `.venv-win\\Scripts\\rocm-sdk init`, then `doctor` (rocm_sdk details.headers must be all true)."),
    Known("missing-import-lib", r"LNK1181: cannot open input file '?(hipblas|amdhip64|c10_hip|torch_hip|torch_python)[^']*\.lib",
          "Import library not found at link time",
          "The linker cannot find a .lib: torch/ROCm library directories are not on LIB, or the SDK devel tree lacks lib/.",
          "Check build-env.json LIB; confirm `doctor` rocm_sdk root_listing has lib/. A missing hipblas.lib means the devel tree is incomplete: re-run `rocm-sdk init`."),
    Known("device-libs", r"cannot find ROCm device library|device-libs|amdgcn[\\/]bitcode",
          "ROCm device library (bitcode) not found",
          "clang cannot locate the amdgcn bitcode directory.",
          "Check HIP_DEVICE_LIB_PATH in build-env.json; it must be <sdk>\\lib\\llvm\\amdgcn\\bitcode (or <sdk>\\amdgcn\\bitcode)."),
    Known("torch-not-rocm", r"CUDA_HOME environment variable is not set|CUDAExtension.*CUDA_HOME",
          "torch does not see a ROCm toolchain",
          "Either torch is not a ROCm wheel, or ROCM_HOME/HIP_PATH was not set for the build.",
          "Run `doctor`: torch_gpu must show hip. If it does, the build env was not applied; use this script's `build`."),
    Known("build-isolation", r"No module named 'torch'|ModuleNotFoundError: No module named 'torch'",
          "build ran outside the venv (torch not importable)",
          "pip/uv built in an isolated environment without torch.",
          "Use `build` (it passes --no-build-isolation with the venv interpreter). Do not run pip install . by hand."),
    Known("path-too-long", r"path.{0,20}too long|filename or extension is too long|Filename too long|WinError 206|LNK1104: cannot open file '[^']{240,}",
          "Windows path length limit",
          "A generated object/hipify path exceeds 260 characters (long paths disabled or repo path too deep).",
          "Enable long paths (see `doctor` longpaths hint) or clone to a short path such as C:\\k."),
    Known("dll-missing", r"DLL load failed while importing|WinError 126|WinError 127|0xC0000135|The specified module could not be found",
          "A DLL could not be loaded",
          "A dependent DLL is not on PATH: usually the ROCm SDK bin directory, the VC++ runtime, or a mismatched extension build.",
          "Run `check` (it adds the SDK bin to PATH). If it still fails: `python -c \"import torch\"` inside .venv-win, and compare the DLL named in the log against `doctor` rocm_sdk bin_listing."),
    Known("entrypoint", r"0xC0000139|procedure entry point .* could not be located|ENTRYPOINT_NOT_FOUND",
          "DLL version mismatch",
          "An extension built against one torch/ROCm version is loading with another.",
          "Rebuild: `build --clean`. Do not mix wheels from different index dates (compare pip-freeze.txt)."),
    Known("no-matching-dist", r"No matching distribution found|Could not find a version that satisfies|no version of torch|not found in the package registry|is not available on",
          "A pinned package version is not on the index",
          "The AMD nightly was pruned or the pin does not exist for this Python.",
          "Run `doctor` (amd_index lists the wheels found). Pick an existing version and set KYOJIN_TORCH_VER / KYOJIN_ROCM_VER."),
    Known("network", r"Connection (reset|aborted|refused)|timed out|TimeoutError|SSL(Error|: CERTIFICATE)|Temporary failure in name resolution|getaddrinfo failed|Failed to download",
          "Network failure",
          "A download from the package index failed (proxy, VPN, firewall or transient).",
          "Retry. Behind a corporate proxy set HTTPS_PROXY and SSL_CERT_FILE. `doctor` amd_index tests reachability."),
    Known("oom", r"std::bad_alloc|LLVM ERROR: out of memory|out of memory|C1060|compiler is out of heap space|clang: error: unable to execute command: (Killed|Segmentation)|fatal error C1002",
          "Compiler ran out of memory",
          "Too many parallel hipcc jobs for the free RAM.",
          "Close other programs and re-run `build --jobs 2` (see cpu_ram in the build log for free RAM)."),
    Known("file-locked", r"PermissionError: \[WinError (5|32)\]|Access is denied|being used by another process",
          "File locked or access denied",
          "Another process (a running python, an editor, antivirus) holds a file in the repo or .venv-win.",
          "Stop python.exe processes started from this repo, close editors on it, add the repo and .venv-win to antivirus exclusions, retry."),
    Known("flag-leak", r"unknown argument:? '?-X(compiler|cudafe)|unknown argument: '-Xcompiler|clang: error: unknown argument",
          "An nvcc-only flag reached hipcc",
          "setup.py passed a CUDA/nvcc option into the HIP compile.",
          "Look at the failing command in the log (ninja prints it after FAILED:) and gate that flag in setup.py for torch.version.hip."),
    Known("msvc-flag-on-clang", r"cl : Command line (warning|error) D9002|D8021|D8004",
          "A GCC/clang flag reached cl.exe",
          "setup.py passed a non-MSVC option to the host C++ compiler.",
          "Gate that flag in setup.py for Windows (see how -Wno-register is gated)."),
    Known("python-h", r"Cannot open include file: 'Python\.h'|Python\.h: No such file",
          "Python headers missing",
          "The venv interpreter has no include/ directory (non-uv Python or Microsoft Store Python).",
          "Recreate the venv: `setup --recreate` (uv installs a full CPython 3.12)."),
    Known("invalid-device-function", r"invalid device function|hipErrorNoBinaryForGpu|no kernel image is available",
          "GPU code object does not match this GPU",
          "The extension or a JIT kernel was compiled for a different architecture than the installed GPU.",
          "Confirm `doctor` torch_gpu arch is gfx1151, then `build --clean`. PYTORCH_ROCM_ARCH must be gfx1151."),
    Known("triton-compiler", r"Failed to find C compiler|clang-cl.*(not found|No such)|Please specify one with the CC|CalledProcessError.*(clang-cl|cl\.exe)",
          "Triton cannot find a C compiler for its launcher stub",
          "triton-windows could not find clang-cl (from the ROCm SDK wheel) or MSVC cl.exe.",
          "Run Triton checks through `check` (applies SDK PATH). Otherwise put <sdk>\\lib\\llvm\\bin on PATH or set CC to clang-cl.exe."),
    Known("no-triton", r"No module named 'triton'",
          "triton-windows not installed",
          "setup did not complete, or the wrong interpreter ran.",
          "Run `setup`; verify `.venv-win\\Scripts\\python -c \"import triton\"`."),
)

ERROR_LINE = re.compile(r"(?i)(\berror\b|\bfatal\b|FAILED:|Traceback \(most recent|\bLNK\d{4}\b|\bC\d{4}:|cannot open|no such file|"
                        r"undefined (symbol|reference)|unresolved external|not found|Segmentation|access violation)")
ERROR_NOISE = re.compile(r"(?i)(\b0 errors?\b|-Werror|-Wno-error|error_code|warning:|errors?[- ]handling|stderr)")


def diagnose(text: str) -> list[Known]:
    """Known failure signatures present in text, in catalogue order."""
    return [k for k in KNOWN if re.search(k.pattern, text or "", re.I)]


def extract_errors(text: str, limit: int = 6, first_line_no: int = 1) -> list[tuple[int, str]]:
    """The FIRST error-looking lines (the root cause is almost never in the tail), with their line numbers."""
    out = []
    for i, line in enumerate((text or "").splitlines(), start=first_line_no):
        if ERROR_LINE.search(line) and not ERROR_NOISE.search(line):
            out.append((i, line.strip()[:300]))
            if len(out) >= limit:
                break
    return out


def render_failure(title: str, cmd: list[str], code: int | None, seconds: float, logname: str, known: list[Known],
                   errors: list[tuple[int, str]], tail: str, extra: dict | None = None) -> str:
    """Human-readable failure block (console + FAILURE.md)."""
    L = [f"FAILED: {title}", f"  command : {' '.join(_q(c) for c in cmd)}", f"  exit    : {describe_exit(code)} after {seconds:.0f}s",
         f"  log     : {logname}"]
    if known:
        for k in known[:3]:
            L += [f"  likely  : {k.title} [{k.id}]", f"    why   : {k.cause}", f"    fix   : {k.fix}"]
    else:
        L += ["  likely  : no known signature matched (new failure: please send the log folder)"]
    if errors:
        L.append(f"  first error lines (line numbers refer to {logname}):")
        L += [f"    L{n}: {t}" for n, t in errors]
    if tail.strip():
        L += ["  last output:"] + [f"    | {t}" for t in tail.splitlines()[-12:]]
    for k, v in (extra or {}).items():
        L.append(f"  {k}: {v}")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------------
# Windows probes (each returns a Check; never raises)
# ----------------------------------------------------------------------------------------------------------------------

def is_windows() -> bool:
    return sys.platform == "win32"


def _guard(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001  a probe must never take the run down; the exception IS the datum
            import traceback
            return Check(fn.__name__.removeprefix("probe_"), FAIL, f"probe crashed: {type(e).__name__}: {e}",
                         {"traceback": traceback.format_exc()}, "report this log; the probe itself needs fixing")
    wrapper.__name__ = fn.__name__
    return wrapper


@_guard
def probe_host(run: Run) -> Check:
    d = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
         "python_bits": 64 if sys.maxsize > 2**32 else 32, "executable": sys.executable, "uv_python": os.environ.get("UV_PYTHON")}
    if not is_windows():
        return Check("host", FAIL, f"not Windows ({sys.platform})", d, "run this on the Windows 11 machine")
    v = sys.getwindowsversion()
    d["windows_version"] = f"{v.major}.{v.minor}.{v.build}"
    d["windows_build"] = v.build
    ok = v.build >= 22000 and platform.machine().upper() in ("AMD64", "X86_64") and d["python_bits"] == 64
    return Check("host", PASS if ok else FAIL, f"Windows build {v.build}, {platform.machine()}, Python {d['python_bits']}-bit", d,
                 "" if ok else "need Windows 11 (build >= 22000), x64, 64-bit Python")


@_guard
def probe_cpu_ram(run: Run) -> Check:
    d: dict = {"logical_cpus": os.cpu_count()}
    if is_windows():
        import winreg

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))  # type: ignore[attr-defined]
        d.update(total_phys_gb=round(ms.ullTotalPhys / 2**30, 1), avail_phys_gb=round(ms.ullAvailPhys / 2**30, 1),
                 pagefile_total_gb=round(ms.ullTotalPageFile / 2**30, 1), memory_load_pct=ms.dwMemoryLoad)
        with contextlib.suppress(OSError):
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                d["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        d["jobs_suggested"] = compute_jobs(os.cpu_count() or 1, d["avail_phys_gb"])
        return Check("cpu_ram", INFO, f"{d.get('cpu', '?')}; RAM {d['total_phys_gb']} GB total, {d['avail_phys_gb']} GB free; "
                     f"suggested build jobs {d['jobs_suggested']}", d)
    return Check("cpu_ram", SKIP, "not Windows", d)


@_guard
def probe_gpu(run: Run) -> Check:
    """Display adapters from the registry display-class key (stdlib only). PowerShell CIM output is logged as a second source."""
    if not is_windows():
        return Check("gpu", SKIP, "not Windows")
    import winreg
    adapters = []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11cd-bfc1-08002be10318}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            with contextlib.suppress(OSError):
                with winreg.OpenKey(root, sub) as k:
                    row = {}
                    for val in ("DriverDesc", "DriverVersion", "DriverDate", "ProviderName", "MatchingDeviceId"):
                        with contextlib.suppress(OSError):
                            row[val] = winreg.QueryValueEx(k, val)[0]
                    with contextlib.suppress(OSError):
                        mem = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
                        row["dedicated_vram_gb"] = round(int.from_bytes(mem, "little") / 2**30, 2) if isinstance(mem, bytes) else round(mem / 2**30, 2)
                    if row:
                        adapters.append(row)
    cim = run.run(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM,PNPDeviceID,VideoProcessor | ConvertTo-Json"],
                  "gpu-cim", timeout=60, stream=False)
    d = {"adapters": adapters, "cim_log": cim.logfile.name}
    amd = [a for a in adapters if "AMD" in (a.get("ProviderName", "") + a.get("DriverDesc", "")) or "Radeon" in a.get("DriverDesc", "")]
    if not amd:
        return Check("gpu", FAIL, "no AMD/Radeon display adapter found", d, "install the latest AMD Adrenalin driver; see gpu-cim log")
    names = ", ".join(f"{a.get('DriverDesc')} (driver {a.get('DriverVersion')}, {a.get('DriverDate', '?')})" for a in amd)
    return Check("gpu", PASS, names, d, "")


@_guard
def probe_disk(run: Run) -> Check:
    u = shutil.disk_usage(REPO)
    free = u.free / 2**30
    d = {"repo": str(REPO), "free_gb": round(free, 1), "total_gb": round(u.total / 2**30, 1), "required_gb_estimate": MIN_FREE_GB,
         "repo_path_len": len(str(REPO))}
    ok = free >= MIN_FREE_GB
    return Check("disk", PASS if ok else FAIL, f"{free:.0f} GB free on the repo drive (need ~{MIN_FREE_GB})", d,
                 "" if ok else "free disk space or clone the repo on a bigger drive")


@_guard
def probe_longpaths(run: Run) -> Check:
    if not is_windows():
        return Check("longpaths", SKIP, "not Windows")
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
        v = winreg.QueryValueEx(k, "LongPathsEnabled")[0]
    d = {"LongPathsEnabled": v, "repo_path_len": len(str(REPO))}
    if v == 1:
        return Check("longpaths", PASS, "long paths enabled", d)
    return Check("longpaths", WARN, "long paths disabled (hipify/ninja object paths can exceed 260 chars)", d,
                 r'admin PowerShell: Set-ItemProperty HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled 1  (then reboot); '
                 "or clone to a short path like C:\\k")


@_guard
def probe_tool(run: Run, exe: str, args: list[str], hint: str, required: bool = True) -> Check:
    path = shutil.which(exe)
    d: dict = {"path": path}
    if not path:
        return Check(exe, FAIL if required else WARN, "not found on PATH", d, hint)
    r = run.run([path, *args], f"{exe}-version", timeout=60, stream=False)
    d["version_output"] = r.out.strip()[:300]
    return Check(exe, PASS if r.ok else WARN, (r.out.strip().splitlines() or ["?"])[0], d, "")


def vswhere_path() -> Path | None:
    pf86 = os.environ.get("ProgramFiles(x86)")
    if not pf86:
        return None
    p = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return p if p.is_file() else None


@_guard
def probe_vs(run: Run) -> Check:
    """MSVC x64 tools via vswhere (documented fixed location), plus the Windows SDK."""
    vw = vswhere_path()
    d: dict = {"vswhere": str(vw) if vw else None}
    hint = ("install 'Build Tools for Visual Studio 2022' with workloads 'Desktop development with C++' (MSVC x64 + Windows 11 SDK): "
            "winget install Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools "
            "--includeRecommended\"")
    if not vw:
        return Check("visual_studio", FAIL, "vswhere.exe not found (Visual Studio Installer absent)", d, hint)
    r = run.run([str(vw), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json"],
                "vswhere", timeout=60, stream=False)
    inst = pick_vs(r.out) if r.ok else None
    d["vswhere_exit"] = describe_exit(r.code)
    if not inst:
        return Check("visual_studio", FAIL, "no Visual Studio with MSVC x64 tools", d, hint)
    root = Path(inst["installationPath"])
    vsdev = root / "Common7" / "Tools" / "VsDevCmd.bat"
    sdk_dir = Path(os.environ.get("ProgramFiles(x86)", "")) / "Windows Kits" / "10" / "Include"
    sdks = sorted(p.name for p in sdk_dir.iterdir()) if sdk_dir.is_dir() else []
    d.update(installationPath=str(root), installationVersion=inst.get("installationVersion"), displayName=inst.get("displayName"),
             VsDevCmd=str(vsdev), VsDevCmd_exists=vsdev.is_file(), windows_sdk_versions=sdks)
    if not vsdev.is_file():
        return Check("visual_studio", FAIL, "VsDevCmd.bat missing", d, hint)
    if not sdks:
        return Check("visual_studio", FAIL, "no Windows 10/11 SDK under Windows Kits\\10\\Include", d, hint)
    return Check("visual_studio", PASS, f"{inst.get('displayName')} {inst.get('installationVersion')}; SDK {sdks[-1]}", d)


def capture_vs_env(run: Run, vsdev: Path) -> dict[str, str]:
    """Environment after `VsDevCmd.bat -arch=amd64`: INCLUDE/LIB/PATH for cl, clang-cl and hipcc."""
    r = run.run(["cmd", "/c", f'call "{vsdev}" -arch=amd64 -host_arch=amd64 >nul 2>&1 && set'], "vsdevcmd-env", timeout=120, stream=False)
    env = parse_set_output(r.out)
    diff = env_diff(dict(os.environ), env)
    log.debug("    VsDevCmd changed/added %d variables: %s", len(diff), sorted(diff))
    return env


@_guard
def probe_index(run: Run) -> Check:
    """Is the AMD index reachable and does it hold the pinned Windows wheels? Pure HTTP, works on any OS."""
    d: dict = {"index": AMD_INDEX, "torch_pin": TORCH_VER, "rocm_pin": ROCM_VER}
    t0 = time.monotonic()
    found = {}
    for pkg, ver in (("torch", TORCH_VER), ("rocm-sdk-core", ROCM_VER), ("rocm-sdk-devel", ROCM_VER),
                     ("amd-torch-device-gfx1151", TORCH_VER)):
        url = f"{AMD_INDEX.rstrip('/')}/{pkg}/"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kyojin-win"}), timeout=60) as resp:
                html = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as e:
            d["error"] = f"{url}: {e}"
            return Check("amd_index", FAIL, f"cannot read {url}: {e}", d,
                         "check network/proxy; the index host may have moved (TheRock RELEASES.md lists https://nightly.repo.amd.com/rocm/whl-next/): set KYOJIN_ROCM_INDEX")
        found[pkg] = parse_index_wheels(html, pkg, ver, f"cp{PY_VER.replace('.', '')}")
    d["wheels"] = found
    d["seconds"] = round(time.monotonic() - t0, 1)
    missing = [p for p, w in found.items() if not w]
    if missing:
        return Check("amd_index", FAIL, f"no win_amd64 wheel at the pinned version for: {', '.join(missing)}", d,
                     "the nightly was pruned or renamed: pick a newer version and set KYOJIN_TORCH_VER / KYOJIN_ROCM_VER")
    return Check("amd_index", PASS, f"pinned wheels present ({sum(map(len, found.values()))} files)", d)


# -- probes that need the venv ------------------------------------------------------------------------------------------

def venv_python() -> Path:
    return VENV / "Scripts" / "python.exe" if is_windows() else VENV / "bin" / "python"


def venv_tool(name: str) -> Path:
    return VENV / ("Scripts" if is_windows() else "bin") / (name + (".exe" if is_windows() else ""))


TORCH_PROBE = r"""
import json, sys, time, traceback
out = {"python": sys.version}
try:
    import torch
    out.update(torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda, file=torch.__file__)
    out["is_available"] = torch.cuda.is_available()
    if out["is_available"]:
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        out["device"] = dict(name=p.name, arch=getattr(p, "gcnArchName", None), total_gb=round(p.total_memory / 2**30, 2),
                             mem_free_gb=round(free / 2**30, 2), mem_total_gb=round(total / 2**30, 2), cus=p.multi_processor_count)
        a = torch.randn(512, 512, dtype=torch.float16, device="cuda")
        t = time.time(); (a @ a).sum().item(); torch.cuda.synchronize()
        out["matmul_fp16_512_ms"] = round((time.time() - t) * 1000, 1)
        big = torch.empty(int(2**30), dtype=torch.uint8, device="cuda"); del big
        out["alloc_1gib"] = "ok"
    from torch.utils import cpp_extension as ce
    out["cpp_extension"] = dict(ROCM_HOME=ce.ROCM_HOME, IS_HIP_EXTENSION=ce.IS_HIP_EXTENSION, IS_WINDOWS=ce.IS_WINDOWS)
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"
    out["traceback"] = traceback.format_exc()
print("KYOJIN_PROBE_JSON=" + json.dumps(out))
"""

TRITON_PROBE = r"""
import json, sys, traceback
out = {}
try:
    import torch, triton, triton.language as tl
    out["triton"] = triton.__version__
    @triton.jit
    def add(x, y, o, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
    x = torch.randn(4096, device="cuda"); y = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
    add[(4,)](x, y, o, 4096, B=1024)
    torch.cuda.synchronize()
    out["max_abs_err"] = float((o - (x + y)).abs().max())
    out["ok"] = out["max_abs_err"] == 0.0
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"
    out["traceback"] = traceback.format_exc()
print("KYOJIN_PROBE_JSON=" + json.dumps(out))
"""

EXT_PROBE = r"""
import json, traceback
out = {}
try:
    import exllamav3_ext as e
    out["file"] = e.__file__
    out["n_symbols"] = len([n for n in dir(e) if not n.startswith("_")])
    out["sample"] = sorted(n for n in dir(e) if not n.startswith("_"))[:8]
    import exllamav3
    out["exllamav3"] = getattr(exllamav3, "__version__", "imported")
    out["exllamav3_file"] = exllamav3.__file__
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"
    out["traceback"] = traceback.format_exc()
print("KYOJIN_PROBE_JSON=" + json.dumps(out))
"""


def run_probe_script(run: Run, name: str, script: str, env: dict[str, str] | None = None, timeout: float = 600) -> tuple[Result, dict]:
    """Run a probe snippet in the venv interpreter (subprocess: a native crash cannot kill the orchestrator)."""
    r = run.run([str(venv_python()), "-c", script], name, env=env, cwd=REPO, timeout=timeout, stream=False)
    data: dict = {"exit": describe_exit(r.code), "log": r.logfile.name}
    for line in r.out.splitlines():
        if line.startswith("KYOJIN_PROBE_JSON="):
            with contextlib.suppress(json.JSONDecodeError):
                data.update(json.loads(line.split("=", 1)[1]))
    if "error" not in data and not r.ok:
        data["error"] = f"process exited {describe_exit(r.code)} without a result; last output:\n{r.tail()}"
    return r, data


def probe_error(name: str, d: dict, default_hint: str) -> Check:
    """FAIL check for a probe snippet that reported an error: diagnosis from the traceback, first line as summary."""
    known = diagnose(d.get("traceback", "") + "\n" + d.get("error", ""))
    d["known"] = [k.id for k in known]
    first = d["error"].splitlines()[0] if d.get("error") else "failed"
    if known:
        k = known[0]
        d["why"] = k.cause
        return Check(name, FAIL, f"{first}  [{k.id}: {k.title}]", d, k.fix)
    return Check(name, FAIL, first, d, default_hint)


def need_venv() -> Check | None:
    if venv_python().is_file():
        return None
    return Check("venv", FAIL, f"{VENV} not found", {}, "run: uv run scripts/windows/kyojin_win.py setup")


@_guard
def probe_rocm_sdk(run: Run, env: dict[str, str] | None = None) -> Check:
    miss = need_venv()
    if miss:
        return Check("rocm_sdk", SKIP, "venv missing", {}, miss.hint)
    tool = venv_tool("rocm-sdk")
    d: dict = {"rocm_sdk_tool": str(tool), "exists": tool.is_file()}
    if not tool.is_file():
        return Check("rocm_sdk", FAIL, "rocm-sdk CLI not in the venv", d, "run setup (installs rocm[devel,device-gfx1151])")
    root = run.run([str(tool), "path", "--root"], "rocm-sdk-root", env=env, timeout=120, stream=False)
    d["root_exit"] = describe_exit(root.code)
    rp = Path(root.out.strip().splitlines()[-1]) if root.ok and root.out.strip() else None
    d["root"] = str(rp) if rp else None
    if not rp or not rp.is_dir():
        return Check("rocm_sdk", FAIL, "rocm-sdk path --root failed or points nowhere", d, "run: .venv-win\\Scripts\\rocm-sdk init   (see log)")
    hipcc = rp / "bin" / "hipcc.exe"
    d["hipcc"] = str(hipcc)
    d["hipcc_exists"] = hipcc.is_file()
    d["bin_listing"] = sorted(p.name for p in (rp / "bin").iterdir())[:80] if (rp / "bin").is_dir() else []
    d["root_listing"] = sorted(p.name for p in rp.iterdir())[:40]
    d["headers"] = {h: (rp / "include" / h).exists() for h in ("hipblas", "hipsparse", "thrust", "hip")}
    if not hipcc.is_file():
        return Check("rocm_sdk", FAIL, "hipcc.exe not found in SDK root\\bin (devel tree not expanded?)", d, "run: .venv-win\\Scripts\\rocm-sdk init")
    v = run.run([str(hipcc), "--version"], "hipcc-version", env=sdk_env(rp, env or dict(os.environ)), timeout=120, stream=False)
    d["hipcc_version"] = v.out.strip()[:600]
    d["hipcc_exit"] = describe_exit(v.code)
    missing = [h for h, ok in d["headers"].items() if not ok]
    if not v.ok:
        return Check("rocm_sdk", FAIL, f"hipcc --version failed: {describe_exit(v.code)}", d, "see the hipcc-version log (missing VC runtime DLL?)")
    return Check("rocm_sdk", WARN if missing else PASS, f"{rp}; headers missing: {missing}" if missing else f"{rp}; hipcc ok", d,
                 "include dirs missing: re-run `rocm-sdk init`" if missing else "")


@_guard
def probe_torch(run: Run, env: dict[str, str] | None = None) -> Check:
    miss = need_venv()
    if miss:
        return Check("torch_gpu", SKIP, "venv missing", {}, miss.hint)
    r, d = run_probe_script(run, "probe-torch", TORCH_PROBE, env)
    if "error" in d:
        return probe_error("torch_gpu", d, "see probe-torch log; common: driver too old, wrong torch wheel, DLL missing")
    dev = d.get("device") or {}
    if not d.get("hip"):
        return Check("torch_gpu", FAIL, f"torch {d.get('torch')} is not a ROCm build (torch.version.hip empty)", d, "re-run setup")
    if not d.get("is_available"):
        return Check("torch_gpu", FAIL, "torch.cuda.is_available() is False", d, "update the Adrenalin driver; check `hipInfo.exe` from the SDK bin dir")
    arch = str(dev.get("arch", "")).split(":")[0]
    if arch != GFX:
        return Check("torch_gpu", WARN, f"device arch {arch!r}, expected {GFX}", d, "this POC targets gfx1151 only")
    return Check("torch_gpu", PASS, f"torch {d['torch']} hip {d['hip']}; {dev['name']} {arch}; {dev['total_gb']} GiB visible, "
                 f"{dev['mem_free_gb']} GiB free; fp16 matmul {d['matmul_fp16_512_ms']} ms", d)


@_guard
def probe_triton(run: Run, env: dict[str, str] | None = None) -> Check:
    miss = need_venv()
    if miss:
        return Check("triton", SKIP, "venv missing", {}, miss.hint)
    r, d = run_probe_script(run, "probe-triton", TRITON_PROBE, env)
    if "error" in d:
        return probe_error("triton", d, "see probe-triton log; triton-windows needs clang-cl (ships in the ROCm SDK wheel) or MSVC")
    return Check("triton", PASS if d.get("ok") else FAIL, f"triton {d.get('triton')}; vector add max error {d.get('max_abs_err')}", d,
                 "" if d.get("ok") else "kernel ran but results differ")


@_guard
def probe_extension(run: Run, env: dict[str, str] | None = None) -> Check:
    miss = need_venv()
    if miss:
        return Check("extension", SKIP, "venv missing", {}, miss.hint)
    e = dict(os.environ if env is None else env)
    e["PYTHONPATH"] = str(REPO) + os.pathsep + e.get("PYTHONPATH", "")
    r, d = run_probe_script(run, "probe-extension", EXT_PROBE, e)
    if "error" in d:
        return probe_error("extension", d, "run `build` first; if built, see probe-extension log (missing DLLs? run with -v)")
    return Check("extension", PASS, f"exllamav3_ext at {d['file']}; {d['n_symbols']} symbols; exllamav3 {d['exllamav3']}", d)


# ----------------------------------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------------------------------

def host_probes(run: Run):
    run.add(probe_host(run))
    run.add(probe_cpu_ram(run))
    run.add(probe_gpu(run))
    run.add(probe_disk(run))
    run.add(probe_longpaths(run))
    run.add(probe_tool(run, "git", ["--version"], "winget install Git.Git"))
    run.add(probe_tool(run, "uv", ["--version"], "winget install astral-sh.uv   (then open a new terminal)"))
    run.add(probe_vs(run))
    run.add(probe_index(run))


def cmd_doctor(run: Run, args) -> int:
    host_probes(run)
    if venv_python().is_file():
        sdk = run.add(probe_rocm_sdk(run))
        run.add(probe_torch(run, build_env(run, sdk)))
        run.add(probe_triton(run, build_env(run, sdk)))
        run.add(probe_extension(run, build_env(run, sdk)))
    else:
        run.add(Check("venv", INFO, f"{VENV} not created yet; run `setup`", {}))
    return run.summary()


def build_env(run: Run, sdk: Check) -> dict[str, str] | None:
    root = sdk.details.get("root")
    if not root:
        return None
    return sdk_env(Path(root), dict(os.environ))


def cmd_setup(run: Run, args) -> int:
    host_probes(run)
    if any(c.status == FAIL for c in run.checks):
        log.error("preflight failed; fix the FAIL items above before setup continues")
        return run.summary()
    uv = shutil.which("uv")
    assert uv
    steps = [
        ([uv, "python", "install", PY_VER], "uv-python-install"),
    ]
    if args.recreate and VENV.exists():
        shutil.rmtree(VENV)
    if not venv_python().is_file():
        steps.append(([uv, "venv", str(VENV), "--python", PY_VER], "uv-venv"))
    py = str(venv_python())
    steps += [
        # 1. torch + ROCm runtime + SDK devel from the AMD index only (its dependency mirrors live there too)
        ([uv, "pip", "install", "--python", py, "--index-url", AMD_INDEX, "--prerelease", "allow", "-v",
          f"torch[device-{GFX}]=={TORCH_VER}", f"rocm[devel,device-{GFX}]=={ROCM_VER}"], "uv-pip-torch-rocm"),
        # 2. everything else from PyPI; torch/rocm are already satisfied and are not replaced
        ([uv, "pip", "install", "--python", py, "--index-url", PYPI_INDEX, "-v", TRITON_SPEC, "-r", str(REPO / "requirements.txt")], "uv-pip-requirements"),
    ]
    for cmd, name in steps:
        r = run.run(cmd, name, timeout=3600, stream=True)
        if not r.ok:
            run.add(run.fail_step(r, name, cmd))
            return run.summary()
        run.add(Check(name, PASS, f"{r.seconds:.0f}s", {"log": r.logfile.name}))
    tool = str(venv_tool("rocm-sdk"))
    for sub, name in ((["init"], "rocm-sdk-init"), (["test"], "rocm-sdk-test"), (["targets"], "rocm-sdk-targets")):
        r = run.run([tool, *sub], name, timeout=1800, stream=True)
        if r.ok:
            run.add(Check(name, PASS, f"{r.seconds:.0f}s", {"log": r.logfile.name, "tail": r.tail(10)}))
        else:
            c = run.fail_step(r, name, [tool, *sub])
            if sub != ["init"]:
                c.status = WARN       # rocm-sdk test/targets are diagnostics; only init is required
            run.add(c)
    fr = run.run([uv, "pip", "freeze", "--python", py], "pip-freeze", stream=False)
    (run.dir / "pip-freeze.txt").write_text(fr.out, encoding="utf-8")
    log.info("pip freeze saved to %s (pin from it once a build works)", run.dir / "pip-freeze.txt")
    sdk = run.add(probe_rocm_sdk(run))
    env = build_env(run, sdk)
    run.add(probe_torch(run, env))
    run.add(probe_triton(run, env))
    return run.summary()


def cmd_build(run: Run, args) -> int:
    miss = need_venv()
    if miss:
        run.add(miss)
        return run.summary()
    vs = run.add(probe_vs(run))
    sdk = run.add(probe_rocm_sdk(run))
    ram = probe_cpu_ram(run)
    run.add(ram)
    if vs.status == FAIL or sdk.status == FAIL:
        return run.summary()
    env = capture_vs_env(run, Path(vs.details["VsDevCmd"]))
    root = Path(sdk.details["root"])
    env = sdk_env(root, env)
    jobs = compute_jobs(os.cpu_count() or 1, ram.details.get("avail_phys_gb", 16), args.jobs)
    env["MAX_JOBS"] = str(jobs)
    if args.verbose_build:
        env["EXLLAMA_VERBOSE"] = "1"      # setup.py: verbose setuptools
        env["VERBOSE"] = "1"
    run.snapshot_env(env, "build-env.json")
    log.info("build: jobs=%d arch=%s sdk=%s (full environment in build-env.json)", jobs, GFX, root)
    if args.clean:
        for p in (REPO / "build", *REPO.glob("exllamav3_ext*.pyd")):
            log.info("clean: removing %s", p)
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    uv = shutil.which("uv") or "uv"
    cmd = [uv, "pip", "install", "--python", str(venv_python()), "--no-build-isolation", "--no-deps", "-v", "-e", str(REPO)]
    r = run.run(cmd, "build-extension", env=env, cwd=REPO, timeout=4 * 3600, stream=True)
    pyds = sorted(p.name for p in REPO.glob("exllamav3_ext*.pyd"))
    if r.ok and not pyds:
        run.add(Check("build", FAIL, "install succeeded but no exllamav3_ext*.pyd was produced", {"log": r.logfile.name},
                      "was EXLLAMA_NOCOMPILE set? run `build --clean`; check the log for 'building exllamav3_ext'"))
    elif r.ok:
        run.add(Check("build", PASS, f"{r.seconds / 60:.1f} min; artifacts: {pyds}", {"log": r.logfile.name, "pyd": pyds}))
    else:
        run.add(run.fail_step(r, "build exllamav3_ext", cmd, "search the log for 'FAILED:' (ninja prints the failing command after it)"))
    if r.ok:
        run.add(probe_extension(run, env))
    return run.summary()


def cmd_check(run: Run, args) -> int:
    sdk = run.add(probe_rocm_sdk(run))
    env = build_env(run, sdk)
    run.add(probe_torch(run, env))
    run.add(probe_triton(run, env))
    run.add(probe_extension(run, env))
    return run.summary()


def cmd_serve(run: Run, args) -> int:
    miss = need_venv()
    if miss:
        run.add(miss)
        return run.summary()
    model = Path(args.model)
    needed = ["config.json", "chat_template.jinja", "generation_config.json"]
    have = {n: (model / n).is_file() for n in needed}
    run.add(Check("model_dir", PASS if model.is_dir() and all(have.values()) else WARN, str(model),
                  {"files": have, "listing": sorted(p.name for p in model.iterdir())[:50] if model.is_dir() else None},
                  "tools/qwen/serve.py documents chat_template.jinja + generation_config.json in the pack; copy them from the base model if absent"))
    sdk = probe_rocm_sdk(run)
    root = sdk.details.get("root")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    if root:
        env["EXL3_ROCM_SDK"] = root       # util/hip_compiler.py resolves hipcc from this; ROCM_PATH/HIP_PATH deliberately not set at runtime
    cmd = serve_command(str(venv_python()), str(model), args.port, args.ctx)
    log.info("serving: %s", " ".join(_q(c) for c in cmd))
    r = run.run(cmd, "serve", env=env, cwd=REPO, stream=True)
    run.add(Check("serve", INFO, f"server exited {describe_exit(r.code)}", {"log": r.logfile.name, "tail": r.tail(40)}))
    return run.summary()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="echo all DEBUG output (subprocess output, env deltas) to the console; run.log always has it")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="read-only preflight and probes")
    s = sub.add_parser("setup", help="create .venv-win and install ROCm torch, triton-windows, requirements")
    s.add_argument("--recreate", action="store_true", help="delete and recreate .venv-win first")
    b = sub.add_parser("build", help="compile exllamav3_ext (editable install)")
    b.add_argument("--jobs", type=int, help="override MAX_JOBS (default from free RAM, max 6)")
    b.add_argument("--clean", action="store_true", help="delete build/ and old .pyd first")
    b.add_argument("--verbose-build", action="store_true", help="set EXLLAMA_VERBOSE/VERBOSE for setuptools")
    sub.add_parser("check", help="torch/GPU, Triton and extension smoke tests")
    sv = sub.add_parser("serve", help="run tools/qwen/serve.py on a small pack")
    sv.add_argument("--model", required=True)
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args(argv)
    run = Run(args.cmd, args.verbose)
    try:
        return {"doctor": cmd_doctor, "setup": cmd_setup, "build": cmd_build, "check": cmd_check, "serve": cmd_serve}[args.cmd](run, args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception:  # noqa: BLE001
        import traceback
        log.error("INTERNAL ERROR in kyojin_win.py (a bug in this script, not your environment):\n%s", traceback.format_exc())
        log.error("Please send the log folder: %s", run.dir)
        return 2
    finally:
        run.write_report()
        run.close()


if __name__ == "__main__":
    sys.exit(main())
