"""Logic tests for kyojin_win.py. Stdlib unittest, no torch/GPU/Windows needed:  python -m unittest scripts/windows/test_kyojin_win.py"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kyojin_win as kw  # noqa: E402


class Pure(unittest.TestCase):
    def test_redact(self):
        r = kw.redact_env({"HF_TOKEN": "x", "GITHUB_PAT": "y", "MY_API_KEY": "z", "PATH": "p", "HOME": "h"})
        self.assertEqual(r["HF_TOKEN"], "<redacted>")
        self.assertEqual(r["GITHUB_PAT"], "<redacted>")
        self.assertEqual(r["MY_API_KEY"], "<redacted>")
        self.assertEqual(r["PATH"], "p")

    def test_exit_names(self):
        self.assertIn("ACCESS_VIOLATION", kw.describe_exit(3221225477))
        self.assertIn("DLL_NOT_FOUND", kw.describe_exit(0xC0000135))
        self.assertIn("0x00000001", kw.describe_exit(1))
        self.assertIn("timeout", kw.describe_exit(None))

    def test_parse_set(self):
        txt = "A=1\nPath=C:\\x;C:\\y\n=C:=C:\\\nEMPTY=\nnoequals\nB=a=b\n"
        self.assertEqual(kw.parse_set_output(txt), {"A": "1", "Path": "C:\\x;C:\\y", "EMPTY": "", "B": "a=b"})

    def test_env_diff_case_insensitive(self):
        d = kw.env_diff({"Path": "a", "X": "1"}, {"PATH": "b", "X": "1", "INCLUDE": "i"})
        self.assertEqual(d["PATH"], ("a", "b"))
        self.assertEqual(d["INCLUDE"], (None, "i"))
        self.assertNotIn("X", d)

    def test_pick_vs(self):
        self.assertIsNone(kw.pick_vs("[]"))
        self.assertIsNone(kw.pick_vs("not json"))
        self.assertEqual(kw.pick_vs('[{"installationPath": "C:\\\\VS"}]')["installationPath"], "C:\\VS")

    def test_jobs(self):
        self.assertEqual(kw.compute_jobs(32, 128), 6)
        self.assertEqual(kw.compute_jobs(32, 10), 2)
        self.assertEqual(kw.compute_jobs(2, 128), 2)
        self.assertEqual(kw.compute_jobs(32, 1), 1)
        self.assertEqual(kw.compute_jobs(32, 128, 3), 3)

    def test_index_parse(self):
        html = ('<a href="../torch-2.15.0a0%2Brocm10.1.0a20260822-cp312-cp312-win_amd64.whl">x</a>'
                '<a href="../torch-2.15.0a0%2Brocm10.1.0a20260822-cp312-cp312-linux_x86_64.whl">x</a>'
                '<a href="../torch-2.15.0a0%2Brocm10.1.0a20260822-cp313-cp313-win_amd64.whl">x</a>'
                '<a href="../torch-2.14.0%2Brocm10.1.0a20260822-cp312-cp312-win_amd64.whl">x</a>')
        w = kw.parse_index_wheels(html, "torch", "2.15.0a0+rocm10.1.0a20260822", "cp312")
        self.assertEqual(w, ["torch-2.15.0a0+rocm10.1.0a20260822-cp312-cp312-win_amd64.whl"])
        sdk = '<a href="rocm_sdk_core-10.1.0a20260822-py3-none-win_amd64.whl#sha256=ab">x</a>'
        self.assertEqual(len(kw.parse_index_wheels(sdk, "rocm-sdk-core", "10.1.0a20260822")), 1)
        self.assertEqual(kw.parse_index_wheels(sdk, "rocm-sdk-devel", "10.1.0a20260822"), [])

    def test_sdk_env(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "lib" / "llvm" / "amdgcn" / "bitcode").mkdir(parents=True)
            e = kw.sdk_env(root, {"PATH": "orig"})
            self.assertEqual(e["ROCM_HOME"], str(root))
            self.assertEqual(e["PYTORCH_ROCM_ARCH"], "gfx1151")
            self.assertTrue(e["PATH"].startswith(str(root / "bin")))
            self.assertTrue(e["PATH"].endswith("orig"))
            self.assertEqual(e["HIP_DEVICE_LIB_PATH"], str(root / "lib" / "llvm" / "amdgcn" / "bitcode"))
            self.assertEqual(e["EXL3_HIP_DEFINES"], "EXL3_HIP_STG_PAD")
            self.assertEqual(kw.sdk_env(root, {"PATH": "o", "EXL3_HIP_DEFINES": "A"})["EXL3_HIP_DEFINES"], "A")

    def test_parse_cim_gpus(self):
        one = '{"Name": "AMD Radeon(TM) 8060S Graphics", "DriverVersion": "32.0.1"}'
        self.assertEqual(kw.parse_cim_gpus(one)[0]["Name"], "AMD Radeon(TM) 8060S Graphics")
        self.assertEqual(len(kw.parse_cim_gpus("[" + one + "," + one + "]")), 2)
        self.assertEqual(kw.parse_cim_gpus(""), [])
        self.assertEqual(kw.parse_cim_gpus("not json"), [])

    def test_serve_command(self):
        c = kw.serve_command("py", "m", 8000, 8192)
        self.assertIn("--no-vision", c)
        self.assertEqual(c[c.index("--draft-policy") + 1], "off")
        self.assertTrue(c[2].replace("\\", "/").endswith("tools/qwen/serve.py"))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_ = kw.Run("unit", verbose=False, root=Path(self.tmp.name))

    def tearDown(self):
        self.run_.close()
        self.tmp.cleanup()

    def test_subprocess_logging(self):
        r = self.run_.run([sys.executable, "-c", "import sys;print('out');print('err',file=sys.stderr);sys.exit(3)"], "echo test")
        self.assertEqual(r.code, 3)
        self.assertIn("out", r.out)
        self.assertIn("err", r.out)            # stderr merged
        txt = r.logfile.read_text()
        self.assertIn("exit 3", txt)
        self.assertTrue(r.logfile.name.startswith("01-echo_test"))
        self.assertIn("RUN echo test", (self.run_.dir / "run.log").read_text())

    def test_missing_executable(self):
        r = self.run_.run(["definitely-not-a-binary-xyz"], "missing")
        self.assertIsNone(r.code)
        self.assertTrue(r.error)

    def test_timeout_kills(self):
        r = self.run_.run([sys.executable, "-c", "import time;time.sleep(30)"], "sleepy", timeout=1)
        self.assertNotEqual(r.code, 0)
        self.assertLess(r.seconds, 15)

    def test_env_delta_logged_and_redacted(self):
        env = dict(os.environ, KYOJIN_TEST_VAR="visible", HF_TOKEN="secret123")
        self.run_.run([sys.executable, "-c", "pass"], "envtest", env=env)
        log = (self.run_.dir / "run.log").read_text()
        self.assertIn("KYOJIN_TEST_VAR", log)
        self.assertNotIn("secret123", log)
        self.assertNotIn("secret123", (self.run_.dir / "env.json").read_text())

    def test_report_and_summary(self):
        self.run_.add(kw.Check("a", kw.PASS, "fine", {"k": Path("/x")}))
        self.run_.add(kw.Check("b", kw.FAIL, "broken", {}, "do this"))
        self.assertEqual(self.run_.summary(), 1)
        rep = json.loads((self.run_.dir / "report.json").read_text())
        self.assertEqual([c["status"] for c in rep], ["PASS", "FAIL"])
        self.assertEqual(rep[1]["hint"], "do this")


class HelpfulErrors(unittest.TestCase):
    SAMPLES = {
        "uv-project-sync": "Creating virtual environment at: .venv",
        "no-msvc": "subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.",
        "distutils-use-sdk": "UserWarning: It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.",
        "missing-win-headers": "C:\\sdk\\hip_runtime.h(10,10): fatal error: 'cstdint' file not found",
        "missing-hip-headers": "fatal error: 'hipblas/hipblas.h' file not found",
        "no-hip-runtime": "clang: error: cannot find HIP runtime; provide its path via '--rocm-path'",
        "no-rocm-device-lib": "clang: error: cannot find ROCm device library; provide its path via '--rocm-path'",
        "torch-not-rocm": "OSError: CUDA_HOME environment variable is not set.",
        "build-isolation": "ModuleNotFoundError: No module named 'torch'",
        "restrict-cl": "exl3_kernel_map.cuh(53): error C2146: syntax error: missing ')' before identifier 'A'",
        "mkdir-2arg": "hgemm.cu:163:5: error: use of undeclared identifier 'mkdir'",
        "unresolved-link": "hgemm.obj : error LNK2019: unresolved external symbol rocblas_create_handle referenced in function",
        "missing-import-lib": "LINK : fatal error LNK1181: cannot open input file 'rocblas.lib'",
        "missing-crt-lib": "LINK : fatal error LNK1104: cannot open file 'msvcprt.lib'",
        "path-too-long": "OSError: [WinError 206] The filename or extension is too long",
        "dll-missing": "ImportError: DLL load failed while importing exllamav3_ext: The specified module could not be found.",
        "entrypoint": "ImportError: DLL load failed while importing exllamav3_ext: The specified procedure could not be found.",
        "no-matching-dist": "No matching distribution found for torch==2.15.0a0+rocm10.1.0a20260822",
        "network": "ConnectionResetError: Connection reset by peer",
        "compiler-oom": "LLVM ERROR: out of memory",
        "openblas-oom": "OpenBLAS error: Memory allocation still failed after 10 retries, giving up.",
        "hip-oom": "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 16.00 GiB",
        "file-locked": "PermissionError: [WinError 32] The process cannot access the file because it is being used by another process",
        "flag-leak": "clang: error: unknown argument: '-Xcompiler=/Zc:preprocessor'",
        "python-h": "fatal error: 'Python.h' file not found",
        "ext-not-built": "ImportError: exllamav3_ext is not built. On Windows with ROCm torch the JIT build is not supported",
        "invalid-device-function": "RuntimeError: HIP error: invalid device function",
        "tdr": "RuntimeError: HIP error: unspecified launch failure",
        "triton-hip-dll": "RuntimeError: cannot locate amdhip64.dll after attempted paths ['C:\\\\x']",
        "triton-cc-fail": "Failed to compile. cc_cmd: ['clang-cl.exe', 'main.c']",
        "triton-import-order": "ImportError: DLL load failed while importing libtriton",
        "no-triton": "ModuleNotFoundError: No module named 'triton'",
        "symlink-checkout": 'File "C:\\k\\tools\\qwen\\image_cache.py", line 1',
        "register-at-fork": "AttributeError: module 'os' has no attribute 'register_at_fork'",
        "hip-lib-windows": "OSError: no libamdhip64.so or libamdhip64.so.<version> in the torch lib directory",
    }

    def test_every_signature_has_a_sample_and_matches(self):
        ids = {k.id for k in kw.KNOWN}
        self.assertEqual(ids, set(self.SAMPLES), "add a sample for every signature (and vice versa)")
        for kid, text in self.SAMPLES.items():
            self.assertIn(kid, [k.id for k in kw.diagnose(text)], kid)

    def test_symlink_probe_on_a_real_checkout(self):
        with tempfile.TemporaryDirectory() as t:
            run = kw.Run("unit", verbose=False, root=Path(t))
            try:
                c = kw.probe_symlinks(run)
            finally:
                run.close()
        self.assertEqual(c.status, kw.PASS, c.summary)   # Linux checkout has real symlinks

    def test_no_false_positive_on_clean_output(self):
        clean = "Compiling foo.cu\n[12/40] hipcc -O3 ...\nBuild succeeded. 0 errors. -Wno-error ok\n"
        self.assertEqual(kw.diagnose(clean), [])
        self.assertEqual(kw.extract_errors(clean), [])

    def test_extract_first_errors_with_line_numbers(self):
        out = "\n".join(["ok"] * 5 + ["FAILED: a.o", "a.cu:1: error: bad thing"] + ["noise"] * 50 + ["ninja: error: tail"])
        got = kw.extract_errors(out, limit=2, first_line_no=2)
        self.assertEqual([n for n, _ in got], [7, 8])
        self.assertIn("FAILED", got[0][1])

    def test_render_failure_contains_cause_fix_and_lines(self):
        k = [x for x in kw.KNOWN if x.id == "compiler-oom"]
        txt = kw.render_failure("build", ["uv", "pip"], 3221225477, 12.0, "05-build.log", k, [(9, "error: x")], "a\nb")
        for needle in ("FAILED: build", "ACCESS_VIOLATION", "05-build.log", "likely  : Compiler ran out of memory", "fix", "L9: error: x"):
            self.assertIn(needle, txt)
        self.assertIn("no known signature", kw.render_failure("t", ["x"], 1, 1.0, "l", [], [], ""))


class FailStep(unittest.TestCase):
    def test_fail_step_writes_failure_md_and_check(self):
        with tempfile.TemporaryDirectory() as t:
            run = kw.Run("unit", verbose=False, root=Path(t))
            try:
                r = run.run([sys.executable, "-c", "import sys;print('FAILED: x.o');print('fatal error: \\'hip/hip_runtime.h\\' file not found');sys.exit(1)"], "compile")
                c = run.fail_step(r, "build exllamav3_ext", ["cmd"])
                self.assertEqual(c.status, kw.FAIL)
                self.assertIn("missing-hip-headers", c.details["known"])
                self.assertIn("rocm-sdk init", c.hint)
                md = (run.dir / "FAILURE.md").read_text()
                self.assertIn("hip/hip_runtime.h", md)
                self.assertIn("L2:", md)           # line numbers count the log's "$ command" header as line 1
                # first_errors line number really points at that line of the step log
                n = c.details["first_errors"][0][0]
                self.assertIn("FAILED", r.logfile.read_text().splitlines()[n - 1])
            finally:
                run.close()

    def test_probe_error_diagnoses_traceback(self):
        c = kw.probe_error("triton", {"error": "ImportError: x", "traceback": "ModuleNotFoundError: No module named 'triton'"}, "dflt")
        self.assertIn("no-triton", c.summary)
        self.assertIn("setup", c.hint)
        c2 = kw.probe_error("triton", {"error": "Weird: new"}, "dflt")
        self.assertEqual(c2.hint, "dflt")


@unittest.skipIf(kw.is_windows(), "Linux-side behaviour of Windows-only probes")
class ProbesOffWindows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_ = kw.Run("unit", verbose=False, root=Path(self.tmp.name))

    def tearDown(self):
        self.run_.close()
        self.tmp.cleanup()

    def test_host_fails_with_details(self):
        c = kw.probe_host(self.run_)
        self.assertEqual(c.status, kw.FAIL)
        self.assertIn("platform", c.details)

    def test_probe_crash_is_contained(self):
        with mock.patch.object(kw.shutil, "disk_usage", side_effect=RuntimeError("boom")):
            c = kw.probe_disk(self.run_)
        self.assertEqual(c.status, kw.FAIL)
        self.assertIn("boom", c.summary)
        self.assertIn("Traceback", c.details["traceback"])

    def test_missing_tool(self):
        c = kw.probe_tool(self.run_, "definitely-not-a-binary-xyz", ["--version"], "install it")
        self.assertEqual((c.status, c.hint), (kw.FAIL, "install it"))

    def test_probe_script_error_path(self):
        with mock.patch.object(kw, "venv_python", return_value=Path(sys.executable)):
            r, d = kw.run_probe_script(self.run_, "p", "import json;print('KYOJIN_PROBE_JSON='+json.dumps({'ok': 1}))")
            self.assertEqual(d["ok"], 1)
            r, d = kw.run_probe_script(self.run_, "p2", "raise SystemExit(7)")
            self.assertIn("error", d)
            self.assertIn("0x00000007", d["error"])
            # real torch probe on a box without torch: error is captured as data, not raised
            r, d = kw.run_probe_script(self.run_, "p3", kw.TRITON_PROBE)
            self.assertTrue("error" in d or d.get("ok") is not None)

    def test_index_probe_offline_and_online(self):
        with mock.patch.object(kw.urllib.request, "urlopen", side_effect=OSError("offline")):
            c = kw.probe_index(self.run_)
        self.assertEqual(c.status, kw.FAIL)
        self.assertIn("offline", c.summary)

        class Resp:
            def __init__(self, html): self.html = html
            def read(self): return self.html.encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake(req, timeout=0):
            pkg = req.full_url.rstrip("/").rsplit("/", 1)[1]
            fn = {"torch": "torch-2.15.0a0%2Brocm10.1.0a20260822-cp312-cp312-win_amd64.whl",
                  "rocm-sdk-core": "rocm_sdk_core-10.1.0a20260822-py3-none-win_amd64.whl",
                  "rocm-sdk-devel": "rocm_sdk_devel-10.1.0a20260822-py3-none-win_amd64.whl",
                  "amd-torch-device-gfx1151": "amd_torch_device_gfx1151-2.15.0a0%2Brocm10.1.0a20260822-cp312-cp312-win_amd64.whl"}[pkg]
            return Resp(f'<a href="{fn}">x</a>')
        with mock.patch.object(kw.urllib.request, "urlopen", fake):
            c = kw.probe_index(self.run_)
        self.assertEqual(c.status, kw.PASS, c.summary)


if __name__ == "__main__":
    unittest.main()
