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
