"""Tests for tasks.py (stdlib unittest):  python scripts/windows/test_tasks.py"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as T  # noqa: E402


def make(root: Path):
    docs = root / "doc" / "windows"
    s1 = docs / "sprints" / "s1-first"
    s2 = docs / "sprints" / "s2-second"
    for d in (s1, s2):
        d.mkdir(parents=True)
    (docs / "README.md").write_text("# Index\n", encoding="utf-8")
    (s1 / "README.md").write_text("# Sprint 1\n", encoding="utf-8")
    (s2 / "README.md").write_text("# Sprint 2\n", encoding="utf-8")
    (s1 / "01-hard.md").write_text("# Hard\n\n- Status: todo\n- Depends: -\n- Risk: unknown\n\n[next](02-easy.md)\n", encoding="utf-8")
    (s1 / "02-easy.md").write_text("# Easy\n\n- Status: todo\n- Depends: S1-01\n- Risk: low\n", encoding="utf-8")
    (s2 / "01-later.md").write_text("# Later\n\n- Status: todo\n- Depends: S1-02\n- Risk: r\n", encoding="utf-8")
    return docs


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docs = make(Path(self.tmp.name))
        T.sync(self.docs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_order(self):
        ids = [t.id for t in T.all_tasks(T.load(self.docs))]
        self.assertEqual(ids, ["S1-01", "S1-02", "S2-01"])

    def test_next_respects_dependencies(self):
        self.assertEqual(T.next_task(T.load(self.docs)).id, "S1-01")
        T.set_status("S1-01", "done", self.docs)
        self.assertEqual(T.next_task(T.load(self.docs)).id, "S1-02")
        T.set_status("S1-02", "blocked", self.docs)
        self.assertIsNone(T.next_task(T.load(self.docs)))     # S2-01 waits on a non-done task; blocked is not actionable
        T.set_status("S1-02", "done", self.docs)
        self.assertEqual(T.next_task(T.load(self.docs)).id, "S2-01")

    def test_set_status_updates_file_and_tables(self):
        T.set_status("S1-01", "active", self.docs)
        txt = (self.docs / "sprints" / "s1-first" / "01-hard.md").read_text()
        self.assertIn("- Status: active", txt)
        self.assertIn("| 1 | S1-01 | [Hard](01-hard.md) | active | - |", (self.docs / "sprints" / "s1-first" / "README.md").read_text())
        self.assertIn("in progress", (self.docs / "README.md").read_text())
        self.assertEqual(T.check(self.docs), [])

    def test_sync_idempotent(self):
        self.assertEqual(T.sync(self.docs), [])

    def test_check_finds_problems(self):
        f = self.docs / "sprints" / "s1-first" / "02-easy.md"
        f.write_text("# Easy\n\n- Status: wip\n- Depends: S9-09\n- Risk:\n\n[bad](nope.md)\n", encoding="utf-8")
        probs = "\n".join(T.check(self.docs))
        for needle in ("bad status 'wip'", "unknown dependency S9-09", "missing '- Risk:'", "broken link nope.md", "stale"):
            self.assertIn(needle, probs)

    def test_set_unknown(self):
        with self.assertRaises(SystemExit):
            T.set_status("S9-99", "done", self.docs)
        with self.assertRaises(SystemExit):
            T.set_status("S1-01", "nonsense", self.docs)


class RealDocs(unittest.TestCase):
    def test_repo_docs_are_consistent(self):
        self.assertEqual(T.check(), [])


if __name__ == "__main__":
    unittest.main()
