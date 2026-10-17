"""build_archive.py: main.py contract, safe tar handling, stub archive built and played cold."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import build_archive  # noqa: E402


class MainSourceTests(unittest.TestCase):
    def test_agent_is_last_callable_and_passes_on_missing_policy(self):
        source = build_archive.main_source(threads=1, seed=0, temperature=1.0)
        saved_path = list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                namespace = {"__file__": str(Path(tmp) / "main.py")}
                exec(compile(source, "main.py", "exec"), namespace)
                chosen = [v for v in namespace.values() if callable(v)][-1]
                self.assertEqual(chosen.__name__, "agent")
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    first = chosen({"step": 0, "player": 0}, {})
                    second = chosen({"step": 1, "player": 0}, {})
                self.assertEqual(first, {"farmer": ["PASS"], "hands": [], "market": []})
                self.assertEqual(second, first)
                lines = err.getvalue().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertTrue(lines[0].startswith("[pbc] step=0 ms="), lines)
                self.assertIn("fb=1", lines[1])
                self.assertEqual(namespace["_PBC_STATUS"]["fallbacks"], 2)
        finally:
            sys.path[:] = saved_path
            sys.modules.pop("pbc_agent", None)


class TarTests(unittest.TestCase):
    def test_checked_name(self):
        for bad in ("../x", "/abs", "a/../b", "a//b", "", "a\\b", "./a"):
            with self.assertRaises(ValueError, msg=bad):
                build_archive.checked_name(bad)
        self.assertEqual(build_archive.checked_name("src/a.py"), "src/a.py")

    def test_extract_refuses_unlisted_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.tar.gz"
            manifest = b'{"members": {}}'
            build_archive.write_tar(path, {"MANIFEST.json": manifest, "extra.py": b"x"})
            with self.assertRaises(ValueError):
                build_archive.extract(path, Path(tmp) / "out")

    def test_write_tar_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.tar.gz", Path(tmp) / "b.tar.gz"
            members = {"main.py": b"print(1)\n", "src/x.py": b"y = 2\n"}
            build_archive.write_tar(a, members)
            build_archive.write_tar(b, dict(reversed(list(members.items()))))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with tarfile.open(a) as t:
                self.assertEqual(sorted(m.name for m in t.getmembers()), sorted(members))


class ColdStubArchiveTests(unittest.TestCase):
    def test_stub_archive_plays_complete_cold_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stub.tar.gz"
            built = build_archive.build("stub:9", out, storage="fp16", trace_steps=4)
            names = set(built["manifest"]["members"])
            self.assertIn("main.py", names)
            self.assertIn(f"{build_archive.HERE_REL}/agent.py", names)
            self.assertIn("src/farm_neat/training/bc.py", names)
            self.assertNotIn(f"{build_archive.HERE_REL}/play.py", names)
            self.assertLess(built["archive_bytes"], build_archive.MAX_BYTES)
            receipt = build_archive.verify(out, threads=1, workdir=tmp)
            game = receipt["games"][0]
            self.assertTrue(receipt["complete"], game["cold"])
            self.assertEqual(game["steps"], 719)
            self.assertEqual(game["cold"]["log_lines"], 719)
            self.assertEqual(game["cold"]["hard_errors"], 0)
            self.assertEqual(game["fallbacks"], 0)
            self.assertEqual(game["cold"]["closed"]["our_modules_outside_archive"], [])


if __name__ == "__main__":
    unittest.main()
