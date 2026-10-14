"""Check release hashes against real staged Git blobs, including Windows line endings."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).resolve().parents[1] / "tools" / "check_git_index.py"


class GitIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        (self.repo / "tools").mkdir()
        (self.repo / "tools/check_git_index.py").write_bytes(
            CHECKER.read_bytes().replace(b"\r\n", b"\n"))
        (self.repo / ".gitattributes").write_bytes(b"* text=auto eol=lf\n")
        self.sample = self.repo / "sample.py"
        self.sample.write_bytes(b"value = 1\n")
        self.manifest = self.repo / "MANIFEST_SHA256.json"
        hashes = {
            name: hashlib.sha256((self.repo / name).read_bytes()).hexdigest()
            for name in ["tools/check_git_index.py", ".gitattributes", "sample.py"]
        }
        self.manifest.write_text(json.dumps(hashes) + "\n", encoding="utf-8")
        self.git("add", ".")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args],
                              capture_output=True, check=True)

    def check(self):
        result = subprocess.run([sys.executable, str(self.repo / "tools/check_git_index.py")],
                                capture_output=True, text=True)
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        return result, json.loads(result.stdout)

    def test_valid_index_passes(self):
        result, report = self.check()
        self.assertEqual(result.returncode, 0, report)
        self.assertEqual(report["status"], "PASS")

    def test_worktree_crlf_does_not_change_staged_hash(self):
        self.sample.write_bytes(b"value = 1\r\n")
        result, report = self.check()
        self.assertEqual(result.returncode, 0, report)
        self.assertEqual(report["status"], "PASS")

    def test_staged_corruption_cannot_be_hidden_by_worktree(self):
        self.sample.write_bytes(b"value = 2\n")
        self.git("add", "sample.py")
        self.sample.write_bytes(b"value = 1\n")
        result, report = self.check()
        self.assertEqual(result.returncode, 1, report)
        self.assertEqual(report["source_hash_errors"], ["sample.py: hash mismatch"])

    def test_unstaged_manifest_is_not_used(self):
        hashes = json.loads(self.manifest.read_text())
        hashes["sample.py"] = "0" * 64
        self.manifest.write_text(json.dumps(hashes) + "\n")
        result, report = self.check()
        self.assertEqual(result.returncode, 0, report)

    def test_unexpected_staged_file_is_rejected(self):
        (self.repo / "extra.py").write_bytes(b"value = 3\n")
        self.git("add", "extra.py")
        result, report = self.check()
        self.assertEqual(result.returncode, 1, report)
        self.assertEqual(report["unexpected_tracked"], ["extra.py"])

    def test_missing_staged_file_is_rejected(self):
        self.git("rm", "--cached", "sample.py")
        result, report = self.check()
        self.assertEqual(result.returncode, 1, report)
        self.assertEqual(report["missing_from_git_index"], ["sample.py"])


if __name__ == "__main__":
    unittest.main()
