"""CLI regression checks using synthetic bytes, without PyTorch or real weights."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_staged_checkpoints.py"
EXPORTER = ROOT / "tools" / "export_autodl_checkpoints.py"


class CheckpointExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.staging = self.base / "private_staging"
        self.checkpoints = self.staging / "checkpoints"
        self.inventory = self.base / "inventory.json"
        self.entries = []
        for index in range(51):
            relative = f"synthetic/seed_{index}.pt"
            content = f"synthetic checksum fixture {index:02d}\n".encode()
            path = self.checkpoints / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.entries.append({
                "published_relative_path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        self.write_inventory()

    def write_inventory(self):
        self.inventory.write_text(json.dumps({
            "mode": "inventory_only",
            "found_count": 51,
            "weights_publicly_licensed": "NOT_VERIFIED",
            "files": self.entries,
        }), encoding="utf-8")

    def verify(self):
        result = subprocess.run([
            sys.executable, str(VERIFIER), "--staging", str(self.staging),
            "--inventory", str(self.inventory),
        ], capture_output=True, text=True)
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        report, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        return result, report

    def first_path(self):
        return self.checkpoints / self.entries[0]["published_relative_path"]

    def assert_rejected(self, marker):
        result, report = self.verify()
        self.assertNotEqual(result.returncode, 0, report)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(marker in error for error in report["errors"]), report)

    def test_all_51_files_are_counted_and_accepted(self):
        result, report = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["verified"], 51)
        self.assertEqual(report["errors"], [])

    def test_missing_file_is_rejected(self):
        self.first_path().unlink()
        self.assert_rejected("Missing:")

    def test_same_size_corruption_is_rejected(self):
        path = self.first_path()
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        self.assert_rejected("SHA256 mismatch:")

    def test_wrong_size_is_rejected(self):
        self.first_path().write_bytes(b"truncated")
        self.assert_rejected("Size mismatch:")

    def test_unexpected_file_is_rejected(self):
        (self.checkpoints / "unexpected.pt").write_bytes(b"extra")
        self.assert_rejected("Unexpected file:")

    def test_duplicate_path_is_rejected(self):
        self.entries[1]["published_relative_path"] = self.entries[0]["published_relative_path"]
        self.write_inventory()
        self.assert_rejected("duplicate path:")

    def test_parent_traversal_is_rejected(self):
        self.entries[0]["published_relative_path"] = "../outside.pt"
        self.write_inventory()
        self.assert_rejected("Unsafe or duplicate path:")

    def test_symlink_escape_is_rejected(self):
        path = self.first_path()
        outside = self.base / "outside.pt"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlink unavailable: {error}")
        self.assert_rejected("Path escapes staging:")

    def test_exporter_copies_exactly_the_51_formal_paths(self):
        spec = importlib.util.spec_from_file_location("checkpoint_exporter", EXPORTER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = self.base / "original_project"
        self.assertEqual(len(list(module.pairs())), 51)
        for index, (_, relative) in enumerate(module.pairs()):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"formal-path synthetic fixture {index}".encode())
        (source / "data").mkdir()
        (source / "data" / "raw_private.txt").write_text("excluded")
        exported = self.base / "exported"
        result = subprocess.run([
            sys.executable, str(EXPORTER), "--source", str(source),
            "--destination", str(exported), "--include-weights",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        audit = json.loads((exported / "checkpoint_manifest.json").read_text())
        self.assertEqual(audit["found_count"], 51)
        self.assertFalse(audit["raw_datasets_exported"])
        self.assertFalse((exported / "data").exists())
        self.assertEqual(len(list((exported / "checkpoints").rglob("*.pt"))), 51)
        audit["mode"] = "inventory_only"
        self.inventory.write_text(json.dumps(audit), encoding="utf-8")
        self.staging = exported
        result, report = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["verified"], 51)

    def test_checkpoint_root_symlink_escape_is_rejected(self):
        outside = self.base / "outside_checkpoints"
        self.checkpoints.rename(outside)
        try:
            self.checkpoints.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlink unavailable: {error}")
        self.assert_rejected("Checkpoint root escapes staging")


if __name__ == "__main__":
    unittest.main()
