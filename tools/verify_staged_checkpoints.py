#!/usr/bin/env python3
"""Verify privately exported checkpoint file sizes and SHA256 against the frozen AutoDL manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", required=True, type=Path,
                        help="Private export directory containing checkpoints/")
    parser.add_argument("--inventory", required=True, type=Path,
                        help="Original inventory_only checkpoint_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.inventory.read_text(encoding="utf-8"))
    if manifest.get("mode") != "inventory_only":
        raise SystemExit("Require the original inventory_only manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 51 or manifest.get("found_count") != 51:
        raise SystemExit("Expected exactly 51 inventoried original checkpoints")

    staging = args.staging.resolve()
    base = (staging / "checkpoints").resolve()
    errors = []
    verified = 0
    seen = set()
    root_escapes = not base.is_relative_to(staging)
    if root_escapes:
        errors.append("Checkpoint root escapes staging")
    for entry in ([] if root_escapes else entries):
        rel = Path(entry["published_relative_path"])
        if rel.is_absolute() or ".." in rel.parts or str(rel) in seen:
            errors.append("Unsafe or duplicate path: " + str(rel))
            continue
        seen.add(str(rel))
        file_path = base / rel
        if not file_path.resolve().is_relative_to(base):
            errors.append("Path escapes staging: " + str(rel))
            continue
        if not file_path.is_file():
            errors.append("Missing: " + str(rel))
        elif file_path.stat().st_size != entry.get("size_bytes"):
            errors.append("Size mismatch: " + str(rel))
        elif sha256_file(file_path) != entry.get("sha256"):
            errors.append("SHA256 mismatch: " + str(rel))
        else:
            verified += 1

    if not root_escapes and base.is_dir():
        for p in base.rglob("*"):
            if p.is_file() and str(p.relative_to(base)) not in seen:
                errors.append("Unexpected file: " + str(p.relative_to(base)))
    elif not root_escapes:
        errors.append("Staging checkpoints directory does not exist")

    result = {"status": "PASS" if not errors and verified == 51 else "FAIL",
              "expected": len(entries), "verified": verified, "errors": errors,
              "weight_redistribution_rights": manifest.get("weights_publicly_licensed")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)
    print("Integrity PASS. Model loading, paper metrics, and redistribution rights still require review.")


if __name__ == "__main__":
    main()
