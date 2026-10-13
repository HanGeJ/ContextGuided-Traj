#!/usr/bin/env python3
"""Independently SHA256-check the private checkpoint export against AutoDL inventory."""
import argparse
import hashlib
import json
from pathlib import Path

def sha256_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staging', required=True, type=Path,
                        help='Private export directory containing checkpoints/')
    parser.add_argument('--inventory', required=True, type=Path,
                        help='Original inventory_only checkpoint_manifest.json')
    a = parser.parse_args()
    d = json.loads(a.inventory.read_text(encoding='utf8'))
    if d.get('mode') != 'inventory_only':
        raise SystemExit('Require the original inventory_only manifest')
    rows = d.get('files')
    if not isinstance(rows, list) or len(rows) != 51 or d.get('found_count') != 51:
        raise SystemExit('Expected exactly 51 inventoried original checkpoints')
    base = a.staging.resolve() / 'checkpoints'
    failures = []
    seen = set()
    for entry in rows:
        rel = Path(entry['published_relative_path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:
            failures.append('Unsafe/duplicate relative path: ' + str(rel))
            continue
        seen.add(str(rel))
        path = base / rel
        if not path.is_file():
            failures.append('Missing: ' + str(rel))
            continue
        if path.stat().st_size != entry.get('size_bytes'):
            failures.append('Size mismatch: ' + str(rel))
            continue
        if sha256_file(path) != entry.get('sha256'):
            failures.append('SHA256 mismatch: ' + str(rel))
    unexpected = sorted(str(p.relative_to(base)) for p in base.rglob('*')
                        if p.is_file() and str(p.relative_to(base)) not in seen) if base.is_dir() else []
    failures.extend('Unexpected file: ' + p for p in unexpected)
    result = {'status': 'PASS' if not failures else 'FAIL',
              'expected': len(rows), 'verified': verified,
              'errors': failures, 'original_weight_rights': d.get('weights_publicly_licensed')}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)
    print('Integrity PASS. Weight loading, benchmark scores and public licensing remain unverified.')

if __name__ == '__main__':
    main()
