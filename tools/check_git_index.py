#!/usr/bin/env python3
"""Check actual Git index against the frozen, file-hashed source release manifest."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
if not isinstance(manifest, dict):
    raise SystemExit('Invalid source file hash manifest')
result = subprocess.run(
    ['git', '-C', str(ROOT), 'ls-files', '--cached', '-z'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
)
if result.returncode:
    print('FAIL: This folder is not a usable Git repository:',
          result.stderr.decode(errors='replace'))
    raise SystemExit(2)
tracked = {p.decode('utf-8') for p in result.stdout.split(b'\0') if p}
expected = set(manifest) | {'MANIFEST_SHA256.json'}
missing = sorted(expected - tracked)
extras = sorted(tracked - expected)
corrupted = []
for relative_path, expected_hash in manifest.items():
    path = ROOT / relative_path
    if not path.is_file():
        corrupted.append(relative_path + ': absent')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        corrupted.append(relative_path + ': hash mismatch')
status = 'PASS' if not (missing or extras or corrupted) else 'FAIL'
print(json.dumps({
    'status': status,
    'tracked_files': len(tracked),
    'expected_files': len(expected),
    'missing_from_git_index': missing,
    'unexpected_tracked': extras,
    'source_hash_errors': corrupted,
}, indent=2, ensure_ascii=False))
sys.exit(0 if status == 'PASS' else 1)
