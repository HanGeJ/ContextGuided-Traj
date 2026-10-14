#!/usr/bin/env python3
"""Verify staged Git blobs against the staged source hash manifest."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(
    ['git', '-C', str(ROOT), 'ls-files', '--cached', '-z'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
)
if result.returncode:
    print('FAIL: This folder is not a usable Git repository:',
          result.stderr.decode(errors='replace'))
    raise SystemExit(2)
manifest_result = subprocess.run(
    ['git', '-C', str(ROOT), 'show', ':MANIFEST_SHA256.json'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
)
if manifest_result.returncode:
    print('FAIL: Stage MANIFEST_SHA256.json before checking the Git index.')
    raise SystemExit(2)
manifest = json.loads(manifest_result.stdout.decode('utf-8'))
if not isinstance(manifest, dict):
    raise SystemExit('Invalid source file hash manifest')
tracked = {p.decode('utf-8') for p in result.stdout.split(b'\0') if p}
expected = set(manifest) | {'MANIFEST_SHA256.json'}
missing = sorted(expected - tracked)
extras = sorted(tracked - expected)
corrupted = []
for relative_path, expected_hash in manifest.items():
    blob = subprocess.run(
        ['git', '-C', str(ROOT), 'show', ':' + relative_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if blob.returncode:
        corrupted.append(relative_path + ': absent')
    elif hashlib.sha256(blob.stdout).hexdigest() != expected_hash:
        corrupted.append(relative_path + ': hash mismatch')
status = 'PASS' if not (missing or extras or corrupted) else 'FAIL'
print(json.dumps({
    'status': status,
    'tracked_files': len(tracked),
    'expected_files': len(expected),
    'missing_from_git_index': missing,
    'unexpected_tracked': extras,
    'source_hash_errors': corrupted,
    'hash_basis': 'Git index blob bytes',
}, indent=2, ensure_ascii=False))
sys.exit(0 if status == 'PASS' else 1)
