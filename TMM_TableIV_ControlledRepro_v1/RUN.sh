#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PYEXE="$(bash "$HERE/_python.sh")"
"$PYEXE" -u "$HERE/repro.py" "run" "$@"
