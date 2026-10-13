#!/usr/bin/env bash
# Select an interpreter with PyTorch, honoring a caller-supplied PYTHON override.
set -euo pipefail
if [[ -n "${PYTHON:-}" ]]; then printf '%s\n' "$PYTHON"; exit 0; fi
for cand in /root/miniconda3/envs/tmm_traj/bin/python /root/autodl-tmp/tmm_env/bin/python "$(command -v python || true)"; do
  [[ -n "$cand" && -x "$cand" ]] || continue
  if "$cand" -c 'import torch,numpy' >/dev/null 2>&1; then printf '%s\n' "$cand"; exit 0; fi
done
echo 'Could not locate Python with PyTorch. Set PYTHON=/path/to/python' >&2
exit 2
