#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${1:-ContextGuided-Traj}"
python tools/check_release.py
if ! command -v gh >/dev/null 2>&1; then
 echo 'GitHub CLI (gh) unavailable. Connect GitHub in ChatGPT or install gh on this machine.' >&2
 exit 3
fi
gh auth status
if [ ! -d .git ]; then git init -b main; fi
if ! git diff --cached --quiet; then echo 'Existing staging found'; fi
git add -A
python tools/check_git_index.py
if ! git diff --cached --quiet; then git commit -m 'Add source-only paper reproduction candidate'; fi
if ! git remote get-url origin >/dev/null 2>&1;then
 gh repo create "$REPO" --private --source=. --remote=origin --push
else
 git push -u origin main
fi
echo "PRIVATE_REPO_READY: $REPO (inspect and select software license before public release)"
