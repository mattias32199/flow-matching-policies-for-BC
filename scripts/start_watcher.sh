#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
: "${CODEX_THREAD_ID:?Run this from the Codex thread to monitor}"
command -v codex >/dev/null
tmux -S "$REPO_ROOT/.local/watcher.sock" new-session -d -s ocbench-watcher -c "$REPO_ROOT" \
  'bash -c "source scripts/env.sh; exec .venv/bin/python -u scripts/watch_jobs.py > .local/watcher.log 2>&1"'
echo 'Completion watcher started; health is recorded in .local/watcher-status.json.'
