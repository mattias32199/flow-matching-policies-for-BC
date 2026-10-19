#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from wandb_mirror import ensure_credentials, MissingCredentials
try:
    ensure_credentials()
except MissingCredentials as error:
    raise SystemExit(str(error))
PY
# A dedicated server inherits the configured credential environment without
# embedding any key in command arguments or modifying the training server.
tmux -S "$REPO_ROOT/.local/tracking.sock" new-session -d -s ocbench-tracking -c "$REPO_ROOT" \
  'bash -c "source scripts/env.sh; exec .venv/bin/python -u scripts/wandb_mirror.py --watch > .local/wandb-mirror.log 2>&1"'
echo 'Tracking started; inspect .local/wandb-mirror.log for upload status.'
