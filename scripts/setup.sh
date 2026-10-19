#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
uv venv --python 3.11 .venv
if [ -s results/environment.txt ]; then
    uv pip install --python .venv/bin/python -r results/environment.txt
else
    uv pip install --python .venv/bin/python -e './ocbench[mjwarp,train]' psutil
fi
uv pip freeze --python .venv/bin/python | sed "s@^-e .*@-e ./ocbench@" > results/environment.txt
