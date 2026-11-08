#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python code/run_cartpole.py "$@"
fi
exec uv run --frozen python code/run_cartpole.py "$@"
