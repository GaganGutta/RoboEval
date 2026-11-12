#!/usr/bin/env bash
# Run a command inside the project venv from the repo root (used from WSL).
# Usage: bash scripts/wsl_run.sh python -m eval.replay_check --task can
set -e
source "$HOME/venvs/data_eval/bin/activate"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Silence robosuite's import-time warnings about optional extras we don't use.
export PYTHONWARNINGS=ignore
exec "$@"
