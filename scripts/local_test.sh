#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-.venv}"

python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install -U pip
pip install .[streamlit]
pytest
