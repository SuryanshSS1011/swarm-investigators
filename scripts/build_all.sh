#!/usr/bin/env bash
# Rebuild every result from the raw outputs: score, pre-registered analysis, figures.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/study.py score
python3 scripts/analyze.py
python3 scripts/figures.py
