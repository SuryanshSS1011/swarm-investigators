#!/usr/bin/env bash
# Rebuild every result from the raw outputs: score, pre-registered analysis, secondary analyses, replication, E3, E4,
# E5, the post hoc robustness checks and the figures.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/study.py score
python3 scripts/analyze.py
python3 scripts/analyze_secondary.py
python3 scripts/analyze_replication.py
python3 scripts/analyze_reasoning.py
python3 scripts/analyze_adaptive.py > /dev/null
python3 scripts/analyze_effects_all.py
python3 scripts/analyze_robustness.py
python3 scripts/figures.py
python3 scripts/fig_key_results.py > /dev/null
