#!/usr/bin/env bash
# Regenerate every table (paper/tables/*.tex), number (paper/tables/numbers.tex) and figure
# (paper/figures/*.pdf) in the report from the seeded generator, and check the numbers against
# paper/results.sha256. Requires the `paper` extra. Takes about ten minutes on 16 cores; WORKERS sets the number of
# processes.
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p figures tables
python make_data.py > /dev/null
python check_results.py
python figures.py
echo "Tables in paper/tables, figures in paper/figures."
