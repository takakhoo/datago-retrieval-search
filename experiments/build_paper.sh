#!/bin/sh
# Rebuild every result artifact and the paper from whatever matches have finished.
# Usage: experiments/build_paper.sh [user@host:/path/to/runs/series]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:experiments
if [ -n "$1" ]; then experiments/pull_runs.sh "$1"; fi
python3 experiments/package_results.py
python3 experiments/report.py > /dev/null
python3 experiments/figures.py --results results --out results/figures --policy "50,200,800,3200:sqrt" > /dev/null
for run in main_200 pilot2_stopper_200; do
  if [ -f "runs/series/$run/summary.json" ]; then
    python3 experiments/trace_profile.py "runs/series/$run/games.jsonl" --name main --out results > /dev/null
    echo "profile from $run"
    break
  fi
done
python3 experiments/paper_tables.py | tail -1
(cd paper && tectonic -X compile mikiri.tex 2>&1 | grep -E "^error|Writing" || true)
grep -c "??" paper/numbers.tex | xargs echo "numbers still pending:"
