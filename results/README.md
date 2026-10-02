# Results

Everything here is produced by scripts in `experiments/` from the data in `data/` and from recorded matches.

| Path | What it is | Produced by |
|---|---|---|
| `matches/<run>/summary.json` | Score, Elo with 95% interval, and compute per move for one match | `experiments/series.py` |
| `matches/<run>/games.jsonl.gz` | Every game of that match: moves, result, visits per player | `experiments/series.py`, `package_results.py` |
| `matches/TABLE.md` | All matches in one table | `experiments/report.py` |
| `stopper/report.json` | Cross-fitted stopping-rule frontiers on the ladder dataset | `experiments/train_stopper.py` |
| `ladder/report.json` | Regret by budget and game phase, signal AUCs, hand-written rules | `experiments/analyze_ladder.py` |
| `reuse.json` | How often positions recur across games | `experiments/reuse_curve.py` |
| `approx_retrieval.json` | Whether near-matches transfer | `experiments/approx_retrieval.py` |
| `figures/` | Plots of the above | `experiments/figures.py`, `report.py` |

`matches/TABLE.md` is the authority for match numbers. If a commit message and the table ever disagree, the table is right.
