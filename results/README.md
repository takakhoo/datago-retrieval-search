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
| `stopper/targets.json` | What the stopper should predict and how to price it | `experiments/targets.py` |
| `stopper/hybrid.json` | Prior rules' statistics as extra inputs to the stopper | `experiments/hybrid.py` |
| `baselines.json`, `baselines.txt` | Ten published stopping rules on the ladder dataset | `experiments/baselines.py` |
| `match_analysis.json` | Elo forecast from the ladder dataset against measured Elo, colour split | `experiments/match_analysis.py` |
| `figures/` | Plots of the above | `experiments/figures.py`, `report.py` |

`matches/TABLE.md` is the authority for match numbers. If a commit message and the table ever disagree, the table is right.

## Corrections

- **Six-rung ladder (`long_200`).** Reported as +263 Elo (+230 to +301) after 600 games. The match was extended to 1,000 games and stands at +228 (+203 to +254). The first 600 games ran above the long-run rate. The 1,000-game figure is the one used everywhere.
- **KataGo at 100 visits (`uni_100_vs_200`).** One early commit message quoted 68-320-12 before the run had finished. The recorded result is 77-301-22.
