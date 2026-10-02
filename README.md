# DataGo

DataGo wraps a frozen [KataGo](https://github.com/lightvector/KataGo) and changes one thing: how many visits each move gets. It stops searching early when a decision is settled, keeps searching when it is not, and never searches the same position twice.

> **Status (2 Oct 2026): v2 rebuild in progress.** The match experiments are running now and this page will be replaced with the full results when they finish. Everything stated below is already measured and reproducible from this repository.

## What is established so far

- **The v1 results were an artifact.** The 9-0-1 and 8-0-2 records in the first version came from a match runner that gave DataGo extra moves. [`legacy/AUDIT.md`](legacy/AUDIT.md) explains the bug and reproduces it. v1 is kept unchanged under [`legacy/v1/`](legacy/v1/).
- **A learned stopping rule gets 1.7 to 1.9 times more out of each visit, offline.** On 6,000 positions searched at seven budgets, DataGo's stopper at a mean cost of 200 visits loses as little winrate per move as uniform search at 358 visits. Cross-fitted by game. See [`results/stopper/`](results/stopper/) and the figure below.
- **Exact memory has more to serve than expected.** Across 300 self-play games, 11% of all positions had already occurred in an earlier game up to board symmetry, and half of the positions at move 20 had.
- **Approximate retrieval does not transfer.** A stored search of a similar position points at the right move about 5% of the time when it matters. The policy network's own second choice is right 43% of the time. Only the same position transfers.

![Regret against visits for uniform search, the learned stopper, and an oracle](results/figures/frontier.png)

## Try it

```bash
pip install -e ".[dev,experiments]"
python -m pytest -q                      # full pipeline against a toy engine, no GPU
python -m datago.demo --fake --games 2   # watch the ladder, stopper, and memory work
```

Real play needs a KataGo binary and network. See [`experiments/`](experiments/) for the scripts behind every number, and [`results/README.md`](results/README.md) for what each result file is.

## Credits

v2 (this rebuild, the experiments, and the paper in [`paper/`](paper/)) is by Taka Khoo. The first version, preserved in [`legacy/v1/`](legacy/v1/), was a team project by Benjamin Huh, Jason Peng, Taka Khoo, Olir Eswaramoorthy, David Roos, and Victor Lun Pun at the Thayer School of Engineering, Dartmouth College.
