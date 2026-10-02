# DataGo

DataGo wraps a frozen [KataGo](https://github.com/lightvector/KataGo) and changes one thing: how many visits each move gets. It stops searching once a decision is settled, keeps searching when it is not, and never searches the same position twice. The network and the search algorithm are untouched.

**Against KataGo at 200 visits per move, on 19x19, same network:**

| | Games | Score | Elo (95% interval) |
|---|---|---|---|
| DataGo at the same visits per move | {{main.games}} | {{main.score}}% | **{{main.elo}}** ({{main.elo_lo}} to {{main.elo_hi}}) |
| DataGo held to the same network evaluations ({{strict.visits}} visits) | {{strict.games}} | {{strict.score}}% | **{{strict.elo}}** ({{strict.elo_lo}} to {{strict.elo_hi}}) |
| KataGo with twice the visits, for scale | {{unidouble.games}} | {{unidouble.score}}% | {{unidouble.elo}} ({{unidouble.elo_lo}} to {{unidouble.elo_hi}}) |
| KataGo against itself, as a check | {{unisame.games}} | {{unisame.score}}% | {{unisame.elo}} ({{unisame.elo_lo}} to {{unisame.elo_hi}}) |

Paper: [`paper/datago.pdf`](paper/datago.pdf). Every game behind these numbers is in [`results/matches/`](results/matches/).

## Demo

![A DataGo game against KataGo, with the visits each move cost](results/demo/game.gif)

One real game from the main match. Blue bars are DataGo's visits per move, orange is KataGo's fixed 200. DataGo coasts through the opening, thinks hard while the game is in the balance, and coasts again once it is decided.

Replay it move by move in a terminal, with no GPU and no KataGo:

```bash
pip install -e .
python -m datago.demo --replay results/demo/game.json
```

```
{{demo_excerpt}}
```

To play live games you need a KataGo binary and network (see [Reproduce](#reproduce)):

```bash
python -m datago.demo --net b18 --stopper models/stopper_b18.json --games 2
```

## How it works

1. **A visit ladder with a learned stop.** DataGo searches at 50 visits and asks a small model how much winrate is still at stake. If the answer is too little to justify the next rung, it plays. Otherwise it searches at 200, then 800, then 3,200. The model is a boosted-tree regressor over statistics the search already produced (how contested the root is, how the search disagrees with the network, how close the game is). It is stored as JSON and runs in NumPy.
2. **Exact search memory.** Finished searches are stored under a key that is the same for all eight rotations and reflections of a position and for any move order. Meeting the position again costs zero visits.
3. **A ledger.** DataGo is granted the baseline's budget for every move and a controller keeps its total spending at that grant. Visits saved on easy or remembered positions go to hard ones.

![Winrate given up per move against visits](results/figures/frontier.png)

The stopping rule was trained and tested on a dataset built for this project: 6,000 positions, each searched at seven budgets from 50 to 3,200 visits, with every candidate move scored by a separate forced search. At a mean cost of 200 visits the stopper gives up as little winrate as uniform search at 358 visits.

## Results

Compute is reported three ways because the answer depends on how you count. **Visits** is the size of the deepest search per move. **Restart visits** charges every search DataGo requested, including the smaller ones it later extended. **NN evals** is the number of network evaluations each player's own KataGo process actually ran.

{{table}}

![Elo against compute, counted three ways](results/figures/elo_vs_compute.png)

## What we learned

- **Search error is rare and concentrated.** At 200 visits, 71% of positions have no regret and the worst 10% hold 84% of it. Almost none of it is in the first 40 moves.
- **The stopping rule matters more than the model.** Dividing predicted regret by the cost of the next rung is what makes stopping pay. With a flat threshold, v1's entropy gate is worse than no gate at all (0.63x). With the rate rule, even a hand-written confidence margin reaches 1.45x. The learned model reaches 1.8x.
- **Most of the remaining headroom is invisible to the search.** A stopper that knew the true regret would reach 12x. More than half of the regret that survives sits on moves the shallow search gave under 10% of its visits.
- **Exact positions recur a lot.** In self-play, 11.5% of all positions had occurred in an earlier game up to symmetry, and half of the positions at move 20 had.
- **Approximate retrieval does not transfer.** When the shallow and deep search disagree, a similar stored position has the right move 5.5% of the time. The policy network's own second choice has it 43% of the time. Only the same position transfers.

## What happened to v1

The first version of this project reported 9-0-1 and 8-0-2 records against KataGo. Those came from a bug: its match runner analysed positions with a GTP command that also plays a move, so every "deep search" put an extra black stone on the board. [`legacy/AUDIT.md`](legacy/AUDIT.md) explains it and [`legacy/audit/reproduce_double_move.py`](legacy/audit/reproduce_double_move.py) reproduces it with v1's own code. v1 is preserved unchanged in [`legacy/v1/`](legacy/v1/). None of its numbers are used here.

## Reproduce

```bash
pip install -e ".[dev,experiments]"
python -m pytest -q        # 26 tests, toy engine, no GPU
```

For real runs, set two environment variables: `KATAGO` (path to a KataGo 1.18 binary) and `DG` (a directory containing `nets/` with the networks named in [`datago/engine.py`](datago/engine.py)).

| Step | Command | GPU time (one RTX 6000 Ada) |
|---|---|---|
| Self-play positions | `python experiments/collect.py games --n 300 --out runs/data/games19.jsonl` | 15 min |
| Ladder dataset | `python experiments/collect.py label --games runs/data/games19.jsonl --out runs/data/ladder19.jsonl` | 1.5 h |
| Train the stopper | `python experiments/train_stopper.py data/ladder19.jsonl.gz --export 50,200,800,3200:sqrt` | none |
| Main match | `python experiments/series.py --name main_200 --games 1000 --budget 200 --path 50,200,800,3200 --stopper models/stopper_b18.json --control` | 1.5 h |
| Tables, figures, paper | `experiments/build_paper.sh` | none |

The dataset and the trained model are already in [`data/`](data/) and [`models/`](models/), so the first three steps are optional.

## Layout

```
datago/        the engine wrapper: board, KataGo client, stopper, memory, players, matches
experiments/   scripts behind every number, table, and figure
data/          the ladder dataset and the games it was drawn from
models/        trained stopping models (JSON)
results/       match records, analysis reports, figures
paper/         LaTeX source and PDF
legacy/        v1, unchanged, and the audit of its results
tests/         26 tests that run the whole pipeline against a toy engine
```

## Limits

These results are for KataGo networks at 100 to 800 visits per move, against KataGo itself. Elo per doubling of visits is steep in that range, which is what makes reallocating visits valuable. Tournament-scale budgets are untested. Memory hit rates depend on the opponent's opening variety. AlphaGo is not available to play, so there is no direct comparison with it.

## Credits

v2 (this rebuild, the experiments, and the paper) is by Taka Khoo. The first version, preserved in [`legacy/v1/`](legacy/v1/), was a team project by Benjamin Huh, Jason Peng, Taka Khoo, Olir Eswaramoorthy, David Roos, and Victor Lun Pun at the Thayer School of Engineering, Dartmouth College. Experiments ran on the Dartmouth LISP Lab's machines. KataGo and its networks are the work of David J. Wu and the KataGo community.
