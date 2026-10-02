# DataGo

[![tests](https://github.com/takakhoo/datago-retrieval-search/actions/workflows/tests.yml/badge.svg)](https://github.com/takakhoo/datago-retrieval-search/actions/workflows/tests.yml)

**A Go engine that knows when to stop thinking.** DataGo wraps a frozen [KataGo](https://github.com/lightvector/KataGo) and changes one thing: how many search visits each move gets. It plays settled moves quickly, keeps searching when the game is in the balance, and never searches the same position twice. The network and the search algorithm are untouched.

![A DataGo game against KataGo, with the visits each move cost](results/demo/game.gif)

*One real game from the main match. Blue bars are DataGo's visits per move. KataGo spends a fixed 200 on every move.*

## Headline

Against KataGo at 200 visits per move, 19x19, same network:

| | Games | Score | Elo (95% interval) |
|---|---|---|---|
| **DataGo, same visits per move** | {{main.games}} | {{main.score}}% | **{{main.elo}}** ({{main.elo_lo}} to {{main.elo_hi}}) |
| **DataGo, same network evaluations** ({{strict.visits}} visits per move) | {{strict.games}} | {{strict.score}}% | **{{strict.elo}}** ({{strict.elo_lo}} to {{strict.elo_hi}}) |
| KataGo with twice the visits, for scale | {{unidouble.games}} | {{unidouble.score}}% | {{unidouble.elo}} ({{unidouble.elo_lo}} to {{unidouble.elo_hi}}) |
| KataGo against itself, as a check | {{unisame.games}} | {{unisame.score}}% | {{unisame.elo}} ({{unisame.elo_lo}} to {{unisame.elo_hi}}) |

Paper: [`paper/datago.pdf`](paper/datago.pdf). Every game behind every number is in [`results/matches/`](results/matches/).

## Contents

- [How it works](#how-it-works)
- [The science, step by step](#the-science-step-by-step)
- [Match results](#match-results)
- [Try it](#try-it)
- [What happened to v1](#what-happened-to-v1)
- [Reproduce](#reproduce)

## How it works

![How DataGo decides one move](results/figures/pipeline.png)

1. **Memory first.** If this exact position has been searched before, in any rotation or reflection and by any move order, DataGo plays from the stored search and spends nothing.
2. **A ladder of budgets.** Otherwise it searches at 50 visits and asks a small model whether the decision is settled. If not, it searches at 200, then 800, then 3,200.
3. **A stopping rule with a price in it.** The model predicts how much winrate the current choice might still be giving up. DataGo stops when that stake, divided by the visits needed to reach the next rung, falls below a threshold.
4. **A ledger.** DataGo is granted the baseline's budget for every move, and a controller keeps its total spending at that grant. Visits saved on easy or remembered positions are spent on hard ones.

## The science, step by step

### 1. Search error is rare, and it is concentrated

We built a dataset to measure what more search buys: 6,000 positions from self-play, each searched at seven budgets from 50 to 3,200 visits. A search cannot grade itself, so every move any budget chose was scored by a separate search forced to start with that move. The gap to the best scored move is that budget's **regret**, in winrate.

![Regret is concentrated in a few positions and comes after the opening](results/figures/regret_anatomy.png)

At 200 visits, 71% of positions have no regret at all, and the worst 10% hold 84% of it. Almost none of it is in the first 40 moves. A fixed budget spends most of its visits where they change nothing.

### 2. One position, up close

![A position where thinking longer changed the move](results/figures/case_study.png)

This is a real position from the dataset. At 50 and at 200 visits the engine plays A. The stopper's estimate of what is at stake stays far above its threshold, so DataGo keeps going, and at 800 visits the engine finds B. An independent search rates B about 50 points of winrate better than A. A fixed 200-visit player gives all of that away.

### 3. The stopping rule

DataGo stops at rung $j$ when

$$\frac{\hat r_j(x)}{v_{j+1}-v_j} < \lambda$$

where $\hat r_j(x)$ is the predicted regret left at this rung, $v_{j+1}-v_j$ is the number of visits the next rung would add, and $\lambda$ is the exchange rate between winrate and visits. One $\lambda$ prices every decision on the ladder the same way: going from 800 to 3,200 visits needs four times the stake that going from 200 to 800 does.

That division is the most important design choice in the project.

![The same stopping signals with and without the rate rule](results/figures/rate_rule.png)

The bars show how many times more visits uniform search needs to match each rule (above 1.0 is a gain). With a flat threshold, the entropy gate from v1 of this project is worse than no gate, and random stopping is far worse. With the rate rule, even a hand-written confidence margin beats uniform search by 1.45x, and the learned model reaches 1.82x.

### 4. Training the stopper

The model is a gradient-boosted tree regressor on statistics the search already produced: how contested the root is, how the search disagrees with the network's prior, how close the game is, and how the answer changed since the previous rung. It is fitted to the square root of regret, which keeps a handful of enormous blunders from dominating. It is stored as JSON and evaluated in NumPy, with no pickle and no ML framework at play time.

![Training curves for the stopper](results/figures/training.png)

Left: error on held-out games levels off by the 150 trees we deploy. Right: {{training_sentence}}

![Calibration and feature importance](results/figures/stopper_diagnostics.png)

Left: predicted and realized regret agree on games the model never saw. Right: the model leans on a few signals. Ablations agree: seven features do nearly as well as all of them, and 80 depth-2 trees do as well as 400 depth-3 trees. The rule matters more than the model.

### 5. What it buys, offline

![Winrate given up per move against visits](results/figures/frontier.png)

At a mean cost of 200 visits the stopper gives up as little winrate as uniform search at 358 visits, a factor of 1.8. The factor stays between 1.7 and 1.9 from 100 to 800 visits (1.3 to 1.5 if every restarted search is charged in full). All of it is cross-fitted: each position is scored by a model that never saw its game.

The dashed line is a stopper that knows the true regret. It reaches 12x. Most of that gap is invisible to any rule that reads search statistics: over half of the surviving regret sits on moves the shallow search gave less than 10% of its visits.

### 6. Memory: exact positions only

![Eight orientations of a position share one memory key](results/figures/symmetry.png)

Stored searches are keyed on a canonical form of the position, so rotations, reflections, and transpositions all hit the same entry. Positions recur more than you might expect:

![How often positions recur across games](results/figures/reuse.png)

Across 300 self-play games, a position at move 10 had been seen before 86% of the time and at move 20 half the time. Overall 11.5% of all positions were repeats.

The original idea behind this project was *approximate* retrieval: find similar stored positions and reuse their analysis. We tested it directly and it does not work for a frozen engine.

![Stored searches transfer only to the same position](results/figures/similarity.png)

When the shallow and deep search disagree, the most similar stored position has the right move 5.5% of the time. The policy network's own second choice has it 43% of the time. Transfer is reliable only when the stored position is the same position, which the exact key already finds.

## Match results

Offline regret is a proxy. These are played games, from the empty board, alternating colors, each player on its own KataGo process.

Compute is reported three ways because the answer depends on how you count. **Visits** is the size of the deepest search per move. **Restart visits** charges every search DataGo requested, including smaller ones it later extended. **NN evals** is the number of network evaluations each player's KataGo process actually ran.

{{table}}

![Elo against compute, counted three ways](results/figures/elo_vs_compute.png)

The orange line is KataGo at uniform budgets. DataGo's points sit above it under all three ways of counting.

![Where DataGo spends its visits](results/figures/profile_main.png)

Nobody told DataGo to save in the opening or to think when the game is close. Both fall out of predicting regret in winrate units.

## Try it

Replay the demo game move by move in a terminal. No GPU, no KataGo:

```bash
pip install -e .
python -m datago.demo --replay results/demo/game.json
```

```
{{demo_excerpt}}
```

Watch the mechanics against a toy engine, or play live against a real KataGo:

```bash
python -m datago.demo --fake --games 2
python -m datago.demo --net b18 --stopper models/stopper_b18.json --games 2
```

## What happened to v1

<img src="results/figures/v1_bug.png" width="330" align="right" alt="The v1 bug: three black stones in one turn">

The first version of this project reported 9-0-1 and 8-0-2 records against KataGo. Those came from a bug. Its match runner analysed positions with a GTP command that also plays a move, so the standard search, the deep search, and each recursive search all placed a stone before White replied. Retrieved moves were never sent to the engine at all.

[`legacy/AUDIT.md`](legacy/AUDIT.md) walks through it, and [`legacy/audit/reproduce_double_move.py`](legacy/audit/reproduce_double_move.py) reproduces it with v1's own code: three black stones after one Black turn. v1 is preserved unchanged in [`legacy/v1/`](legacy/v1/). None of its numbers are used here.

<br clear="right">

## Reproduce

```bash
pip install -e ".[dev,experiments]"
python -m pytest -q        # 26 tests against a toy engine, no GPU
```

For real runs set `KATAGO` (path to a KataGo 1.18 binary) and `DG` (a directory with `nets/` holding the networks named in [`datago/engine.py`](datago/engine.py)).

| Step | Command | GPU time (one RTX 6000 Ada) |
|---|---|---|
| Self-play positions | `python experiments/collect.py games --n 300 --out runs/data/games19.jsonl` | 15 min |
| Ladder dataset | `python experiments/collect.py label --games runs/data/games19.jsonl --out runs/data/ladder19.jsonl` | 1.5 h |
| Train the stopper | `python experiments/train_stopper.py data/ladder19.jsonl.gz --export 50,200,800,3200:sqrt` | none |
| Main match | `python experiments/series.py --name main_200 --games 1000 --budget 200 --path 50,200,800,3200 --stopper models/stopper_b18.json --control` | 1.5 h |
| Tables, figures, paper | `experiments/build_paper.sh` | none |

The dataset and trained model ship in [`data/`](data/) and [`models/`](models/), so the first three steps are optional.

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

- Results are for KataGo networks at 100 to 800 visits per move, against KataGo itself. Elo per doubling of visits is steep in that range, which is what makes moving visits around valuable. Tournament-scale budgets are untested.
- Regret is measured by a search, so it inherits the engine's blind spots. The match results do not depend on it.
- Memory hit rates depend on how varied the opponent's openings are.
- AlphaGo is not available to play, so there is no direct comparison with it. KataGo is among the strongest engines anyone can run.

## Credits

v2 (this rebuild, the experiments, and the paper) is by Taka Khoo. The first version, preserved in [`legacy/v1/`](legacy/v1/), was a team project by Benjamin Huh, Jason Peng, Taka Khoo, Olir Eswaramoorthy, David Roos, and Victor Lun Pun at the Thayer School of Engineering, Dartmouth College. Experiments ran on the Dartmouth LISP Lab's machines. KataGo and its networks are the work of David J. Wu and the KataGo community.
