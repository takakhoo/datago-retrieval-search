# Mikiri

[![tests](https://github.com/takakhoo/mikiri-beats-katago/actions/workflows/tests.yml/badge.svg)](https://github.com/takakhoo/mikiri-beats-katago/actions/workflows/tests.yml)

**Mikiri makes KataGo {{long.elo_abs}} Elo stronger at the same search budget, nearly as much as KataGo gains from doubling its search. Same network, same search algorithm, no retraining.**

KataGo is the strongest open engine in the AlphaGo family: a policy and value network guiding a tree search. Like every engine in that family, it gives each move the same number of search visits. Mikiri wraps it and decides, move by move, when the search has seen enough. It plays settled moves at a quarter of the budget, keeps searching when the game is on the line, and never searches the same position twice.

Mikiri (見切り) is Japanese for the judgment that you have seen enough.

![A Mikiri game against KataGo, with the visits each move cost](results/demo/game.gif)

*One real game from the main match. Blue bars are Mikiri's visits per move. KataGo spends a fixed 200 on every move. At move 137 Mikiri climbs the whole ladder to 3,200 visits and sees its winrate estimate fall from 65% to 19%. It keeps thinking hard for the next few moves and goes on to win. Over the whole game both sides averaged about 200 visits per move.*

## The result

{{total_games}} recorded games against KataGo on 19x19, both sides running the same network:

| Against KataGo at 200 visits per move | Games | Record (W-L-D) | Elo gain (95% interval) |
|---|---|---|---|
| **Mikiri at the same 200 visits, six-rung ladder** | {{long.games}} | {{long.wld}} | **{{long.elo}}** ({{long.elo_lo}} to {{long.elo_hi}}) |
| **Mikiri at the same 200 visits, four-rung ladder** | {{main.games}} | {{main.wld}} | **{{main.elo}}** ({{main.elo_lo}} to {{main.elo_hi}}) |
| **Mikiri at {{strict.visits}} visits, 20% fewer** (four-rung ladder) | {{strict.games}} | {{strict.wld}} | **{{strict.elo}}** ({{strict.elo_lo}} to {{strict.elo_hi}}) |
| **Mikiri with move sampling switched off** (paired openings, four-rung ladder) | {{paired.games}} | {{paired.wld}} | **{{paired.elo}}** ({{paired.elo_lo}} to {{paired.elo_hi}}) |
| **Mikiri with four-rung ladder and memory** | {{full.games}} | {{full.wld}} | **{{full.elo}}** ({{full.elo_lo}} to {{full.elo_hi}}) |
| KataGo given twice the visits, for scale | {{unidouble.games}} | {{unidouble.wld}} | {{unidouble.elo}} ({{unidouble.elo_lo}} to {{unidouble.elo_hi}}) |
| KataGo against itself, as a check on the harness | {{unisame.games}} | {{unisame.wld}} | {{unisame.elo}} ({{unisame.elo_lo}} to {{unisame.elo_hi}}) |

- **At equal visits, Mikiri wins {{long.score}}% of the points** with its six-rung ladder (50, 200, 400, 800, 1,600, 3,200) and {{main.score}}% with the four-rung ladder (50, 200, 800, 3,200). The stopping rule is worth nearly a doubling of KataGo's search: {{long.elo}} and {{main.elo}} Elo, against {{unidouble.elo}} for twice the visits.
- **Against KataGo with twice the visits, it holds even.** Mikiri at 200 visits per move against KataGo at 400: {{half.wld}} over {{half.games}} games ({{half.elo}} Elo, {{half.elo_lo}} to {{half.elo_hi}}), running {{half.evals}} network evaluations per move to KataGo's {{half.base_evals}}.
- **With 20% fewer visits it still wins {{strict.score}}%.**
- **It holds up under the strictest accounting.** Counting actual network evaluations, Mikiri at {{long.evals}} per move gains {{long.elo}} Elo. KataGo at {{uniroot.evals}} per move (283 visits) gains {{uniroot.elo}}, and at {{unidouble.evals}} per move (400 visits) gains {{unidouble.elo}}.
- **It is not an artifact of move sampling.** With both sides always playing their top move from 400 balanced openings, the gain is {{paired.elo}}.

**Status.** This work is being prepared for submission to **IJCAI 2027** (paper deadline 11 January 2027). It has not been submitted or peer reviewed. The short version in IJCAI format is in [`paper/ijcai/`](paper/ijcai/), and the full technical report is [`paper/mikiri.pdf`](paper/mikiri.pdf). Every game behind every number is in [`results/matches/`](results/matches/).

## Contents

- [How it works](#how-it-works)
- [The science, step by step](#the-science-step-by-step)
- [Against prior stopping rules](#against-prior-stopping-rules)
- [Match results](#match-results)
- [Try it](#try-it)
- [What happened to v1](#what-happened-to-v1)
- [Reproduce](#reproduce)

## How it works

![How Mikiri decides one move](results/figures/pipeline.png)

1. **Memory first.** If this exact position has been searched before, in any rotation or reflection and by any move order, Mikiri plays from the stored search and spends nothing.
2. **A ladder of budgets.** Otherwise it searches at 50 visits and asks a small model whether the decision is settled. If not, it searches at 200, then 800, then 3,200.
3. **A stopping rule with a price in it.** The model predicts how much winrate the current choice might still be giving up. Mikiri stops when that stake, divided by the visits needed to reach the next rung, falls below a threshold.
4. **A ledger.** Mikiri is granted the baseline's budget for every move, and a controller keeps its total spending at that grant. Visits saved on easy or remembered positions are spent on hard ones.

## The science, step by step

### 1. Search error is rare, and it is concentrated

We built a dataset to measure what more search buys: 6,000 positions from self-play, each searched at seven budgets from 50 to 3,200 visits. A search cannot grade itself, so every move any budget chose was scored by a separate search forced to start with that move. The gap to the best scored move is that budget's **regret**, in winrate.

![Regret is concentrated in a few positions and comes after the opening](results/figures/regret_anatomy.png)

At 200 visits, 71% of positions have no regret at all, and the worst 10% hold 84% of it. Almost none of it is in the first 40 moves. A fixed budget spends most of its visits where they change nothing.

### 2. One position, up close

![A position where thinking longer changed the move](results/figures/case_study.png)

This is a real position from the dataset. At 50 and at 200 visits the engine plays A. The stopper's estimate of what is at stake stays far above its threshold, so Mikiri keeps going, and at 800 visits the engine finds B. An independent search rates B about 50 points of winrate better than A. A fixed 200-visit player gives all of that away.

### 3. The stopping rule

Mikiri stops at rung $j$ when

$$\frac{\hat r_j(x)}{v_{j+1}-v_j} < \lambda$$

where $\hat r_j(x)$ is the predicted regret left at this rung, $v_{j+1}-v_j$ is the number of visits the next rung would add, and $\lambda$ is the exchange rate between winrate and visits. One $\lambda$ prices every decision on the ladder the same way: going from 800 to 3,200 visits needs four times the stake that going from 200 to 800 does.

That division is the most important design choice in the project.

![The same stopping signals with and without the rate rule](results/figures/rate_rule.png)

The bars show how many times more visits uniform search needs to match each rule (above 1.0 is a gain). With a flat threshold, the entropy gate from v1 of this project is worse than no gate, and random stopping is far worse. With the rate rule, even a hand-written confidence margin beats uniform search by 1.45x, and the learned model reaches 1.82x.

### 4. Training the stopper

The model is a gradient-boosted tree regressor on statistics the search already produced: how contested the root is, how the search disagrees with the network's prior, how close the game is, and how the answer changed since the previous rung. It is fitted to the square root of regret, which keeps a handful of enormous blunders from dominating. It is stored as JSON and evaluated in NumPy, with no pickle and no ML framework at play time.

![Training curves for the stopper](results/figures/training.png)

Left: error on held-out games levels off after about 100 trees. We deploy 150. Right: {{training_sentence}}

![Calibration and feature importance](results/figures/stopper_diagnostics.png)

Left: on games the model never saw, positions it scores higher really do carry more regret, from 0.02% of winrate in the lowest tenth to 4.2% in the highest. Right: the model leans mostly on one signal, the product of how undecided the game is and how contested the move is. Ablations agree: seven features do nearly as well as all of them, and 80 depth-2 trees do as well as 400 depth-3 trees. The rule matters more than the model.

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

In play, memory alone (no stopper) served {{memonly.hit}}% of Mikiri's moves over {{memonly.games}} games with no change in strength ({{memonly.elo}} Elo, {{memonly.elo_lo}} to {{memonly.elo_hi}}). It saves search visits. It does not save network evaluations ({{memonly.evals}} per move against {{memonly.base_evals}}), because KataGo already caches evaluations of positions it has seen. The strength comes from the stopping rule.

With memory and the four-rung stopper together, the controller reinvests the visits that memory saves. Over {{full.games}} games that system gains {{full.elo}} Elo ({{full.elo_lo}} to {{full.elo_hi}}), with {{full.hit}}% of its moves served from memory, against {{main.elo}} for the stopper alone. The two intervals overlap, and the combined system runs more network evaluations ({{full.evals}} per move against {{main.evals}}), so memory adds little on top of the stopper.

## Against prior stopping rules

We re-implemented ten published rules for deciding how long to search, from their papers and source code, and ran them on the same dataset with the same scoring. Each threshold was swept so every rule is shown at its best. Numbers are how many times more visits uniform search needs to match the rule at a 200-visit mean budget (above 1.0 is a gain), with 95% intervals.

{{baselines_table}}

Two things stand out. Mikiri is ahead of the strongest prior rule, V-MCTS at its published settings, by {{paired_vmcts}} in a paired test over the same games. And Mikiri's rate rule improves other people's signals too: it lifts the DS-MCTS-style classifier from {{ds_flat}}x to {{ds_rate}}x.

One caveat on accounting: the table charges every rule the size of its deepest search. Mikiri restarts its search at each rung, and if every restart is charged in full its four-rung ladder comes to 1.44x, below V-MCTS, which stops inside one search and has nothing to restart. On fresh engines a restarted search cost about the same network evaluations as one continuous search, and the head-to-head matches below report evaluations for both players.

Two follow-up experiments explain where the gap comes from.

- **The prior rules do not see anything Mikiri misses.** Feeding their statistics (KLD gain, VOI bound, BAI gap, V-MCTS distance) to Mikiri's model as extra inputs, one at a time or all together, moves the multiplier from 2.14x to between 2.07x and 2.17x, with no gain distinguishable from zero ([`results/stopper/hybrid.json`](results/stopper/hybrid.json)). The rules differ in what they do with the search statistics.
- **Predicting the regret that remains, and dividing by the cost of the next rung, beats every alternative we tried.** In theory the right score is the best recovery per visit over any continuation, and an oracle that reads it matches the hindsight optimum (14.1x against 14.2x). Learned from search statistics it reaches 1.63x to 1.87x. Remaining regret under the rate rule reaches 1.83x to 2.14x, and with no price only 1.41x to 1.56x ([`results/stopper/targets.json`](results/stopper/targets.json)).

These are re-implementations from root search statistics on KataGo, with the adaptations listed in the paper.

### Head to head, in real games

The two strongest prior rules were packaged into the same player and held to the same 200 visits per move by the same controller.

| Against KataGo at 200 visits | Games | Elo gain (95% interval) | Network evaluations per move (it : KataGo) |
|---|---|---|---|
| V-MCTS rule, published settings | {{vmcts.games}} | {{vmcts.elo}} ({{vmcts.elo_lo}} to {{vmcts.elo_hi}}) | {{vmcts.evals}} : {{vmcts.base_evals}} |
| DS-MCTS-style rule | {{dsmcts.games}} | {{dsmcts.elo}} ({{dsmcts.elo_lo}} to {{dsmcts.elo_hi}}) | {{dsmcts.evals}} : {{dsmcts.base_evals}} |
| Mikiri, four-rung ladder | {{main.games}} | {{main.elo}} ({{main.elo_lo}} to {{main.elo_hi}}) | {{main.evals}} : {{main.base_evals}} |
| Mikiri, six-rung ladder | {{long.games}} | {{long.elo}} ({{long.elo_lo}} to {{long.elo_hi}}) | {{long.evals}} : {{long.base_evals}} |
| Mikiri's rule on V-MCTS's own ladder (50, 100, 200, 400) | {{short.games}} | {{short.elo}} ({{short.elo_lo}} to {{short.elo_hi}}) | {{short.evals}} : {{short.base_evals}} |

At equal visits Mikiri is clearly ahead of both: the intervals do not overlap. The DS-MCTS-style rule also runs more network evaluations than Mikiri, so Mikiri leads it on every measure. Against V-MCTS, counting network evaluations, it is closer. V-MCTS never searches past 400 visits, so more of its work is already in KataGo's cache, and it ran {{vmcts.evals}} evaluations per move to Mikiri's {{main.evals}}. Interpolating between Mikiri's 160-visit and 200-visit matches puts it at about +135 Elo at {{vmcts.evals}} evaluations, level with V-MCTS. The last row is the controlled test, with both rules on the same ladder so that they differ in nothing else. There Mikiri's rule gains {{short.elo}} to V-MCTS's {{vmcts.elo}}, with intervals that overlap almost entirely: on that ladder the two cannot be told apart. So Mikiri's lead at equal visits comes from ladders that reach 3,200 visits, which its rate rule makes affordable and which cost more evaluations per visit. Per network evaluation, Mikiri and V-MCTS are level.

## Match results

The offline analysis predicted a gain. These are the played games that confirm it: from the empty board, alternating colors, each player on its own KataGo process.

Compute is reported three ways because the answer depends on how you count. **Visits** is the size of the deepest search per move. **Restart visits** charges every search Mikiri requested, including smaller ones it later extended. **NN evals** is the number of network evaluations each player's KataGo process actually ran.

{{table}}

{{progress_note}}The paired-openings row is a control. Both players always play their engine's top move from 400 balanced openings, each played twice with colors swapped, so the gain cannot come from how moves are sampled.

![Elo against compute, counted three ways](results/figures/elo_vs_compute.png)

The orange line is KataGo at uniform budgets. Both Mikiri runs sit above it under all three ways of counting. At about the same network evaluations per move (99 against 98), Mikiri gains +206 Elo where KataGo at 283 visits gains +108.

**The offline analysis called it.** From the dataset alone, before a single match game, the regret numbers forecast +197 Elo for the main match. The match gave +206.

![Where Mikiri spends its visits](results/figures/profile_main.png)

Nobody told Mikiri to save visits in the opening or to ease off once a game is decided. Both fall out of predicting regret in winrate units: it averages about 65 visits over the first 20 moves, peaks near 270 around move 140, and drops to about 130 when it rates its own winrate above 90%.

![Mikiri's winrate estimate by move number](results/figures/advantage.png)

The edge is the same with either color (76.2% as Black, 77.0% as White) and it builds where the visits go: level through the opening, then climbing steadily from about move 70.

## Try it

Replay the demo game move by move in a terminal. No GPU, no KataGo:

```bash
pip install -e .
python -m mikiri.demo --replay results/demo/game.json
```

```
{{demo_excerpt}}
```

Watch the mechanics against a toy engine, or play live against a real KataGo:

```bash
python -m mikiri.demo --fake --games 2
python -m mikiri.demo --net b18 --stopper models/stopper_b18.json --games 2
```

## What happened to v1

<img src="results/figures/v1_bug.png" width="330" align="right" alt="The v1 bug: three black stones in one turn">

The first version of this project, called DataGo, reported 9-0-1 and 8-0-2 records against KataGo. Those came from a bug. Its match runner analysed positions with a GTP command that also plays a move, so the standard search, the deep search, and each recursive search all placed a stone before White replied. Retrieved moves were never sent to the engine at all.

[`legacy/AUDIT.md`](legacy/AUDIT.md) walks through it, and [`legacy/audit/reproduce_double_move.py`](legacy/audit/reproduce_double_move.py) reproduces it with v1's own code: three black stones after one Black turn. v1 is preserved unchanged in [`legacy/v1/`](legacy/v1/). None of its numbers are used here.

<br clear="right">

## Reproduce

```bash
pip install -e ".[dev,experiments]"
python -m pytest -q        # 26 tests against a toy engine, no GPU
```

For real runs set `KATAGO` (path to a KataGo 1.18 binary) and `MIKIRI_HOME` (a directory with `nets/` holding the networks named in [`mikiri/engine.py`](mikiri/engine.py)).

| Step | Command | GPU time (one RTX 6000 Ada) |
|---|---|---|
| Self-play positions | `python experiments/collect.py games --n 300 --out runs/data/games19.jsonl` | 15 min |
| Ladder dataset | `python experiments/collect.py label --games runs/data/games19.jsonl --out runs/data/ladder19.jsonl` | 1.5 h |
| Train the stopper | `python experiments/train_stopper.py data/ladder19.jsonl.gz --export 50,200,800,3200:sqrt` | none |
| Main match | `python experiments/series.py --name main_200 --games 1000 --budget 200 --path 50,200,800,3200 --stopper models/stopper_b18.json --control` | 1.5 h |
| Tables, figures, paper | `experiments/build_paper.sh` | none |

The dataset and trained model ship in [`data/`](data/) and [`models/`](models/), so the first three steps are optional.

```
mikiri/        the engine wrapper: board, KataGo client, stopper, memory, players, matches
experiments/   scripts behind every number, table, and figure
data/          the ladder dataset and the games it was drawn from
models/        trained stopping models (JSON)
results/       match records, analysis reports, figures
paper/         LaTeX source and PDF
legacy/        v1, unchanged, and the audit of its results
tests/         26 tests that run the whole pipeline against a toy engine
```

## Scope

- Measured on KataGo networks at 100 to 800 visits per move, against KataGo itself. That is fast-play territory, where each doubling of search is worth about 230 Elo. Tournament-scale budgets are the next thing to test.
- AlphaGo itself is not available to play, so the comparison is with KataGo, the strongest open engine built on its design.
- Regret in the offline analysis is measured by a search. The match results do not depend on it.

## Credits

v2 (this rebuild, the experiments, and the paper) is by Taka Khoo. The first version, preserved in [`legacy/v1/`](legacy/v1/), was a team project by Benjamin Huh, Jason Peng, Taka Khoo, Olir Eswaramoorthy, David Roos, and Victor Lun Pun at the Thayer School of Engineering, Dartmouth College. Experiments ran on the Dartmouth LISP Lab's machines. KataGo and its networks are the work of David J. Wu and the KataGo community.
