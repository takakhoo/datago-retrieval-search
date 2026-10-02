"""What should a stopping model predict, and how should the prediction be priced?

The rate rule stops at rung j when predicted_regret_j / (v_{j+1} - v_j) < lambda.
Remaining regret is an upper bound on what continuing can recover. This script
compares it with the quantities the derivation names directly:

  regret      r_j                                  regret left at rung j
  step        r_j - r_{j+1}                        what the next rung recovers
  top         r_j - r_last                         what the whole ladder recovers
  index       max_{k>j} (r_j - r_k)/(v_k - v_j)    best recovery per visit over any continuation

Each is tried as a learned target (cross-fitted by game) and as an oracle that
reads the true value. The hindsight optimum picks, for each position, the rung
minimising r_k + lambda * v_k with all rungs known. No causal rule can beat it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from mikiri.ladder import evaluate_stops, load
from train_stopper import at_cost, folds_by_game, frontier, path_features

BUDGETS = (100, 200, 400, 800)
LADDERS = ("50,200,800,3200", "50,200,400,800,1600,3200", "50,100,200,400,800,1600,3200")


def targets(regret: np.ndarray, visits: np.ndarray) -> dict[str, np.ndarray]:
    n, P = regret.shape
    step = regret[:, :-1] - regret[:, 1:]
    top = regret[:, :-1] - regret[:, -1:]
    index = np.full((n, P - 1), -np.inf)
    for j in range(P - 1):
        for k in range(j + 1, P):
            index[:, j] = np.maximum(index[:, j], (regret[:, j] - regret[:, k]) / (visits[k] - visits[j]))
    return {"regret": regret[:, :-1], "step": step, "top": top, "index": index}


def pricing(visits: np.ndarray) -> dict[str, np.ndarray]:
    nxt = np.diff(visits).astype(float)
    return {"flat": np.ones(len(nxt)), "rate": 1.0 / nxt, "to_top": 1.0 / (visits[-1] - visits[:-1])}


def crossfit(X: np.ndarray, y: np.ndarray, game: np.ndarray, seed: int) -> np.ndarray:
    n, Q, F = X.shape
    scale = np.abs(y).max() or 1.0
    t = np.sign(y) * np.sqrt(np.abs(y) / scale)
    folds = folds_by_game(game, 5, seed)
    pred = np.zeros((n, Q))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=30, random_state=seed,
        ).fit(X[tr].reshape(-1, F), t[tr].reshape(-1))
        p = model.predict(X[te].reshape(-1, F)).reshape(-1, Q)
        pred[te] = np.sign(p) * p ** 2 * scale
    return pred


def multipliers(lad, path: list[int], scores: np.ndarray) -> dict:
    full = np.concatenate([scores, np.zeros((len(scores), 1))], axis=1)
    front = frontier(lad, path, full, points=400)
    out = {}
    for b in BUDGETS:
        pt = at_cost(front, b)
        if pt and pt["cost_continue"] > 0.8 * b:
            out[str(b)] = {"gain": pt["gain_continue"], "cost": pt["cost_continue"], "regret": pt["regret"]}
    return out


def hindsight(lad, path: list[int], regret: np.ndarray, visits: np.ndarray) -> dict:
    front = []
    for lam in np.concatenate([[0.0], np.geomspace(1e-8, 1e-2, 600)]):
        stop = np.array(path)[np.argmin(regret + lam * visits[None, :], axis=1)]
        front.append(evaluate_stops(lad, stop, path))
    out = {}
    for b in BUDGETS:
        pt = at_cost(front, b)
        if pt and pt["cost_continue"] > 0.8 * b:
            out[str(b)] = {"gain": pt["gain_continue"], "cost": pt["cost_continue"], "regret": pt["regret"]}
    return out


def mean_over_seeds(runs: list[dict]) -> dict:
    out = {}
    for b in map(str, BUDGETS):
        vals = [r[b] for r in runs if b in r]
        if vals:
            out[b] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="results/stopper/targets.json")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    lad = load(args.data)
    rung_index = {int(v): i for i, v in enumerate(lad.rungs)}
    report: dict = {"positions": len(lad), "budgets": BUDGETS, "ladders": {}}
    combos = [("regret", "flat"), ("regret", "rate"), ("regret", "to_top"), ("step", "flat"), ("step", "rate"),
              ("top", "flat"), ("top", "to_top"), ("top", "rate"), ("index", "flat")]
    for ladder in LADDERS:
        visits = np.array([int(v) for v in ladder.split(",")])
        path = [rung_index[int(v)] for v in visits]
        X = path_features(lad, path)[:, :-1]
        regret = lad.regret[:, path]
        tg, pr = targets(regret, visits), pricing(visits)
        rows = {"hindsight optimum": hindsight(lad, path, regret, visits)}
        preds = {name: [crossfit(X, y, lad.game, s) for s in range(args.seeds)] for name, y in tg.items()}
        for target, price in combos:
            rows[f"oracle {target} / {price}"] = multipliers(lad, path, tg[target] * pr[price][None, :])
            rows[f"learned {target} / {price}"] = mean_over_seeds(
                [multipliers(lad, path, p * pr[price][None, :]) for p in preds[target]])
        report["ladders"][ladder] = rows
        print(ladder)
        for name, row in rows.items():
            print(f"  {name:28s}", "  ".join(f"b{b}: x{v['gain']:.2f}@{v['cost']:.0f}" for b, v in row.items()), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
