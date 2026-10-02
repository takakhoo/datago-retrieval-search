"""Which parts of the stopper matter? Cross-fitted ablations on the ladder dataset.

Each row reports the compute multiplier at several mean budgets, by continuation
cost and (after the slash) with every restarted search counted, averaged over
three different fold assignments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from mikiri.ladder import load
from mikiri.stopper import STOPPER_FEATURES, TRAJECTORY_NAMES, Stopper
from train_stopper import at_cost, folds_by_game, frontier, path_features

BUDGETS = (100, 200, 400, 800)


def run(lad, ladder: str, cols: list[int], target: str = "sqrt", rule: str = "rate",
        trees: int = 150, depth: int = 3, seeds=(0, 1, 2)) -> dict:
    rung_index = {int(v): i for i, v in enumerate(lad.rungs)}
    visits = [int(v) for v in ladder.split(",")]
    path = [rung_index[v] for v in visits]
    X = path_features(lad, path)[:, :, cols]
    regret = lad.regret[:, path]
    n, P = regret.shape
    scales = np.array(Stopper.rate_scales(visits)) if rule == "rate" else np.ones(P - 1)
    gains = {b: [] for b in BUDGETS}
    for seed in seeds:
        folds = folds_by_game(lad.game, 5, seed)
        scores = np.zeros((n, P))
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            y = regret[tr][:, :-1].reshape(-1)
            model = GradientBoostingRegressor(
                n_estimators=trees, max_depth=depth, learning_rate=0.05, subsample=0.8,
                min_samples_leaf=30, random_state=seed,
            ).fit(X[tr][:, :-1].reshape(-1, X.shape[2]), np.sqrt(y) if target == "sqrt" else y)
            pred = model.predict(X[te][:, :-1].reshape(-1, X.shape[2])).reshape(-1, P - 1)
            scores[te, :-1] = np.maximum(pred, 0) ** 2 if target == "sqrt" else pred
        scores[:, :-1] *= scales[None, :]
        front = frontier(lad, path, scores, points=200)
        for b in BUDGETS:
            pt = at_cost(front, b)
            if pt and pt["cost_continue"] > 0.8 * b:
                gains[b].append((pt["gain_continue"], pt["gain_restart"]))
    return {str(b): {"continue": float(np.mean([g[0] for g in v])),
                     "restart": float(np.mean([g[1] for g in v]))} for b, v in gains.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="results/stopper/ablation.json")
    args = ap.parse_args()
    lad = load(args.data)
    names = STOPPER_FEATURES
    every = list(range(len(names)))
    main_ladder = "50,200,800,3200"
    core = [names.index(x) for x in ("lcb_margin", "top1_share", "undecided", "phase",
                                     "log_visits", "q_gap", "root_winrate")]
    variants = {
        "full model": dict(ladder=main_ladder, cols=every),
        "regret target (no square root)": dict(ladder=main_ladder, cols=every, target="reg"),
        "flat threshold (no rate rule)": dict(ladder=main_ladder, cols=every, rule="flat"),
        "no trajectory features": dict(ladder=main_ladder, cols=[i for i in every if names[i] not in TRAJECTORY_NAMES]),
        "no KataGo raw error features": dict(ladder=main_ladder, cols=[i for i in every if not names[i].startswith("raw_")]),
        "seven core features": dict(ladder=main_ladder, cols=core),
        "LCB margin only": dict(ladder=main_ladder, cols=[names.index("lcb_margin"), names.index("log_visits")]),
        "80 trees of depth 2": dict(ladder=main_ladder, cols=every, trees=80, depth=2),
        "400 trees of depth 3": dict(ladder=main_ladder, cols=every, trees=400),
        "ladder 50,200,800": dict(ladder="50,200,800", cols=every),
        "ladder 100,400,1600": dict(ladder="100,400,1600", cols=every),
        "ladder 50,200,400,800,1600,3200": dict(ladder="50,200,400,800,1600,3200", cols=every),
        "ladder 50,100,200,400,800,1600,3200": dict(ladder="50,100,200,400,800,1600,3200", cols=every),
    }
    report = {"positions": len(lad), "budgets": BUDGETS, "variants": {}}
    for name, kw in variants.items():
        report["variants"][name] = run(lad, **kw)
        print(f"{name:38s}", "  ".join(
            f"b{b}: x{v['continue']:.2f}/{v['restart']:.2f}" for b, v in report["variants"][name].items()),
            flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
