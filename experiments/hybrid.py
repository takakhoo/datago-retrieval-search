"""Do the prior rules see something Mikiri's stopper does not?

Each prior rule reduces the root statistics to one number: the KLD gain of
Leela Chess Zero, the value-of-information bound of Hay et al., the
best-arm gap of Kaufmann and Koolen, and the virtual-expansion distance of
Ye et al. This script feeds those numbers to Mikiri's regressor as extra
inputs, on the doubling ladder, and reports the change in the compute
multiplier with a paired bootstrap over games. A gain would mean the
hand-derived statistics carry information the 30 features miss.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from baselines import best_point, build_arrays, frontier, paired_diff, root_signals, virtual_policy
from mikiri.ladder import load
from mikiri.stopper import Stopper
from train_stopper import folds_by_game, path_features

BUDGETS = (100, 200, 400, 800)


def virtual_distance(P, N, Q, rungs) -> np.ndarray:
    """L1 distance between virtually expanded policies at consecutive rungs, filled to twice the current rung."""
    n, K = N.shape[:2]
    out = np.zeros((n, K))
    prior = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-9)
    for k in range(K):
        total = 2 * rungs[k]
        a = virtual_policy(P, N[:, k], Q[:, k], total)
        b = prior if k == 0 else virtual_policy(P, N[:, k - 1], Q[:, k - 1], total)
        out[:, k] = np.abs(a - b).sum(axis=1)
    return out


def crossfit(X, regret, game, seed) -> np.ndarray:
    n, Q, F = X.shape
    folds = folds_by_game(game, 5, seed)
    pred = np.zeros((n, Q))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=30, random_state=seed,
        ).fit(X[tr].reshape(-1, F), np.sqrt(regret[tr].reshape(-1)))
        pred[te] = (np.maximum(model.predict(X[te].reshape(-1, F)), 0) ** 2).reshape(-1, Q)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ladder19.jsonl.gz")
    ap.add_argument("--maps", default="data/ladder19_maps.jsonl.gz")
    ap.add_argument("--out", default="results/stopper/hybrid.json")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    lad = load(args.data, keep_maps=True, maps_path=args.maps)
    recs = [json.loads(l) for l in gzip.open(args.data, "rt")]
    rungs = [int(v) for v in lad.rungs]
    K = len(rungs)
    path = list(range(K))
    P, N, Q = build_arrays(recs, lad.policy, rungs)
    sig = root_signals(P, N, Q, rungs)
    sig["vmcts"] = virtual_distance(P, N, Q, rungs)
    extra = {k: np.log1p(np.maximum(v, 0)) for k, v in sig.items()}

    base = path_features(lad, path)[:, :-1]
    regret = lad.regret[:, :-1]
    scales = np.array(Stopper.rate_scales(rungs))
    variants = {"Mikiri": []}
    variants.update({f"+ {k}": [k] for k in extra})
    variants["+ all four"] = list(extra)

    report: dict = {"positions": len(lad), "ladder": rungs, "budgets": BUDGETS, "variants": {}}
    stops: dict[tuple[str, int, int], np.ndarray] = {}
    for name, keys in variants.items():
        X = base if not keys else np.concatenate([base] + [extra[k][:, :-1, None] for k in keys], axis=2)
        gains = {b: [] for b in BUDGETS}
        for seed in range(args.seeds):
            pred = crossfit(X, regret, lad.game, seed)
            scores = np.concatenate([pred * scales[None, :], np.zeros((len(lad), 1))], axis=1)
            front = frontier(lad, path, scores, points=300)
            for b in BUDGETS:
                pt = best_point(front, b)
                if pt:
                    gains[b].append(pt["gain_continue"])
                    stops[(name, seed, b)] = pt["stop"]
        row = {str(b): float(np.mean(v)) for b, v in gains.items() if v}
        if name != "Mikiri":
            for b in BUDGETS:
                if (name, 0, b) in stops and ("Mikiri", 0, b) in stops:
                    d = paired_diff(lad, stops[(name, 0, b)], stops[("Mikiri", 0, b)])
                    row[f"paired_{b}"] = [d["difference"], d["lo"], d["hi"]]
        report["variants"][name] = row
        print(f"{name:14s}", "  ".join(f"b{b}: x{row[str(b)]:.2f}" for b in BUDGETS if str(b) in row),
              "  ".join(f"d{b}: {row[f'paired_{b}'][0]:+.2f} [{row[f'paired_{b}'][1]:+.2f},{row[f'paired_{b}'][2]:+.2f}]"
                       for b in BUDGETS if f"paired_{b}" in row), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
