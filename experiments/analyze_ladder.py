"""Offline study of visit allocation on the ladder dataset.

Questions answered, all with predictions cross-fitted by game so no position
is scored by a model that saw its own game:

1. How fast does regret fall with uniform visits?
2. How concentrated is regret? (If a few positions carry most of it, a gate
   has something to find.)
3. Which signals predict that more search pays, and by how much does a
   learned stopping rule beat uniform allocation at equal mean visits?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datago.ladder import FEATURE_NAMES, Ladder, equivalent_visits, evaluate_stops, load, \
    sequential_stops, uniform_curve


def folds_by_game(game: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    ids = np.unique(game)
    rng = np.random.default_rng(seed)
    assign = dict(zip(ids, rng.permutation(len(ids)) % k))
    return np.array([assign[g] for g in game])


def crossfit(lad: Ladder, target: np.ndarray, make_model, folds: np.ndarray,
             classify: bool = False) -> np.ndarray:
    """Out-of-fold predictions of target[:, k] from the rung-k features, one model per rung."""
    n, K = target.shape
    pred = np.zeros((n, K))
    for k in range(K):
        X, y = lad.feats[:, k, :], target[:, k]
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            if classify and len(np.unique(y[tr])) < 2:
                pred[te, k] = y[tr].mean()
                continue
            model = make_model().fit(X[tr], y[tr])
            pred[te, k] = model.predict_proba(X[te])[:, 1] if classify else model.predict(X[te])
    return pred


def v1_gate_score(lad: Ladder) -> np.ndarray:
    """The v1 uncertainty score: (0.5 E + 0.5 K) * phase, on the same search outputs."""
    f = {name: lad.feats[:, :, i] for i, name in enumerate(FEATURE_NAMES)}
    return f["visit_entropy"] * (0.5 * f["phase"] + 0.75)


def frontier(lad: Ladder, path: list[int], scores: np.ndarray, n_thresholds: int = 60) -> list[dict]:
    flat = scores[:, path[:-1]].ravel()
    qs = np.quantile(flat, np.linspace(0, 1, n_thresholds))
    out = []
    for t in np.unique(qs):
        stop = sequential_stops(lad, path, scores, t)
        r = evaluate_stops(lad, stop, path)
        r["threshold"] = float(t)
        r["stop_hist"] = np.bincount(stop, minlength=len(lad.rungs)).tolist()
        out.append(r)
    return out


def gain_at(front: list[dict], budget: float, cost_key: str) -> dict | None:
    """Frontier point whose mean cost is closest to (and not above) budget."""
    ok = [r for r in front if r[cost_key] <= budget * 1.02]
    return max(ok, key=lambda r: r[cost_key]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="results/ladder")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lad = load(args.data)
    n, K = lad.regret.shape
    folds = folds_by_game(lad.game, args.folds)
    report: dict = {"positions": n, "games": int(len(np.unique(lad.game))),
                    "rungs": lad.rungs.tolist()}

    curve = uniform_curve(lad)
    report["uniform"] = [
        {"visits": int(v), "regret": float(r), "blunder_rate": float((lad.regret[:, k] > 0.02).mean()),
         "move_differs_from_top_rung": float((lad.move[:, k] != lad.move[:, -1]).mean())}
        for k, (v, r) in enumerate(zip(lad.rungs, curve))
    ]

    conc = []
    for k in range(K):
        r = np.sort(lad.regret[:, k])[::-1]
        tot = r.sum()
        conc.append({"visits": int(lad.rungs[k]),
                     "nonzero_frac": float((r > 0).mean()),
                     "top5pct_share": float(r[: max(1, n // 20)].sum() / tot) if tot > 0 else 0.0,
                     "top10pct_share": float(r[: max(1, n // 10)].sum() / tot) if tot > 0 else 0.0})
    report["concentration"] = conc

    # Target: regret that remains at this rung. A stopping rule halts when it is predicted small.
    regret_hat = crossfit(lad, lad.regret, lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0), folds)
    positive = (lad.regret > 0.005).astype(float)
    p_hat = crossfit(lad, positive, lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), folds, classify=True)

    names = FEATURE_NAMES
    fi = {name: lad.feats[:, :, i] for i, name in enumerate(names)}
    scorers = {
        "learned_gbm": regret_hat,
        "learned_logistic": p_hat,
        "v1_entropy_gate": v1_gate_score(lad),
        "lcb_margin": -fi["lcb_margin"],
        "top1_share": -fi["top1_share"],
        "visit_value_disagree": fi["visit_value_disagree"] + 1e-3 * (-fi["lcb_margin"]),
        "katago_raw_wr_error": fi["raw_wr_error"],
        "oracle": lad.regret,
        "random": np.random.default_rng(0).random(lad.regret.shape),
    }

    report["auc_regret_positive"] = {
        name: [float(roc_auc_score(positive[:, k], s[:, k])) if 0 < positive[:, k].mean() < 1 else None
               for k in range(K)]
        for name, s in scorers.items() if name not in ("oracle", "random")
    }

    paths = {"double": list(range(K)), "quad_from_50": list(range(0, K, 2)),
             "quad_from_100": list(range(1, K, 2))}
    report["frontiers"] = {}
    for pname, path in paths.items():
        report["frontiers"][pname] = {
            name: frontier(lad, path, s) for name, s in scorers.items()}

    summary = []
    for budget in lad.rungs[1:-1]:
        row = {"budget": int(budget), "uniform_regret": float(np.interp(
            np.log(budget), np.log(lad.rungs), curve))}
        for pname in paths:
            for name in ("learned_gbm", "learned_logistic", "v1_entropy_gate", "lcb_margin",
                         "oracle", "random"):
                for key in ("cost_continue", "cost_restart"):
                    pt = gain_at(report["frontiers"][pname][name], float(budget), key)
                    if pt:
                        row[f"{pname}/{name}/{key}"] = {
                            "regret": pt["regret"], "cost": pt[key],
                            "equivalent_visits": pt["equivalent_visits"],
                            "multiplier": pt["equivalent_visits"] / pt[key]}
        summary.append(row)
    report["at_budget"] = summary

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("positions", "games", "uniform", "concentration",
                                              "auc_regret_positive")}, indent=1))
    for row in summary:
        print(f"\nbudget {row['budget']}: uniform regret {row['uniform_regret']:.5f}")
        for key, val in row.items():
            if isinstance(val, dict):
                print(f"  {key:48s} regret {val['regret']:.5f} cost {val['cost']:7.1f} "
                      f"equiv {val['equivalent_visits']:7.1f} x{val['multiplier']:.2f}")


if __name__ == "__main__":
    main()
