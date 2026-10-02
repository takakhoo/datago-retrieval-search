"""Train and evaluate the stopping rule on the ladder dataset.

For a ladder path such as 50,200,800 the model sees the search at each
non-final rung (plus how it changed since the previous rung) and predicts the
regret that remains if search stops there. Evaluation is cross-fitted by game:
every position is scored by a model that never saw its game. The exported
model is trained on all games.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from datago.ladder import Ladder, evaluate_stops, load, sequential_stops, uniform_curve
from datago.stopper import STOPPER_FEATURES, Stopper, TreeEnsemble, trajectory


def path_features(lad: Ladder, path: list[int]) -> np.ndarray:
    """(N, len(path), F) stopper inputs along a path of rung indices."""
    n = len(lad)
    out = np.zeros((n, len(path), len(STOPPER_FEATURES)))
    for i in range(n):
        prev_x, prev_move, changes = None, None, 0
        for j, k in enumerate(path):
            x = lad.feats[i, k]
            t, changes = trajectory(prev_x, x, prev_move, int(lad.move[i, k]), changes)
            out[i, j] = np.concatenate([x, t])
            prev_x, prev_move = x, int(lad.move[i, k])
    return out


def folds_by_game(game: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    ids = np.unique(game)
    assign = dict(zip(ids, np.random.default_rng(seed).permutation(len(ids)) % k))
    return np.array([assign[g] for g in game])


def make(kind: str, trees: int, depth: int):
    common = dict(n_estimators=trees, max_depth=depth, learning_rate=0.05, subsample=0.8,
                  min_samples_leaf=30, random_state=0)
    return GradientBoostingClassifier(**common) if kind == "clf" else GradientBoostingRegressor(**common)


def fit(kind: str, X: np.ndarray, regret: np.ndarray, trees: int, depth: int, cut: float):
    y = {"clf": (regret > cut).astype(int), "reg": regret, "sqrt": np.sqrt(regret)}[kind]
    return make(kind, trees, depth).fit(X, y)


def raw(kind: str, model, X: np.ndarray) -> np.ndarray:
    if kind == "clf":
        return model.decision_function(X)
    pred = model.predict(X)
    return np.maximum(pred, 0.0) ** 2 if kind == "sqrt" else pred


def frontier(lad: Ladder, path: list[int], scores_on_path: np.ndarray, points: int = 120) -> list[dict]:
    K = len(lad.rungs)
    full = np.zeros((len(lad), K))
    full[:, path] = scores_on_path
    out = []
    for t in np.unique(np.quantile(scores_on_path[:, :-1].ravel(), np.linspace(0, 1, points))):
        r = evaluate_stops(lad, sequential_stops(lad, path, full, t), path)
        r["threshold"] = float(t)
        out.append(r)
    return sorted(out, key=lambda r: r["cost_continue"])


def at_cost(front: list[dict], budget: float, key: str = "cost_continue") -> dict | None:
    ok = [r for r in front if r[key] <= budget]
    return max(ok, key=lambda r: r[key]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--paths", default="50,200,800,3200;50,200,400,800,1600,3200")
    ap.add_argument("--kinds", default="sqrt",
                    help="reg: fit regret; sqrt: fit its square root and square the prediction")
    ap.add_argument("--rule", default="rate", choices=["rate", "flat"],
                    help="rate scales predicted regret by the visits needed to reach the next rung")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.005)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budgets", default="100,150,200,300,400,600,800")
    ap.add_argument("--export", default=None, help="path:kind to export, e.g. 50,200,800:reg")
    ap.add_argument("--out", default="results/stopper")
    args = ap.parse_args()

    lad = load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rung_index = {int(v): i for i, v in enumerate(lad.rungs)}
    folds = folds_by_game(lad.game, args.folds)
    budgets = [float(b) for b in args.budgets.split(",")]
    curve = uniform_curve(lad)
    print(f"{len(lad)} positions from {len(np.unique(lad.game))} games")
    print("uniform:", {int(v): round(float(r), 5) for v, r in zip(lad.rungs, curve)})

    report = {"positions": len(lad), "rungs": lad.rungs.tolist(), "uniform_regret": curve.tolist(),
              "policies": {}}
    for pstr in args.paths.split(";"):
        path = [rung_index[int(v)] for v in pstr.split(",")]
        X = path_features(lad, path)
        regret = lad.regret[:, path]
        P = len(path)
        for kind in args.kinds.split(","):
            scores = np.zeros((len(lad), P))
            for f in np.unique(folds):
                tr, te = folds != f, folds == f
                model = fit(kind, X[tr][:, :-1].reshape(-1, X.shape[2]),
                            regret[tr][:, :-1].reshape(-1), args.trees, args.depth, args.cut)
                scores[te, :-1] = raw(kind, model, X[te][:, :-1].reshape(-1, X.shape[2])).reshape(-1, P - 1)
            scales = np.array(Stopper.rate_scales([int(v) for v in pstr.split(",")])
                              if args.rule == "rate" else [1.0] * (P - 1))
            scores[:, :-1] *= scales[None, :]
            front = frontier(lad, path, scores)
            oracle = frontier(lad, path, regret)
            key = f"{pstr}:{kind}"
            report["policies"][key] = {"frontier": front, "oracle": oracle}
            print(f"\npath {pstr} [{kind}]")
            for b in budgets:
                pt, orc = at_cost(front, b), at_cost(oracle, b)
                if pt is None:
                    continue
                uni = float(np.interp(np.log(b), np.log(lad.rungs), curve))
                print(f"  budget {b:5.0f}: uniform regret {uni:.5f} | stopper {pt['regret']:.5f} "
                      f"at cost {pt['cost_continue']:6.1f} (restart {pt['cost_restart']:6.1f}) "
                      f"= uniform {pt['equivalent_visits']:6.1f} visits, x{pt['gain_continue']:.2f} "
                      f"(restart x{pt['gain_restart']:.2f}) | oracle x{orc['gain_continue']:.2f}")

    (out / "report.json").write_text(json.dumps(report))

    if args.export:
        pstr, kind = args.export.split(":")
        path = [rung_index[int(v)] for v in pstr.split(",")]
        X = path_features(lad, path)
        regret = lad.regret[:, path]
        model = fit(kind, X[:, :-1].reshape(-1, X.shape[2]), regret[:, :-1].reshape(-1),
                    args.trees, args.depth, args.cut)
        front = report["policies"][args.export]["frontier"]
        calib = [{"threshold": r["threshold"], "cost_continue": r["cost_continue"],
                  "cost_restart": r["cost_restart"], "regret": r["regret"],
                  "equivalent_visits": r["equivalent_visits"]} for r in front]
        visits_path = [int(v) for v in pstr.split(",")]
        stopper = Stopper(TreeEnsemble.from_sklearn(model), calib[len(calib) // 2]["threshold"],
                          visits_path,
                          Stopper.rate_scales(visits_path) if args.rule == "rate" else None,
                          squared=kind == "sqrt",
                          meta={"kind": kind, "rule": args.rule, "positions": len(lad), "trees": args.trees,
                                "depth": args.depth, "cut": args.cut, "calibration": calib,
                                "note": "calibration is cross-fitted; thresholds map to mean cost"})
        name = f"stopper_{pstr.replace(',', '-')}.json"
        stopper.save(out / name)
        print(f"\nexported {out / name}")


if __name__ == "__main__":
    main()
