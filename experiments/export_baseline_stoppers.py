"""Package the two strongest prior rules as playable stoppers.

  models/baseline_vmcts.json    V-MCTS (Ye et al. 2022): ladder 50/100/200/400, total 400, r = 0.2,
                                threshold eps (0.1 as published; the controller may move it)
  models/baseline_dsmcts.json   DS-MCTS-style classifier (Lan et al. 2021 label) on root statistics,
                                flat threshold on the probability, doubling ladder
Each carries a calibration table from the ladder dataset so the budget controller can hold its mean cost.
"""
import gzip
import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from baselines import GBM, build_arrays, virtual_policy
from mikiri.ladder import evaluate_stops, load, sequential_stops
from mikiri.stopper import Stopper, TreeEnsemble, VirtualExpansionModel
from train_stopper import folds_by_game, path_features

DATA, MAPS = "data/ladder19.jsonl.gz", "data/ladder19_maps.jsonl.gz"
lad = load(DATA, keep_maps=True, maps_path=MAPS)
recs = [json.loads(l) for l in gzip.open(DATA, "rt")]
n, K = lad.regret.shape
rungs = [int(v) for v in lad.rungs]

# --- V-MCTS, total budget 400: checkpoints at 100 and 200 ---
P, N, Q = build_arrays(recs, lad.policy, rungs)
top, checks = 3, [1, 2]
dist = {k: np.abs(virtual_policy(P, N[:, k], Q[:, k], rungs[top])
                  - virtual_policy(P, N[:, k - 1], Q[:, k - 1], rungs[top])).sum(axis=1) for k in checks}
calib = []
for eps in sorted(set(np.quantile(np.concatenate(list(dist.values())), np.linspace(0, 1, 120)).tolist() + [0.1])):
    stop = np.full(n, top)
    active = np.ones(n, dtype=bool)
    for k in checks:
        halt = active & (dist[k] < eps)
        stop[halt] = k
        active &= ~halt
    r = evaluate_stops(lad, stop, [0, 1, 2, 3])
    calib.append({"threshold": float(eps), "cost_continue": r["cost_continue"], "cost_restart": r["cost_restart"],
                  "regret": r["regret"], "equivalent_visits": r["equivalent_visits"]})
Stopper(VirtualExpansionModel(400, 0.2), 0.1, [50, 100, 200, 400], [1.0, 1.0, 1.0],
        meta={"kind": "vmcts", "calibration": calib}).save("models/baseline_vmcts.json")
pub = min(calib, key=lambda c: abs(c["threshold"] - 0.1))
print(f"V-MCTS total 400: eps 0.1 costs {pub['cost_continue']:.0f} visits, equivalent {pub['equivalent_visits']:.0f}; "
      f"cost range {calib[-1]['cost_continue']:.0f} to {calib[0]['cost_continue']:.0f}")

# --- DS-MCTS-style classifier on the doubling ladder ---
path = list(range(K))
X = path_features(lad, path)
q_final = lad.q[:, -1]
later_worst = np.maximum.accumulate((q_final[:, None] - lad.q)[:, ::-1], axis=1)[:, ::-1]
U = (later_worst >= 0.005).astype(int)
flat = lambda a: a[:, :-1].reshape(-1, *a.shape[2:])  # noqa: E731
folds = folds_by_game(lad.game, 5, 0)
scores = np.zeros((n, K))
for f in np.unique(folds):
    tr, te = folds != f, folds == f
    m = GradientBoostingClassifier(**GBM).fit(flat(X[tr]), flat(U[tr]))
    scores[te, :-1] = m.predict_proba(flat(X[te]))[:, 1].reshape(-1, K - 1)
calib = []
for t in np.unique(np.quantile(scores[:, :-1].ravel(), np.linspace(0, 1, 200))):
    r = evaluate_stops(lad, sequential_stops(lad, path, scores, t), path)
    calib.append({"threshold": float(t), "cost_continue": r["cost_continue"], "cost_restart": r["cost_restart"],
                  "regret": r["regret"], "equivalent_visits": r["equivalent_visits"]})
model = GradientBoostingClassifier(**GBM).fit(flat(X), flat(U))
at200 = max((c for c in calib if c["cost_continue"] <= 200), key=lambda c: c["cost_continue"])
Stopper(TreeEnsemble.from_sklearn(model), at200["threshold"], rungs, [1.0] * (K - 1), sigmoid=True,
        meta={"kind": "dsmcts", "eps": 0.005, "calibration": calib}).save("models/baseline_dsmcts.json")
print(f"DS-MCTS-style: threshold {at200['threshold']:.3f} costs {at200['cost_continue']:.0f}, "
      f"equivalent {at200['equivalent_visits']:.0f}")
