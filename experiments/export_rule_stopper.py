"""Package a hand-written stopping signal as a stopper file, with a calibration
table from the ladder dataset so the budget controller can drive it.

Usage: export_rule_stopper.py DATA RULE flat|rate OUT.json
"""
import sys

import numpy as np

from mikiri.ladder import load
from mikiri.stopper import RULES, RuleModel, Stopper
from train_stopper import frontier, path_features

data, rule, scaling, out = sys.argv[1:5]
ladder = [50, 200, 800, 3200]
lad = load(data)
rung_index = {int(v): i for i, v in enumerate(lad.rungs)}
path = [rung_index[v] for v in ladder]
X = path_features(lad, path)
model = RuleModel(rule)
scales = np.array(Stopper.rate_scales(ladder)) if scaling == "rate" else np.ones(len(ladder) - 1)
scores = np.zeros(X.shape[:2])
for j in range(len(ladder) - 1):
    scores[:, j] = scales[j] * model.raw_batch(X[:, j])
front = frontier(lad, path, scores, points=200)
calib = [{"threshold": r["threshold"], "cost_continue": r["cost_continue"], "cost_restart": r["cost_restart"],
          "regret": r["regret"], "equivalent_visits": r["equivalent_visits"]} for r in front]
at200 = max((r for r in front if r["cost_continue"] <= 200), key=lambda r: r["cost_continue"])
Stopper(model, at200["threshold"], ladder, scales.tolist(),
        meta={"kind": "rule", "rule": rule, "scaling": scaling, "calibration": calib}).save(out)
print(f"{rule} ({scaling}): at cost {at200['cost_continue']:.0f} multiplier x{at200['gain_continue']:.2f}; wrote {out}")
