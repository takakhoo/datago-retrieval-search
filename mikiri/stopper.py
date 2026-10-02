"""Learned stopping rule for the visit ladder.

The model predicts, from the search just finished (and how it changed since
the previous rung), how much winrate the current best move is still expected
to give up. Search stops once that prediction drops below a threshold.

The model is a gradient-boosted tree ensemble stored as plain JSON and
evaluated with NumPy, so playing needs no scikit-learn and no pickle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES

TRAJECTORY_NAMES = ["best_changed", "n_changes", "abs_d_winrate", "d_winrate",
                    "d_top1_share", "d_lcb_margin"]
STOPPER_FEATURES = FEATURE_NAMES + TRAJECTORY_NAMES
_WR = FEATURE_NAMES.index("root_winrate")
_TOP1 = FEATURE_NAMES.index("top1_share")
_LCB = FEATURE_NAMES.index("lcb_margin")


def trajectory(prev: np.ndarray | None, cur: np.ndarray, prev_move: int | None,
               cur_move: int, n_changes: int) -> tuple[np.ndarray, int]:
    """Change features between consecutive rungs. Returns (features, updated change count)."""
    if prev is None:
        return np.zeros(len(TRAJECTORY_NAMES)), 0
    changed = float(prev_move != cur_move)
    n_changes += int(changed)
    dw = cur[_WR] - prev[_WR]
    return np.array([changed, n_changes, abs(dw), dw, cur[_TOP1] - prev[_TOP1],
                     cur[_LCB] - prev[_LCB]]), n_changes


class TreeEnsemble:
    """Sum of regression trees: init + lr * sum_t tree_t(x)."""

    def __init__(self, init: float, lr: float, trees: list[dict]):
        self.init, self.lr = init, lr
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in trees]

    def raw(self, x: np.ndarray) -> float:
        total = self.init
        for t in self.trees:
            node = 0
            left, right, feat, thr = t["left"], t["right"], t["feature"], t["threshold"]
            while left[node] != -1:
                node = left[node] if x[feat[node]] <= thr[node] else right[node]
            total += self.lr * t["value"][node]
        return float(total)

    def raw_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.raw(x) for x in X])

    @classmethod
    def from_sklearn(cls, gbm) -> "TreeEnsemble":
        """Export a fitted GradientBoostingRegressor or binary GradientBoostingClassifier."""
        trees = []
        for est in gbm.estimators_[:, 0]:
            t = est.tree_
            trees.append({
                "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                "feature": np.maximum(t.feature, 0).tolist(), "threshold": t.threshold.tolist(),
                "value": t.value[:, 0, 0].tolist(),
            })
        zero = np.zeros((1, gbm.n_features_in_))
        if hasattr(gbm, "predict_proba"):
            init = float(gbm._raw_predict_init(zero).ravel()[0])
        else:
            init = float(gbm.init_.predict(zero).ravel()[0])
        return cls(init, float(gbm.learning_rate), trees)

    def to_json(self) -> dict:
        return {"init": self.init, "lr": self.lr,
                "trees": [{k: v.tolist() for k, v in t.items()} for t in self.trees]}


class Stopper:
    """Stop at rung j when scale[j] * predicted_regret < threshold.

    With scale[j] proportional to 1 / (visits needed to reach the next rung),
    the rule compares the regret still on the table with the price of going
    on, so a high rung needs more at stake than a low one to continue.
    """

    def __init__(self, model: TreeEnsemble, threshold: float, path: list[int],
                 scales: list[float] | None = None, features: list[str] | None = None,
                 meta: dict | None = None, squared: bool = False):
        self.model, self.threshold, self.path = model, threshold, list(path)
        # A model fitted to the square root of regret is less swayed by a few huge
        # blunders. Its output is squared back before use.
        self.squared = squared
        self.scales = list(scales) if scales is not None else [1.0] * (len(path) - 1)
        self.features = features or STOPPER_FEATURES
        self.meta = meta or {}
        if self.features != STOPPER_FEATURES:
            raise ValueError("stopper was trained on a different feature list")
        if len(self.scales) != len(self.path) - 1:
            raise ValueError("need one scale per non-final rung")

    @staticmethod
    def rate_scales(path: list[int]) -> list[float]:
        steps = np.diff(np.asarray(path, dtype=float))
        return (steps[0] / steps).tolist()

    def score(self, x: np.ndarray, rung: int = 0) -> float:
        raw = self.model.raw(x)
        if self.squared:
            raw = max(raw, 0.0) ** 2
        return self.scales[rung] * raw

    def should_stop(self, x: np.ndarray, rung: int = 0) -> bool:
        return self.score(x, rung) < self.threshold

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "version": 2, "threshold": self.threshold, "path": self.path, "scales": self.scales,
            "squared": self.squared,
            "features": self.features, "meta": self.meta, "model": self.model.to_json()}))

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> "Stopper":
        d = json.loads(Path(path).read_text())
        m = d["model"]
        return cls(TreeEnsemble(m["init"], m["lr"], m["trees"]),
                   d["threshold"] if threshold is None else threshold,
                   d["path"], d.get("scales"), d["features"], d.get("meta"),
                   bool(d.get("squared", False)))
