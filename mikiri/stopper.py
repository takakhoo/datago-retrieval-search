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


_ENT = FEATURE_NAMES.index("visit_entropy")
_PHASE = FEATURE_NAMES.index("phase")

RULES = {
    # The best rival's winrate minus the chosen move's lower confidence bound, floored at zero.
    "lcb_margin": lambda x: max(-x[_LCB], 0.0) + 1e-6 * x[_ENT],
    # The first version's gate: visit entropy scaled by a linear game-phase factor.
    "v1_entropy": lambda x: x[_ENT] * (0.5 * x[_PHASE] + 0.75),
}


class RuleModel:
    """A hand-written stopping signal with the same interface as TreeEnsemble."""

    def __init__(self, rule: str):
        self.rule, self.fn = rule, RULES[rule]

    def raw(self, x: np.ndarray) -> float:
        return float(self.fn(x))

    def raw_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.raw(x) for x in X])

    def to_json(self) -> dict:
        return {"rule": self.rule}


class VirtualExpansionModel:
    """The stopping signal of Ye et al. (2022), for comparison.

    At a checkpoint with k visits it fills the rest of a budget of `total`
    visits by root UCB with Q frozen, does the same from the previous rung, and
    returns the L1 distance between the two visit distributions. Search stops
    when that distance is below the threshold. Checkpoints below r * total are
    skipped, as in the paper.
    """
    needs_policy = True

    def __init__(self, total: int, r: float = 0.2, c1: float = 1.25, c2: float = 19652.0):
        self.total, self.r, self.c1, self.c2 = total, r, c1, c2

    def _expand(self, P: np.ndarray, N: np.ndarray, Q: np.ndarray) -> np.ndarray:
        Nh = N.astype(float).copy()
        for _ in range(max(self.total - int(N.sum()), 0)):
            s = Nh.sum()
            ucb = Q + P * np.sqrt(s) / (1 + Nh) * (self.c1 + np.log((s + self.c2 + 1) / self.c2))
            Nh[int(ucb.argmax())] += 1
        return Nh / Nh.sum()

    def raw_ctx(self, x: np.ndarray, ctx: dict) -> float:
        prev, cur = ctx.get("prev"), ctx["cur"]
        if prev is None or cur.visits < self.r * self.total:
            return float("inf")
        size = ctx["size"]
        points = sorted({m.point for m in cur.moves} | {m.point for m in prev.moves})
        if cur.policy is not None:
            pol = np.asarray(cur.policy)
            extra = [int(i) for i in np.argsort(-pol[: size * size])[:48] if int(i) not in points]
            points += extra
            prior = {p: max(float(pol[size * size if p < 0 else p]), 0.0) for p in points}
        else:
            prior = {m.point: m.prior for m in cur.moves}
        P = np.array([prior.get(p, 0.0) for p in points])
        out = []
        for res in (cur, prev):
            stats = {m.point: m for m in res.moves}
            qmin = min(m.winrate for m in res.moves)
            N = np.array([stats[p].visits if p in stats else 0 for p in points], dtype=float)
            Q = np.array([stats[p].winrate if p in stats else qmin for p in points])
            out.append(self._expand(P, N, Q))
        return float(np.abs(out[0] - out[1]).sum())

    def raw(self, x: np.ndarray) -> float:
        raise RuntimeError("virtual expansion needs the search context")

    def to_json(self) -> dict:
        return {"virtual": {"total": self.total, "r": self.r}}


class Stopper:
    """Stop at rung j when scale[j] * predicted_regret < threshold.

    With scale[j] proportional to 1 / (visits needed to reach the next rung),
    the rule compares the regret still on the table with the price of going
    on, so a high rung needs more at stake than a low one to continue.
    """

    def __init__(self, model: TreeEnsemble, threshold: float, path: list[int],
                 scales: list[float] | None = None, features: list[str] | None = None,
                 meta: dict | None = None, squared: bool = False, sigmoid: bool = False):
        self.model, self.threshold, self.path = model, threshold, list(path)
        # A model fitted to the square root of regret is less swayed by a few huge
        # blunders. Its output is squared back before use.
        self.squared = squared
        # A classifier's logit is mapped to a probability before any scaling.
        self.sigmoid = sigmoid
        self.needs_policy = bool(getattr(model, "needs_policy", False))
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

    def score(self, x: np.ndarray, rung: int = 0, ctx: dict | None = None) -> float:
        raw = self.model.raw_ctx(x, ctx) if hasattr(self.model, "raw_ctx") else self.model.raw(x)
        if self.squared:
            raw = max(raw, 0.0) ** 2
        if self.sigmoid:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return self.scales[rung] * raw

    def should_stop(self, x: np.ndarray, rung: int = 0, ctx: dict | None = None) -> bool:
        return self.score(x, rung, ctx) < self.threshold

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "version": 2, "threshold": self.threshold, "path": self.path, "scales": self.scales,
            "squared": self.squared, "sigmoid": self.sigmoid,
            "features": self.features, "meta": self.meta, "model": self.model.to_json()}))

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> "Stopper":
        d = json.loads(Path(path).read_text())
        m = d["model"]
        if "rule" in m:
            model = RuleModel(m["rule"])
        elif "virtual" in m:
            model = VirtualExpansionModel(m["virtual"]["total"], m["virtual"]["r"])
        else:
            model = TreeEnsemble(m["init"], m["lr"], m["trees"])
        return cls(model,
                   d["threshold"] if threshold is None else threshold,
                   d["path"], d.get("scales"), d["features"], d.get("meta"),
                   bool(d.get("squared", False)), bool(d.get("sigmoid", False)))
