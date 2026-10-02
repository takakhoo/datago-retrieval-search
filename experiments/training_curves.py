"""Training diagnostics for the stopper, all on held-out games.

1. Boosting curve: train and held-out error as trees are added.
2. Data curve: compute multiplier as a function of how many games were labeled.
3. Calibration: predicted against realized regret, by decile of prediction.
4. Feature importance: drop in held-out fit when one feature is shuffled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from mikiri.ladder import load
from mikiri.stopper import STOPPER_FEATURES, Stopper
from figures import BLUE, GRAY, INK, MUTED, ORANGE, save
from train_stopper import at_cost, folds_by_game, frontier, path_features

LADDER = [50, 200, 800, 3200]
PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="results/stopper/training.json")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()

    lad = load(args.data)
    rung_index = {int(v): i for i, v in enumerate(lad.rungs)}
    path = [rung_index[v] for v in LADDER]
    X = path_features(lad, path)
    regret = lad.regret[:, path]
    n, P, F = X.shape
    scales = np.array(Stopper.rate_scales(LADDER))
    folds = folds_by_game(lad.game, 5, 0)
    games = np.unique(lad.game)
    flat = lambda a, m: a[m][:, :-1].reshape(-1, *a.shape[2:])  # noqa: E731

    # 1. Boosting curve (one fold held out, 400 trees to show the plateau).
    tr, te = folds != 0, folds == 0
    Xtr, ytr, Xte, yte = flat(X, tr), np.sqrt(flat(regret, tr)), flat(X, te), np.sqrt(flat(regret, te))
    long_model = GradientBoostingRegressor(**{**PARAMS, "n_estimators": 400}, random_state=0).fit(Xtr, ytr)
    train_curve = [float(np.mean((p - ytr) ** 2)) for p in long_model.staged_predict(Xtr)]
    test_curve = [float(np.mean((p - yte) ** 2)) for p in long_model.staged_predict(Xte)]
    base = float(np.mean((yte - ytr.mean()) ** 2))

    # 2. Data curve: train on a subset of the training games, score on held-out folds.
    sizes = [15, 30, 60, 120, 240]
    data_curve = []
    for size in sizes:
        gains = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            scores = np.zeros((n, P))
            for f in np.unique(folds):
                train_games = np.unique(lad.game[folds != f])
                keep = rng.choice(train_games, size=min(size, len(train_games)), replace=False)
                m_tr, m_te = np.isin(lad.game, keep), folds == f
                model = GradientBoostingRegressor(**PARAMS, random_state=seed).fit(
                    flat(X, m_tr), np.sqrt(flat(regret, m_tr)))
                pred = model.predict(flat(X, m_te)).reshape(-1, P - 1)
                scores[m_te, :-1] = np.maximum(pred, 0) ** 2 * scales[None, :]
            pt = at_cost(frontier(lad, path, scores, points=200), 200)
            gains.append((pt["gain_continue"], pt["gain_restart"]))
        data_curve.append({"games": size, "positions": size * 20,
                           "multiplier": float(np.mean([g[0] for g in gains])),
                           "multiplier_min": float(np.min([g[0] for g in gains])),
                           "multiplier_max": float(np.max([g[0] for g in gains])),
                           "multiplier_restart": float(np.mean([g[1] for g in gains]))})
        print(data_curve[-1], flush=True)

    # 3 and 4 use cross-fitted predictions from the deployed configuration.
    pred = np.zeros((n, P - 1))
    importance = np.zeros(F)
    rng = np.random.default_rng(0)
    for f in np.unique(folds):
        m_tr, m_te = folds != f, folds == f
        model = GradientBoostingRegressor(**PARAMS, random_state=0).fit(
            flat(X, m_tr), np.sqrt(flat(regret, m_tr)))
        Xh, yh = flat(X, m_te), np.sqrt(flat(regret, m_te))
        p = model.predict(Xh)
        pred[m_te] = (np.maximum(p, 0) ** 2).reshape(-1, P - 1)
        err = np.mean((p - yh) ** 2)
        for j in range(F):
            Xs = Xh.copy()
            Xs[:, j] = rng.permutation(Xs[:, j])
            importance[j] += (np.mean((model.predict(Xs) - yh) ** 2) - err) / err / 5

    pf, rf = pred.ravel(), regret[:, :-1].ravel()
    order = np.argsort(pf)
    calibration = [{"predicted": float(pf[idx].mean()), "realized": float(rf[idx].mean()),
                    "share_with_regret": float((rf[idx] > 0.005).mean())}
                   for idx in np.array_split(order, 10)]
    feats = sorted(({"feature": STOPPER_FEATURES[j], "increase_in_error": float(importance[j])}
                    for j in range(F)), key=lambda d: -d["increase_in_error"])

    report = {"ladder": LADDER, "positions": n, "games": int(len(games)),
              "boosting": {"train": train_curve, "held_out": test_curve, "held_out_constant": base,
                           "deployed_trees": PARAMS["n_estimators"]},
              "data_curve": data_curve, "calibration": calibration, "permutation_importance": feats}
    Path(args.out).write_text(json.dumps(report, indent=1))

    plot(report, Path(args.figures))


def plot(report: dict, out: Path) -> None:
    b, data_curve = report["boosting"], report["data_curve"]
    train_curve, test_curve = b["train"], b["held_out"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    t = np.arange(1, len(train_curve) + 1)
    axes[0].plot(t, 1e3 * np.array(train_curve), color=GRAY, label="Training games")
    axes[0].plot(t, 1e3 * np.array(test_curve), color=BLUE, label="Held-out games")
    axes[0].axvline(b["deployed_trees"], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[0].text(b["deployed_trees"] + 6, 1e3 * max(train_curve[0], test_curve[0]) * 0.985,
                 "deployed model", color=MUTED, fontsize=8, va="top")
    axes[0].set_xlabel("Boosting rounds (trees)")
    axes[0].set_ylabel(r"Squared error of $\sqrt{\mathrm{regret}}$ ($\times 10^{-3}$)")
    axes[0].set_title("Training the stopper")
    axes[0].legend(loc="upper right")
    xs = [d["positions"] for d in data_curve]
    axes[1].fill_between(xs, [d["multiplier_min"] for d in data_curve],
                         [d["multiplier_max"] for d in data_curve], color=BLUE, alpha=0.15, linewidth=0)
    axes[1].plot(xs, [d["multiplier"] for d in data_curve], color=BLUE, marker="o", markersize=5,
                 label="Continuation cost")
    axes[1].plot(xs, [d["multiplier_restart"] for d in data_curve], color=ORANGE, marker="o",
                 markersize=5, label="Every restart counted")
    axes[1].axhline(1.0, color=MUTED, linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{x:,}" for x in xs])
    axes[1].minorticks_off()
    axes[1].set_ylim(0.9, None)
    axes[1].set_xlabel("Labeled positions used for training")
    axes[1].set_ylabel("Compute multiplier at 200 visits")
    axes[1].set_title("How much data it needs")
    axes[1].legend(loc="lower right")
    save(fig, out, "training")

    cal, feats = report["calibration"], report["permutation_importance"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7), gridspec_kw={"wspace": 0.78, "width_ratios": [1.1, 1]})
    deciles = np.arange(1, len(cal) + 1)
    bars = axes[0].bar(deciles, [100 * c["realized"] for c in cal], width=0.62, color=BLUE)
    for bar, c in zip(bars, cal):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                     f"{100 * c['realized']:.2f}", ha="center", fontsize=7.5, color=INK)
    axes[0].set_xticks(deciles)
    axes[0].set_ylim(0, 100 * cal[-1]["realized"] * 1.12)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_xlabel("Decile of the stopper's predicted stake (held-out games)")
    axes[0].set_ylabel("Realized regret (% winrate)")
    axes[0].set_title("Higher predicted stake, higher real regret")
    top = feats[:8][::-1]
    axes[1].barh(range(len(top)), [100 * d["increase_in_error"] for d in top], color=BLUE, height=0.62)
    axes[1].set_yticks(range(len(top)))
    pretty = {"wr_x_lcb_margin": "game undecided x move contested", "best_prior": "prior of chosen move",
              "raw_wr_error": "KataGo's predicted winrate error", "lcb_margin": "confidence margin (LCB)",
              "visit_entropy": "visit entropy", "q_gap": "value gap to best rival",
              "top1_share": "top move's visit share", "max_prior": "largest prior",
              "undecided": "game undecided", "log_visits": "log visits"}
    axes[1].set_yticklabels([pretty.get(d["feature"], d["feature"].replace("_", " ")) for d in top], color=INK)
    axes[1].grid(axis="y", visible=False)
    axes[1].set_xlabel("Increase in held-out error when shuffled (%)")
    axes[1].set_title("What the model relies on")
    save(fig, out, "stopper_diagnostics")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        plot(json.loads(Path("results/stopper/training.json").read_text()), Path("results/figures"))
    else:
        main()
