"""Descriptive statistics of the ladder dataset and hand-written stopping rules.

1. How fast regret falls with uniform visits, and how often the move changes.
2. How concentrated regret is across positions.
3. How well single signals rank positions by remaining regret (AUC).
4. What hand-written stopping rules achieve on the same ladder the learned
   stopper uses: v1's entropy gate, the LCB margin, the "most visited is not
   highest valued" rule from classical time management, and random stopping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from mikiri.ladder import FEATURE_NAMES, evaluate_stops, load, sequential_stops, uniform_curve


def frontier(lad, path, scores, points=160):
    out = []
    for t in np.unique(np.quantile(scores[:, path[:-1]].ravel(), np.linspace(0, 1, points))):
        r = evaluate_stops(lad, sequential_stops(lad, path, scores, t), path)
        out.append(r)
    return sorted(out, key=lambda r: r["cost_continue"])


def at_cost(front, budget):
    ok = [r for r in front if r["cost_continue"] <= budget]
    return max(ok, key=lambda r: r["cost_continue"]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--path", default="50,200,800,3200")
    ap.add_argument("--budgets", default="100,200,400,800")
    ap.add_argument("--out", default="results/ladder")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lad = load(args.data)
    n, K = lad.regret.shape
    f = {name: lad.feats[:, :, i] for i, name in enumerate(FEATURE_NAMES)}
    curve = uniform_curve(lad)
    report = {"positions": n, "games": int(len(np.unique(lad.game))), "rungs": lad.rungs.tolist()}

    report["uniform"] = [
        {"visits": int(v), "regret": float(curve[k]),
         "share_with_regret_over_2pct": float((lad.regret[:, k] > 0.02).mean()),
         "share_with_any_regret": float((lad.regret[:, k] > 0).mean()),
         "move_differs_from_3200": float((lad.move[:, k] != lad.move[:, -1]).mean())}
        for k, v in enumerate(lad.rungs)]

    conc = []
    for k in range(K):
        r = np.sort(lad.regret[:, k])[::-1]
        conc.append({"visits": int(lad.rungs[k]),
                     "top_5pct_of_positions_hold": float(r[: n // 20].sum() / r.sum()),
                     "top_10pct_of_positions_hold": float(r[: n // 10].sum() / r.sum())})
    report["concentration"] = conc

    phases = [(0, 40), (40, 100), (100, 160), (160, 1000)]
    k200 = int(np.flatnonzero(lad.rungs == 200)[0])
    report["regret_at_200_by_move_number"] = [
        {"moves": f"{a}-{b}", "positions": int(((lad.move_number >= a) & (lad.move_number < b)).sum()),
         "regret": float(lad.regret[(lad.move_number >= a) & (lad.move_number < b), k200].mean())}
        for a, b in phases]

    signals = {
        "lcb_margin (low)": -f["lcb_margin"],
        "top1_share (low)": -f["top1_share"],
        "visit_entropy": f["visit_entropy"],
        "v1 entropy gate": f["visit_entropy"] * (0.5 * f["phase"] + 0.75),
        "visit_value_disagree": f["visit_value_disagree"],
        "policy_surprise": f["policy_surprise"],
        "value_surprise": f["value_surprise"],
        "katago rawStWrError": f["raw_wr_error"],
        "undecided": f["undecided"],
    }
    positive = (lad.regret > 0.005).astype(int)
    report["auc"] = {name: [float(roc_auc_score(positive[:, k], s[:, k])) for k in range(K - 1)]
                     for name, s in signals.items()}

    rung_index = {int(v): i for i, v in enumerate(lad.rungs)}
    path = [rung_index[int(v)] for v in args.path.split(",")]
    steps = np.diff(lad.rungs[path]).astype(float)
    scale = np.ones(K)
    scale[path[:-1]] = steps[0] / steps
    rng = np.random.default_rng(0)
    rules = {
        "v1 entropy gate": signals["v1 entropy gate"],
        "lcb margin": np.maximum(-f["lcb_margin"], 0) + 1e-6 * f["visit_entropy"],
        "unstable (visits vs value)": f["visit_value_disagree"] + 1e-3 * f["visit_entropy"],
        "random": rng.random((n, K)),
        "oracle": lad.regret,
    }
    budgets = [float(b) for b in args.budgets.split(",")]
    report["heuristic_rules"] = {"path": args.path, "rows": []}
    for name, s in rules.items():
        for rule, sc in (("flat", s), ("rate", s * scale[None, :])):
            front = frontier(lad, path, sc)
            for b in budgets:
                pt = at_cost(front, b)
                if pt:
                    report["heuristic_rules"]["rows"].append({
                        "rule": name, "scaling": rule, "budget": b, "cost": pt["cost_continue"],
                        "regret": pt["regret"], "equivalent_visits": pt["equivalent_visits"],
                        "multiplier": pt["gain_continue"], "multiplier_restart": pt["gain_restart"]})

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("positions", "games", "uniform", "concentration",
                                              "regret_at_200_by_move_number")}, indent=1))
    print("\nAUC for remaining regret > 0.5% (per non-final rung)")
    for name, v in report["auc"].items():
        print(f"  {name:24s}", " ".join(f"{x:.3f}" for x in v))
    print(f"\nhand-written stopping rules on ladder {args.path} (continuation cost)")
    for r in report["heuristic_rules"]["rows"]:
        print(f"  {r['rule']:28s} {r['scaling']:5s} budget {r['budget']:5.0f}: cost {r['cost']:6.1f} "
              f"regret {r['regret']:.5f} x{r['multiplier']:.2f}")


if __name__ == "__main__":
    main()
