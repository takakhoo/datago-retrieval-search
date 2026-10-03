"""Second-order analysis of the recorded matches.

1. Does offline regret predict match Elo? For each stopper run against the
   200-visit baseline, take the mean visits it actually used, read the
   cross-fitted offline regret at that cost, convert it to equivalent uniform
   visits, and read the Elo of that budget off the measured uniform curve.
2. Colour split: score as Black and as White.
3. Where the advantage builds: Mikiri's mean winrate estimate by move number,
   with finished games carried forward at their result.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures import BLUE, MUTED, ORANGE, save
from mikiri.stats import elo_from_score

RES = Path("results")


def uniform_elo_curve() -> tuple[np.ndarray, np.ndarray]:
    pts = [(200.0, 0.0)]
    for name in ("uni_100_vs_200", "uni_283_vs_200", "uni_400_vs_200", "uni_566_vs_200"):
        if not (RES / "matches" / name / "summary.json").exists():
            continue
        s = json.loads((RES / "matches" / name / "summary.json").read_text())
        pts.append((float(s["visits_per_move"]), float(s["elo"])))
    pts.sort()
    return np.log2([p[0] for p in pts]), np.array([p[1] for p in pts])


def offline_equivalent(cost: float, policy: str = "50,200,800,3200:sqrt") -> tuple[float, float]:
    rep = json.loads((RES / "stopper/report.json").read_text())
    front = sorted(rep["policies"][policy]["frontier"], key=lambda r: r["cost_continue"])
    costs = np.array([r["cost_continue"] for r in front])
    regret = float(np.interp(cost, costs, [r["regret"] for r in front]))
    equiv = float(np.interp(cost, costs, [r["equivalent_visits"] for r in front]))
    return regret, equiv


def games(run: str) -> list[dict]:
    with gzip.open(RES / "matches" / run / "games.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def score(rec: dict, name: str = "mikiri") -> float:
    if rec["winner"] == "draw":
        return 0.5
    return float((rec["winner"] == "B") == (rec["black"] == name))


def main() -> None:
    logv, elo = uniform_elo_curve()
    report: dict = {"prediction": [], "colour": {}, "advantage": {}}
    four, six = "50,200,800,3200:sqrt", "50,200,400,800,1600,3200:sqrt"
    for run, policy in (("main_140", four), ("main_stopper_160", four), ("main_175", four), ("main_200", four), ("long_200", six)):
        p = RES / "matches" / run / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        regret, equiv = offline_equivalent(s["visits_per_move"], policy)
        predicted = float(np.interp(np.log2(equiv), logv, elo))
        report["prediction"].append({
            "run": run, "ladder": "Mikiri, 6 rungs" if policy == six else "Mikiri, 4 rungs",
            "visits": s["visits_per_move"], "offline_regret": regret,
            "equivalent_uniform_visits": equiv, "predicted_elo": predicted,
            "measured_elo": s["elo"], "measured_lo": s["elo_lo"], "measured_hi": s["elo_hi"]})

    # Other rules carry their own cross-fitted calibration in the model file.
    for run, label, model in (("double_200", "Mikiri, 7 rungs", "stopper_b18_double"),
                              ("short_200", "Mikiri, short ladder", "stopper_b18_short"),
                              ("vmcts_200", "V-MCTS", "baseline_vmcts"),
                              ("dsmcts_200", "DS-MCTS-style", "baseline_dsmcts")):
        p = RES / "matches" / run / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        calib = sorted(json.loads(Path(f"models/{model}.json").read_text())["meta"]["calibration"],
                       key=lambda r: r["cost_continue"])
        costs = [r["cost_continue"] for r in calib]
        regret = float(np.interp(s["visits_per_move"], costs, [r["regret"] for r in calib]))
        equiv = float(np.interp(s["visits_per_move"], costs, [r["equivalent_visits"] for r in calib]))
        report["prediction"].append({
            "run": run, "ladder": label, "visits": s["visits_per_move"], "offline_regret": regret,
            "equivalent_uniform_visits": equiv, "predicted_elo": float(np.interp(np.log2(equiv), logv, elo)),
            "measured_elo": s["elo"], "measured_lo": s["elo_lo"], "measured_hi": s["elo_hi"]})

    for run in ("main_200", "main_stopper_160", "paired_greedy_200"):
        if not (RES / "matches" / run / "games.jsonl.gz").exists():
            continue
        recs = games(run)
        b = [score(r) for r in recs if r["black"] == "mikiri"]
        w = [score(r) for r in recs if r["white"] == "mikiri"]
        report["colour"][run] = {
            "as_black": {"games": len(b), "score": float(np.mean(b)), "elo": elo_from_score(float(np.mean(b)))},
            "as_white": {"games": len(w), "score": float(np.mean(w)), "elo": elo_from_score(float(np.mean(w)))},
            "resignations": float(np.mean([r["reason"] == "resign" for r in recs])),
            "mean_length": float(np.mean([len(r["moves"]) for r in recs])),
        }

    raw = Path("runs/series/main_200/games.jsonl")
    if raw.exists():
        horizon = 260
        total = np.zeros(horizon)
        n = 0
        for line in open(raw):
            rec = json.loads(line)
            wr = {t["n"]: t["wr"] for t in rec["trace"] if t["p"] == "mikiri"}
            final = score(rec)
            last, row = 0.5, np.empty(horizon)
            end = len(rec["moves"])
            for t in range(horizon):
                if t in wr:
                    last = wr[t]
                row[t] = final if t >= end else last
            total += row
            n += 1
        curve = total / n
        report["advantage"] = {"games": n, "mean_winrate_by_move": [float(x) for x in curve[::10]],
                               "moves": list(range(0, horizon, 10))}
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(np.arange(horizon), 100 * curve, color=BLUE)
        ax.axhline(50, color=MUTED, linewidth=0.8)
        ax.axhline(100 * curve[-1], color=ORANGE, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(2, 100 * curve[-1] + 1.2, f"final score {100 * curve[-1]:.1f}%", color=ORANGE, fontsize=8.5)
        ax.set_ylim(40, 85)
        ax.set_xlabel("Move number")
        ax.set_ylabel("Mikiri's mean winrate estimate (%)")
        ax.set_title("The advantage builds through the middle game")
        save(fig, RES / "figures", "advantage")

    (RES / "match_analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["prediction"], indent=1))
    print(json.dumps(report["colour"], indent=1))
    if report["advantage"]:
        print([round(100 * x, 1) for x in report["advantage"]["mean_winrate_by_move"]])


if __name__ == "__main__":
    main()
