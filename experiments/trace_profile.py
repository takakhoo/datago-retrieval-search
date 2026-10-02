"""Where does DataGo spend its visits? Profile a traced match by move number and by game state."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures import BLUE, MUTED, ORANGE, save

ap = argparse.ArgumentParser()
ap.add_argument("games", help="games.jsonl with traces")
ap.add_argument("--name", required=True)
ap.add_argument("--budget", type=int, default=200)
ap.add_argument("--out", default="results")
args = ap.parse_args()

by_move = defaultdict(list)
by_wr = defaultdict(list)
rungs = defaultdict(int)
hits = defaultdict(list)
total = 0
for line in open(args.games):
    rec = json.loads(line)
    for t in rec["trace"]:
        if t["p"] != "datago":
            continue
        total += 1
        by_move[t["n"] // 10].append(t["v"])
        if t["n"] >= 60:
            by_wr[min(int(t["wr"] * 10), 9)].append(t["v"])
        rungs[t["v"]] += 1
        hits[t["n"] // 10].append(1.0 if t.get("hit") else 0.0)

report = {
    "moves": total,
    "share_of_moves_by_final_search_size": {str(k): v / total for k, v in sorted(rungs.items())},
    "share_of_visits_by_final_search_size": {
        str(k): k * v / sum(kk * vv for kk, vv in rungs.items()) for k, v in sorted(rungs.items())},
    "mean_visits_by_move_number": {f"{10 * k}-{10 * k + 9}": float(np.mean(v))
                                   for k, v in sorted(by_move.items()) if len(v) >= 30},
    "hit_rate_by_move_number": {f"{10 * k}-{10 * k + 9}": float(np.mean(v))
                                for k, v in sorted(hits.items()) if len(v) >= 30},
    "mean_visits_by_own_winrate_after_move_60": {f"{10 * k}-{10 * k + 10}%": float(np.mean(v))
                                   for k, v in sorted(by_wr.items())},
}
out = Path(args.out)
(out / f"profile_{args.name}.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
ks = [k for k, v in sorted(by_move.items()) if len(v) >= 30]
axes[0].plot([10 * k + 5 for k in ks], [np.mean(by_move[k]) for k in ks], color=BLUE, label="DataGo")
axes[0].axhline(args.budget, color=ORANGE, label="KataGo (fixed)")
axes[0].set_xlabel("Move number")
axes[0].set_ylabel("Mean visits per move")
axes[0].set_title("DataGo saves in the opening, spends in the middle game")
axes[0].legend(loc="upper right")
ws = sorted(by_wr)
axes[1].plot([10 * k + 5 for k in ws], [np.mean(by_wr[k]) for k in ws], color=BLUE, marker="o", markersize=5)
axes[1].axhline(args.budget, color=ORANGE)
axes[1].set_xlabel("DataGo's winrate estimate (%), moves 60 and later")
axes[1].set_title("and eases off once the game is decided")
for ax in axes:
    ax.set_ylim(0, None)
save(fig, out / "figures", f"profile_{args.name}")
