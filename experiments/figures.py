"""Figures for the README and paper, drawn from the JSON reports in results/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False, "figure.dpi": 160, "axes.axisbelow": True,
})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig_frontier(report: dict, key: str, out: Path) -> None:
    rungs = np.array(report["rungs"])
    uniform = 100 * np.array(report["uniform_regret"])
    pol = report["policies"][key]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(rungs, uniform, color=ORANGE, marker="o", markersize=5, label="Uniform visits (KataGo)")
    front = [r for r in pol["frontier"] if rungs[0] <= r["cost_continue"] <= rungs[-1]]
    ax.plot([r["cost_continue"] for r in front], [100 * r["regret"] for r in front],
            color=BLUE, label="Learned stopping (DataGo)")
    orc = [r for r in pol["oracle"] if rungs[0] <= r["cost_continue"] <= rungs[-1]]
    # Past its cheapest best point the oracle has nothing left to gain.
    best = min(range(len(orc)), key=lambda i: (orc[i]["regret"], orc[i]["cost_continue"]))
    orc = orc[: best + 1]
    ax.plot([r["cost_continue"] for r in orc], [100 * r["regret"] for r in orc],
            color=AQUA, linewidth=1.5, linestyle=(0, (4, 2)), label="Oracle stopping (upper bound)")
    ax.set_xscale("log")
    ax.set_xticks(rungs)
    ax.set_xticklabels([str(v) for v in rungs])
    ax.minorticks_off()
    ax.set_xlim(rungs[0] * 0.9, max(r["cost_continue"] for r in front) * 1.1)
    ax.set_ylim(0, None)
    ax.set_xlabel("Mean visits per move")
    ax.set_ylabel("Mean winrate given up per move (%)")
    ax.set_title("Same search budget, less winrate lost")
    ax.legend(loc="upper right")
    save(fig, out, "frontier")


def fig_reuse(report: dict, out: Path) -> None:
    rows = [r for r in report["table"] if r["move"] <= 50]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r["move"] for r in rows], [100 * r["hit_rate_second_half"] for r in rows],
            color=BLUE, marker="o", markersize=5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Positions already in memory (%)")
    ax.set_title(f"How long games stay on known ground "
                 f"(memory of {report['games'] // 2}-{report['games']} earlier games)")
    save(fig, out, "reuse")


def fig_similarity(report: dict, out: Path) -> None:
    rows = [r for r in report["agreement_by_similarity"] if r["neighbour_deep_move_matches"] is not None]
    labels = ["< 0.6", "0.6-0.8", "0.8-0.9", "0.9-0.97", "0.97-0.999", "> 0.999"][: len(rows)]
    vals = [100 * r["neighbour_deep_move_matches"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(len(rows)), vals, width=0.62, color=BLUE)
    for b, v, r in zip(bars, vals, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, -13, f"n={r['positions']}", ha="center",
                color=MUTED, fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Cosine similarity to nearest stored position (above 0.999 is the same position)",
                  labelpad=16)
    ax.set_ylabel("Neighbour's best move is\nthis position's best move (%)")
    ax.set_title("Stored searches transfer only to the same position")
    save(fig, out, "similarity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--policy", default="50,200,800:reg")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    if (res / "stopper" / "report.json").exists():
        fig_frontier(json.loads((res / "stopper" / "report.json").read_text()), args.policy, out)
    if (res / "reuse.json").exists():
        fig_reuse(json.loads((res / "reuse.json").read_text()), out)
    if (res / "approx_retrieval.json").exists():
        fig_similarity(json.loads((res / "approx_retrieval.json").read_text()), out)


if __name__ == "__main__":
    main()
