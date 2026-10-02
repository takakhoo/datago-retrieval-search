"""Turn results/matches/*/summary.json into a table and the Elo-vs-compute figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures import BLUE, GRID, INK, MUTED, ORANGE, SURFACE, save  # noqa: F401  (sets rcParams)


def load(results: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text())
            for p in sorted(results.glob("*/summary.json"))}


def describe(name: str, s: dict) -> str:
    c = s["config"]
    if name.startswith("uni_"):
        return f"KataGo {c['path']} visits"
    parts = []
    if c.get("stopper"):
        parts.append("stopper")
    if c.get("memory"):
        parts.append("memory" + (" + deepening" if c.get("deepen") else ""))
    grant = c.get("grant") or c["budget"]
    label = f"Mikiri ({' + '.join(parts)}), {grant:g}-visit grant"
    if "long" in name:
        label += ", six-rung ladder"
    if c.get("net", "b18") != "b18":
        label += f", {c['net']} network"
    if c.get("size", 19) != 19:
        label += f", {c['size']}x{c['size']} board"
    if c.get("paired"):
        label += ", paired openings, no sampling"
    if name.startswith("pilot"):
        label += " (pilot, early model)"
    return label


def row(name: str, s: dict) -> dict:
    rows = s.get("nn_rows_per_move") or {}
    return {
        "run": name, "player": describe(name, s),
        "baseline": f"KataGo {s['config']['budget']}" + (
            f" ({s['config']['net']})" if s["config"].get("net", "b18") != "b18" else ""),
        "games": s["games"], "wld": f"{s['wins']}-{s['losses']}-{s['draws']}",
        "score": s["score"], "score_lo": s["score_lo"], "score_hi": s["score_hi"],
        "elo": s["elo"], "elo_lo": s["elo_lo"], "elo_hi": s["elo_hi"],
        "visits": s["visits_per_move"], "restart": s["restart_visits_per_move"],
        "base_visits": s["baseline_visits_per_move"],
        "rows": rows.get("mikiri"), "base_rows": rows.get("katago"),
        "hit_rate": (s.get("per_move") or {}).get("hit", 0.0),
    }


def markdown(rows: list[dict]) -> str:
    head = ("| Player | vs | Games | W-L-D | Score | Elo (95% CI) | Visits/move | "
            "Restart visits/move | NN evals/move (it : baseline) |\n|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        evals = f"{r['rows']:.0f} : {r['base_rows']:.0f}" if r["rows"] else "n/a"
        body += (f"| {r['player']} | {r['baseline']} | {r['games']} | {r['wld']} | {r['score']:.3f} | "
                 f"{r['elo']:+.0f} ({r['elo_lo']:+.0f} to {r['elo_hi']:+.0f}) | {r['visits']:.0f} | "
                 f"{r['restart']:.0f} | {evals} |\n")
    return head + body


def fig_elo_compute(rows: list[dict], out: Path, budget: int = 200) -> None:
    rows = [r for r in rows if r["baseline"] == f"KataGo {budget}" and r["rows"]
            and "paired" not in r["player"] and "board" not in r["player"]]
    uni = sorted((r for r in rows if r["run"].startswith("uni_")), key=lambda r: r["visits"])
    dg = [r for r in rows if not r["run"].startswith("uni_") and not r["run"].startswith("pilot")]
    measures = [("visits", "base_visits", "Visits per move"),
                ("restart", "base_visits", "Visits per move, every restart counted"),
                ("rows", "base_rows", "Network evaluations per move")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, (key, base, label) in zip(axes, measures):
        ux = [r[key] / r[base] for r in uni]
        ax.plot(ux, [r["elo"] for r in uni], color=ORANGE, marker="o", markersize=5,
                label="KataGo, uniform budget")
        for r in uni:
            ax.plot([r[key] / r[base]] * 2, [r["elo_lo"], r["elo_hi"]], color=ORANGE, linewidth=1)
        for i, r in enumerate(dg):
            x = r[key] / r[base]
            ax.plot([x, x], [r["elo_lo"], r["elo_hi"]], color=BLUE, linewidth=1.2)
            ax.plot([x], [r["elo"]], color=BLUE, marker="D", markersize=6, linestyle="none",
                    label="Mikiri" if i == 0 else None)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ticks = [0.5, 0.7, 1.0, 1.4, 2.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}x" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(f"{label}\n(relative to KataGo at {budget} visits)")
    axes[0].set_ylabel(f"Elo vs KataGo at {budget} visits")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Strength against compute, three ways of counting")
    save(fig, out, "elo_vs_compute")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/matches")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    summaries = load(Path(args.results))
    rows = [row(n, s) for n, s in summaries.items()]
    order = lambda r: (r["run"].startswith("pilot"), not r["run"].startswith("uni_"), r["run"])
    rows.sort(key=order)
    table = markdown(rows)
    Path(args.results, "TABLE.md").write_text(table)
    Path(args.results, "table.json").write_text(json.dumps(rows, indent=1))
    print(table)
    if sum(r["run"].startswith("uni_") for r in rows) >= 2:
        fig_elo_compute(rows, Path(args.out))


if __name__ == "__main__":
    main()
