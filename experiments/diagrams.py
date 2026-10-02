"""Explanatory diagrams for the README and paper.

pipeline         how DataGo decides one move
symmetry         eight orientations of a position share one memory key
case_study       a real position where thinking longer changed the move
regret_anatomy   how concentrated search error is, and when in the game it occurs
rate_rule        why dividing by the cost of the next rung matters
v1_bug           the protocol error behind the first version's results
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from datago.board import BLACK, WHITE, Board, gtp_to_point, point_to_gtp, symmetry_tables
from datago.ladder import result_from_summary
from datago.features import extract
from datago.stopper import Stopper, trajectory
from figures import AQUA, BLUE, GRAY, GRID, INK, MUTED, ORANGE, SURFACE, save
from render_game import draw_board

SOFT_BLUE, SOFT_ORANGE, SOFT_AQUA, SOFT_GRAY = "#e3eefb", "#fde9e0", "#dff4ec", "#f0efec"


def box(ax, x, y, w, h, text, face=SOFT_GRAY, edge=GRAY, size=9, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.35)


def arrow(ax, a, b, color=MUTED, label=None, offset=(0, 0.14), style="-|>", rad=0.0, size=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color,
                                 linewidth=1.3, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], label, ha="center",
                va="center", fontsize=size, color=color)


def pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.9))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.9)
    ax.axis("off")
    ax.text(0.1, 5.62, "How DataGo decides one move", fontsize=13, fontweight="bold", color=INK)

    box(ax, 0.1, 2.9, 1.3, 0.9, "Position", face="#ffffff", weight="bold", size=10)
    box(ax, 1.9, 2.7, 2.5, 1.3, "Seen before?\nSame position in any\nof 8 orientations,\nby any move order",
        face=SOFT_AQUA, edge=AQUA, size=8.2)
    arrow(ax, (1.4, 3.35), (1.9, 3.35))
    box(ax, 2.15, 0.5, 2.0, 1.0, "Play from the\nstored search\n0 visits", face=SOFT_AQUA, edge=AQUA, size=9,
        weight="bold")
    arrow(ax, (3.15, 2.7), (3.15, 1.5), color=AQUA, label="yes", offset=(0.25, 0))

    xs = [5.0, 6.9, 8.8, 10.7]
    rungs = ["50", "200", "800", "3,200"]
    for i, (x, v) in enumerate(zip(xs, rungs)):
        box(ax, x, 2.9, 1.3, 0.9, f"Search\n{v} visits", face=SOFT_BLUE, edge=BLUE, size=9.5, weight="bold")
        if i < 3:
            ax.add_patch(Circle((x + 0.65, 1.95), 0.33, facecolor="#ffffff", edgecolor=BLUE, linewidth=1.3))
            ax.text(x + 0.65, 1.95, "stop?", ha="center", va="center", fontsize=8, color=BLUE)
            arrow(ax, (x + 0.65, 2.9), (x + 0.65, 2.28), color=BLUE)
            arrow(ax, (x + 0.98, 1.95), (xs[i + 1] + 0.3, 2.9), color=BLUE, label="no", offset=(0.12, -0.22),
                  rad=-0.25)
            arrow(ax, (x + 0.65, 1.62), (x + 0.65, 1.02), color=ORANGE, label="yes", offset=(0.25, 0))
        else:
            arrow(ax, (x + 0.65, 2.9), (x + 0.65, 1.02), color=ORANGE)
    arrow(ax, (4.4, 3.35), (5.0, 3.35), color=MUTED, label="no")
    box(ax, 5.0, 0.3, 7.0, 0.7, "Play the engine's move from the last search, and store that search",
        face=SOFT_ORANGE, edge=ORANGE, size=9.5, weight="bold")

    box(ax, 5.0, 4.25, 7.0, 1.1, "", face="#ffffff", edge=BLUE)
    ax.text(8.5, 5.02, "Stop when   predicted regret left  ÷  visits to the next rung   <   λ",
            ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(8.5, 4.56, "Regret is predicted from the search just finished: how contested the root is,\n"
            "how the search disagrees with the network, and how close the game is.",
            ha="center", va="center", fontsize=7.8, color=MUTED, linespacing=1.35)
    box(ax, 0.1, 4.25, 4.3, 1.1, "Ledger\nEvery move is granted the baseline's budget.\n"
        "A controller tunes λ so that total\nspending equals the grant.", face="#ffffff", edge=GRAY, size=8.2)
    arrow(ax, (4.4, 4.8), (5.0, 4.8), color=GRAY)
    save(fig, out, "pipeline")


def mini_board(ax, board: Board, marks: dict[int, tuple[str, str]] | None = None, title: str = "") -> None:
    draw_board(ax, board, None, title)
    ax.set_xticks([])
    ax.set_yticks([])
    n = board.size
    for p, (label, color) in (marks or {}).items():
        y, x = divmod(p, n)
        ax.add_patch(Circle((x, n - 1 - y), 0.42, facecolor=color, edgecolor="#111111", linewidth=0.6, zorder=5))
        ax.text(x, n - 1 - y, label, ha="center", va="center", fontsize=8, color="#ffffff",
                fontweight="bold", zorder=6)


def symmetry(out: Path) -> None:
    size = 9
    base = Board(size)
    for mv in ["G7", "C3", "C7", "F3", "G4", "E5", "D2"]:
        base.play_gtp(mv)
    answer = gtp_to_point("H3", size)
    to_sym, _ = symmetry_tables(size)
    key0, sym0 = base.key(7.0)
    fig = plt.figure(figsize=(12.6, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.35, 1.5, 0.05], wspace=0.08, hspace=0.12,
                          left=0.02, right=0.98, top=0.86, bottom=0.04)
    for s in range(8):
        b = Board(size)
        for color, p in base.moves:
            b.play(int(to_sym[s][p]), color)
        key, sym = b.key(7.0)
        assert key == key0
        ax = fig.add_subplot(gs[s // 4, s % 4])
        mini_board(ax, b, {b.from_canonical(base.to_canonical(answer, sym0), sym): ("A", BLUE)})
    ax = fig.add_subplot(gs[:, 5])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box(ax, 0.02, 0.36, 0.96, 0.3, f"one memory entry\nkey {key0[:12]}...\nstored move A", face=SOFT_AQUA,
        edge=AQUA, size=10, weight="bold")
    ax.text(0.5, 0.2, "A is stored once, in canonical\norientation, and mapped back\nthrough each board's own symmetry.",
            ha="center", va="center", fontsize=8.5, color=MUTED, linespacing=1.4)
    ax.annotate("", xy=(0.0, 0.51), xytext=(-0.28, 0.51), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AQUA, linewidth=1.6))
    fig.text(0.02, 0.93, "Eight boards, one position: rotations and reflections share a single memory key",
             fontsize=12, fontweight="bold", color=INK)
    save(fig, out, "symmetry")


def pick_case(recs: list[dict]) -> dict:
    best, best_score = None, -1.0
    for r in recs:
        lad, forced = r["ladder"], r["forced"]
        if not (70 <= r["move_number"] <= 170):
            continue
        m50, m200, m800, m3200 = (lad[k]["moves"][0][0] for k in ("50", "200", "800", "3200"))
        if not (m50 == m200 != m800 == m3200) or "pass" in (m200, m800):
            continue
        gain = forced[m800]["winrate"] - forced[m200]["winrate"]
        wr = forced[m800]["winrate"]
        if gain > best_score and 0.3 < wr < 0.75:
            best, best_score = r, gain
    return best


def case_study(recs: list[dict], stopper: Stopper, out: Path) -> None:
    r = pick_case(recs)
    size = r["size"]
    board = Board(size, superko=False)
    for mv in r["moves"]:
        board.play_gtp(mv)
    path = stopper.path
    rows, prev_x, prev_move, changes = [], None, None, 0
    for j, v in enumerate(path):
        res = result_from_summary(r["ladder"][str(v)], size)
        x = extract(res, board)
        t, changes = trajectory(prev_x, x, prev_move, res.best.point, changes)
        move = r["ladder"][str(v)]["moves"][0][0]
        score = stopper.score(np.concatenate([x, t]), j) if j < len(path) - 1 else None
        rows.append({"visits": v, "move": move, "search_wr": res.winrate,
                     "true_wr": r["forced"][move]["winrate"], "score": score})
        prev_x, prev_move = x, res.best.point
    shallow, deep = rows[1]["move"], rows[2]["move"]
    lam = float(np.interp(200, [c["cost_continue"] for c in stopper.meta["calibration"]],
                          [c["threshold"] for c in stopper.meta["calibration"]]))

    fig = plt.figure(figsize=(12.6, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.08, left=0.03, right=0.98, top=0.88, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    side = "Black" if board.to_move == BLACK else "White"
    mini_board(ax, board, {gtp_to_point(shallow, size): ("A", ORANGE), gtp_to_point(deep, size): ("B", BLUE)},
               f"Move {r['move_number'] + 1}, {side} to play")
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(0, 9.6, "What each rung of the ladder saw", fontsize=11, fontweight="bold", color=INK)
    heads = ["Visits", "Engine's\nmove", "Search says\n(winrate)", "Independent\ncheck", "Stopper:\nstake / cost", "Decision"]
    cols = [0.2, 1.6, 3.1, 4.9, 6.6, 8.3]
    for c, h in zip(cols, heads):
        ax2.text(c, 8.6, h, fontsize=8.5, color=MUTED, va="center", linespacing=1.3)
    ax2.plot([0.1, 9.9], [7.95, 7.95], color=GRID, linewidth=1)
    for i, row in enumerate(rows):
        y = 7.2 - 1.25 * i
        tag, color = ("A", ORANGE) if row["move"] == shallow else ("B", BLUE)
        ax2.text(cols[0], y, f"{row['visits']:,}", fontsize=10.5, color=INK, va="center", fontweight="bold")
        ax2.scatter([cols[1] + 0.25], [y], s=330, color=color, edgecolors="#111111", linewidths=0.6, zorder=3)
        ax2.text(cols[1] + 0.25, y, tag, ha="center", va="center", fontsize=8.5, color="#ffffff",
                 fontweight="bold", zorder=4)
        ax2.text(cols[1] + 0.7, y, row["move"], fontsize=9.5, color=INK, va="center")
        ax2.text(cols[2], y, f"{100 * row['search_wr']:.1f}%", fontsize=10, color=INK, va="center")
        ax2.text(cols[3], y, f"{100 * row['true_wr']:.1f}%", fontsize=10, color=INK, va="center")
        if row["score"] is None:
            ax2.text(cols[4], y, "last rung", fontsize=9, color=MUTED, va="center")
            ax2.text(cols[5], y, "play", fontsize=10, color=INK, va="center")
        else:
            go = row["score"] >= lam
            ax2.text(cols[4], y, f"{row['score'] / lam:.1f} x λ", fontsize=10, color=INK, va="center")
            ax2.text(cols[5], y, "keep thinking" if go else "stop and play", fontsize=10,
                     color=BLUE if go else ORANGE, va="center", fontweight="bold")
    gain = 100 * (r["forced"][deep]["winrate"] - r["forced"][shallow]["winrate"])
    ax2.text(0.1, 1.35, f"A is what a fixed 200-visit search plays. An independent {r['forced'][shallow]['visits']:,}-visit search of each\n"
             f"move puts B {gain:.1f} points of winrate ahead of A. The stopper's stake estimate stayed above\n"
             f"its threshold λ at 50 and at 200 visits, so DataGo went on to 800 and found B.",
             fontsize=9, color=MUTED, va="center", linespacing=1.5)
    fig.text(0.03, 0.94, "A real position where thinking longer changed the move",
             fontsize=13, fontweight="bold", color=INK)
    save(fig, out, "case_study")
    (out.parent / "case_study.json").write_text(json.dumps({"id": r["id"], "rows": rows, "lambda": lam}, indent=1))


def regret_anatomy(recs: list[dict], out: Path) -> None:
    def regret(r, k):
        forced = r["forced"]
        return max(f["winrate"] for f in forced.values()) - forced[r["ladder"][k]["moves"][0][0]]["winrate"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for k, color in (("50", GRAY), ("200", BLUE), ("800", ORANGE)):
        reg = np.sort(np.array([regret(r, k) for r in recs]))[::-1]
        share = np.concatenate([[0], np.cumsum(reg) / reg.sum()])
        axes[0].plot(100 * np.linspace(0, 1, len(share)), 100 * share, color=color, label=f"{k} visits")
    axes[0].plot([0, 100], [0, 100], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[0].set_xlim(0, 40)
    axes[0].set_ylim(0, 102)
    axes[0].set_xlabel("Positions, worst first (%)")
    axes[0].set_ylabel("Share of all winrate lost (%)")
    axes[0].set_title("A few positions hold almost all the error")
    axes[0].legend(loc="lower right")
    bins = list(range(0, 260, 20))
    for k, color in (("50", GRAY), ("200", BLUE), ("800", ORANGE)):
        ys = []
        for a in bins:
            vals = [regret(r, k) for r in recs if a <= r["move_number"] < a + 20]
            ys.append(100 * np.mean(vals) if len(vals) >= 40 else np.nan)
        axes[1].plot([a + 10 for a in bins], ys, color=color, marker="o", markersize=4, label=f"{k} visits")
    axes[1].set_ylim(0, None)
    axes[1].set_xlabel("Move number")
    axes[1].set_ylabel("Mean winrate lost per move (%)")
    axes[1].set_title("and they come after the opening")
    save(fig, out, "regret_anatomy")


def rate_rule(results: Path, out: Path) -> None:
    rep = json.loads((results / "ladder/report.json").read_text())
    rows = rep["heuristic_rules"]["rows"]
    ab = json.loads((results / "stopper/ablation.json").read_text())["variants"]
    names = [("random", "Random\nstopping"), ("v1 entropy gate", "v1 entropy\ngate"),
             ("lcb margin", "Confidence\nmargin (LCB)")]
    flat = [next(x["multiplier"] for x in rows if x["rule"] == n and x["scaling"] == "flat" and x["budget"] == 200.0)
            for n, _ in names] + [ab["flat threshold (no rate rule)"]["200"]["continue"]]
    rate = [next(x["multiplier"] for x in rows if x["rule"] == n and x["scaling"] == "rate" and x["budget"] == 200.0)
            for n, _ in names] + [ab["full model"]["200"]["continue"]]
    labels = [l for _, l in names] + ["Learned\nstopper"]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = np.arange(len(labels))
    b1 = ax.bar(x - 0.19, flat, width=0.34, color=GRAY, label="Flat threshold")
    b2 = ax.bar(x + 0.19, rate, width=0.34, color=BLUE, label="Rate rule: divide by cost of the next rung")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.2f}x",
                    ha="center", fontsize=8.5, color=INK)
    ax.axhline(1.0, color=ORANGE, linewidth=1.2)
    ax.text(-0.62, 1.03, "uniform search", color=ORANGE, fontsize=8.5, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, len(labels) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(rate) * 1.28)
    ax.set_ylabel("Compute multiplier at 200 visits")
    ax.set_title("The same signals, with and without the rate rule")
    ax.legend(loc="upper left")
    save(fig, out, "rate_rule")


def v1_bug(out: Path) -> None:
    board = Board(19, superko=False)
    stones = [("Q16", BLACK, "1"), ("D4", BLACK, "2"), ("Q4", BLACK, "3"), ("D16", WHITE, "1")]
    for mv, color, _ in stones:
        board.play_gtp(mv, color)
    fig, ax = plt.subplots(figsize=(4.6, 4.9), facecolor=SURFACE)
    draw_board(ax, board, None, "v1: one Black turn, three black stones")
    ax.set_xticks([])
    ax.set_yticks([])
    for mv, color, label in stones:
        y, x = divmod(gtp_to_point(mv, 19), 19)
        ax.text(x, 18 - y, label, ha="center", va="center", fontsize=8, fontweight="bold",
                color="#ffffff" if color == BLACK else "#111111", zorder=6)
    ax.text(9, -1.25, "v1 called an analyse-and-play command for the standard search, the deep search,\n"
            "and a recursive search. Each call placed a stone before White replied.",
            ha="center", va="top", fontsize=7.5, color=MUTED, linespacing=1.4)
    save(fig, out, "v1_bug")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ladder19.jsonl.gz")
    ap.add_argument("--stopper", default="models/stopper_b18.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.results) / "figures"
    want = set(args.only.split(",")) if args.only else None
    recs = None

    def data():
        nonlocal recs
        if recs is None:
            recs = [json.loads(l) for l in gzip.open(args.data, "rt")]
        return recs

    jobs = {
        "pipeline": lambda: pipeline(out),
        "symmetry": lambda: symmetry(out),
        "v1_bug": lambda: v1_bug(out),
        "rate_rule": lambda: rate_rule(Path(args.results), out),
        "regret_anatomy": lambda: regret_anatomy(data(), out),
        "case_study": lambda: case_study(data(), Stopper.load(args.stopper), out),
    }
    for name, job in jobs.items():
        if want is None or name in want:
            job()


if __name__ == "__main__":
    main()
