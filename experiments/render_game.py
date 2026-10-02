"""Render one recorded game as an animated GIF and a still.

Left: the board. Right: visits spent on each move by each player, and the
winrate Mikiri reported. Memory hits show as zero-cost moves, early stops as
short bars, and long thinks as tall ones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from mikiri.board import BLACK, EMPTY, GTP_COLS, PASS, Board, gtp_to_point

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WOOD, LINE = "#e6c387", "#6b5330"


def draw_board(ax, board: Board, last: int | None = None, title: str = "") -> None:
    n = board.size
    ax.clear()
    ax.set_facecolor(WOOD)
    for i in range(n):
        ax.plot([0, n - 1], [i, i], color=LINE, linewidth=0.6, zorder=1)
        ax.plot([i, i], [0, n - 1], color=LINE, linewidth=0.6, zorder=1)
    if n == 19:
        for x in (3, 9, 15):
            for y in (3, 9, 15):
                ax.add_patch(Circle((x, y), 0.09, color=LINE, zorder=2))
    for p in np.flatnonzero(board.stones != EMPTY):
        y, x = divmod(int(p), n)
        black = board.stones[p] == BLACK
        ax.add_patch(Circle((x, n - 1 - y), 0.46, facecolor="#111111" if black else "#f7f7f5",
                            edgecolor="#111111", linewidth=0.5, zorder=3))
        if last is not None and p == last:
            ax.add_patch(Circle((x, n - 1 - y), 0.16, color="#f7f7f5" if black else "#111111", zorder=4))
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(list(GTP_COLS[:n]), fontsize=6, color=MUTED)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i + 1) for i in range(n)], fontsize=6, color=MUTED)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)


def render(rec: dict, out: Path, name: str, step: int = 1, fps: int = 6, budget: int | None = None,
           gif_dpi: int = 84) -> None:
    size, trace = rec["size"], rec["trace"]
    mikiri_black = rec["black"] == "mikiri"
    moves = [gtp_to_point(m, size) for m in rec["moves"]]
    n = len(trace)
    xs = np.arange(n)
    visits = np.array([t["v"] for t in trace], dtype=float)
    mine = np.array([t["p"] == "mikiri" for t in trace])
    hit = np.array([bool(t.get("hit")) for t in trace])
    wr = np.array([t["wr"] if t["p"] == "mikiri" else np.nan for t in trace])
    vmax = max(visits.max(), 1)
    ladder = sorted({int(v) for v, m in zip(visits, mine) if m and v > 0}) or [1]

    fig = plt.figure(figsize=(10.4, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.5, 1.0],
                          wspace=0.14, hspace=0.42, left=0.03, right=0.98, top=0.9, bottom=0.12)
    ax_b = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1], sharex=ax_v)
    result = "draw" if rec["winner"] == "draw" else (
        ("Mikiri" if (rec["winner"] == "B") == mikiri_black else "KataGo") + " wins"
        + (" by resignation" if rec["reason"] == "resign" else f" by {abs(rec['margin']):g}"))
    result = result.replace(" by resignation", ", resign")

    def frame(k: int):
        board = Board(size, superko=False)
        for p in moves[:k]:
            board.play(p)
        last = moves[k - 1] if k and moves[k - 1] != PASS else None
        side = "Black" if mikiri_black else "White"
        draw_board(ax_b, board, last, f"Mikiri ({side}) vs KataGo, move {k}"
                   + (f" ({result})" if k >= len(moves) else ""))
        # What Mikiri decided on its most recent move, shown under the board and as a ring.
        mine_idx = [i for i in range(min(k, n)) if trace[i]["p"] == "mikiri"]
        if mine_idx:
            t = trace[mine_idx[-1]]
            if t.get("hit"):
                note, ring = "from memory, 0 visits", AQUA
            elif t["v"] <= ladder[0]:
                note, ring = f"settled at {t['v']} visits", MUTED
            else:
                climbed = [v for v in ladder if v <= t["v"]]
                note = "kept thinking: " + " \u2192 ".join(f"{v:,}" for v in climbed) + " visits"
                ring = BLUE if t["v"] >= ladder[min(2, len(ladder) - 1)] else MUTED
            ax_b.text(0.0, -0.045, f"Mikiri's last move ({t['mv']}): {note}", transform=ax_b.transAxes,
                      fontsize=9, color=ring if ring != MUTED else INK, va="top",
                      fontweight="bold" if ring == BLUE else "normal")
            if t["mv"] != "pass" and ring != MUTED:
                py, px = divmod(gtp_to_point(t["mv"], size), size)
                if board.stones[py * size + px] != EMPTY:
                    ax_b.add_patch(Circle((px, size - 1 - py), 0.62, fill=False, edgecolor=ring,
                                          linewidth=2.2, zorder=5))
        for ax in (ax_v, ax_w):
            ax.clear()
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(length=0, colors=MUTED, labelsize=8)
        shown = xs < min(k, n)
        d, o = shown & mine & ~hit, shown & ~mine
        ax_v.bar(xs[o], visits[o], width=0.9, color=ORANGE, label="KataGo")
        ax_v.bar(xs[d], visits[d], width=0.9, color=BLUE, label="Mikiri")
        h = shown & mine & hit
        ax_v.scatter(xs[h], np.full(h.sum(), vmax * 0.02), marker="v", s=14, color=BLUE,
                     label="Mikiri: from memory (0 visits)")
        if budget:
            ax_v.axhline(budget, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax_v.set_xlim(-1, n)
        ax_v.set_ylim(0, vmax * 1.12)
        ax_v.set_ylabel("Visits spent", fontsize=8, color=MUTED)
        spent_d = visits[shown & mine].sum() / max((shown & mine).sum(), 1)
        spent_o = visits[o].sum() / max(o.sum(), 1)
        ax_v.set_title(f"Visits per move (mean so far: Mikiri {spent_d:.0f}, KataGo {spent_o:.0f})",
                       loc="left", fontsize=10, fontweight="bold", color=INK, pad=20)
        ax_v.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, frameon=False, ncol=3,
                    borderaxespad=0.1, handlelength=1.2, columnspacing=1.2)
        w = np.where(shown, wr, np.nan)
        ok = ~np.isnan(w)
        ax_w.plot(xs[ok], 100 * w[ok], color=BLUE, linewidth=1.8)
        ax_w.axhline(50, color=MUTED, linewidth=0.8)
        ax_w.set_ylim(0, 100)
        ax_w.set_ylabel("Mikiri winrate (%)", fontsize=8, color=MUTED)
        ax_w.set_xlabel("Move number", fontsize=8, color=MUTED)
        ax_w.set_title("Mikiri's own estimate of the game", loc="left", fontsize=10,
                       fontweight="bold", color=INK)

    out.mkdir(parents=True, exist_ok=True)
    frame(len(moves))
    fig.savefig(out / f"{name}.png", dpi=150, facecolor=SURFACE)
    frames = list(range(0, len(moves), step)) + [len(moves)] * (fps * 3)
    anim = FuncAnimation(fig, frame, frames=frames, interval=1000 / fps)
    anim.save(out / f"{name}.gif", writer=PillowWriter(fps=fps), dpi=gif_dpi)
    plt.close(fig)
    print("wrote", out / f"{name}.gif", "and .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--name", default="game")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.games)]
    render(recs[args.index], Path(args.out), args.name, args.step, budget=args.budget)


if __name__ == "__main__":
    main()
