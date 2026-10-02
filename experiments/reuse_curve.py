"""How often does a position recur across games, up to symmetry?

Replays games in order and asks, for every position, whether its canonical key
already occurred in an earlier game. This is the ceiling on what an exact-match
memory can ever serve, before any question of whether the stored search helps.
"""
import argparse
import gzip
import json
from collections import defaultdict

import numpy as np

from datago.board import Board

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("--out", default=None)
args = ap.parse_args()

opener = gzip.open if args.games.endswith(".gz") else open
games = [json.loads(l) for l in opener(args.games, "rt")]
seen: set[str] = set()
hits = defaultdict(list)      # move number -> list of 0/1
by_game = []
for g in games:
    board = Board(g["size"])
    keys = []
    for t, mv in enumerate(g["moves"]):
        keys.append(board.key(g["komi"])[0])
        board.play_gtp(mv)
    row = [int(k in seen) for k in keys]
    for t, h in enumerate(row):
        hits[t].append(h)
    by_game.append((sum(row), len(row), next((t for t, h in enumerate(row) if not h), len(row))))
    seen.update(keys)

n = len(games)
half = n // 2
print(f"{n} games, {sum(b[1] for b in by_game)} positions, {len(seen)} distinct up to symmetry")
print("move  hit-rate(all games)  hit-rate(second half of games)")
table = []
for t in list(range(0, 16)) + [20, 30, 50]:
    a = float(np.mean(hits[t])) if hits[t] else 0.0
    b = float(np.mean(hits[t][half:])) if len(hits[t]) > half else 0.0
    table.append({"move": t, "hit_rate": a, "hit_rate_second_half": b})
    print(f"{t:4d}  {a:6.3f}  {b:6.3f}")
late = by_game[half:]
depth = [b[2] for b in late]
frac = sum(b[0] for b in late) / sum(b[1] for b in late)
print(f"second half of games: mean first-miss move {np.mean(depth):.2f}, "
      f"max {max(depth)}, share of all positions served from memory {frac:.4f}")
if args.out:
    json.dump({"games": n, "table": table, "mean_first_miss": float(np.mean(depth)),
               "max_first_miss": int(max(depth)), "share_served": frac}, open(args.out, "w"), indent=1)
