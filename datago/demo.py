"""Watch DataGo play KataGo move by move.

    python -m datago.demo --fake            # no GPU needed, toy engine, 9x9
    python -m datago.demo --net b18 --stopper models/stopper_b18.json

With --fake the engine is a stand-in that only mimics the protocol, so the
game shows the mechanics (ladder, early stops, memory hits) and says nothing
about Go strength. With a real KataGo it plays real games.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .board import BLACK, Board, point_to_gtp
from .engine import AnalysisEngine, open_katago
from .features import FEATURE_NAMES
from .match import GameConfig
from .memory import Memory
from .players import DataGoPlayer, KataGoPlayer, Temperature
from .stopper import Stopper

_LCB = FEATURE_NAMES.index("lcb_margin")


class MarginStopper:
    """Hand-written stand-in used only with the fake engine: stop when the best
    move's lower confidence bound clears every rival."""

    def should_stop(self, x: np.ndarray, rung: int = 0) -> bool:
        return x[_LCB] > 0.0


def describe(counts: dict, visits: int, path: list[int]) -> str:
    if counts.get("hit"):
        return "memory hit, 0 visits"
    rung = int(counts.get("rung", 0))
    if len(path) == 1:
        return f"{visits} visits"
    if rung == 0:
        return f"settled at {visits} visits"
    return f"kept thinking: {' -> '.join(str(v) for v in path[: rung + 1])} visits"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="use the built-in toy engine (no GPU)")
    ap.add_argument("--net", default="b18")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--stopper", default=None, help="stopper JSON (required without --fake)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--komi", type=float, default=7.0)
    ap.add_argument("--budget", type=int, default=None, help="KataGo's visits per move")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="print only game summaries")
    args = ap.parse_args()

    if args.fake:
        size, budget, path = args.size or 9, args.budget or 64, [16, 64, 256]
        engine = AnalysisEngine([sys.executable, "-m", "datago.fake_engine"])
        stopper = MarginStopper()
    else:
        if not args.stopper:
            ap.error("--stopper is required with a real engine")
        size, budget = args.size or 19, args.budget or 200
        engine = open_katago(args.net, args.gpu)
        stopper = Stopper.load(args.stopper, args.threshold)
        path = stopper.path

    memory = Memory()
    temp = Temperature()
    datago = DataGoPlayer(engine, path, stopper, memory, temp=temp, name="DataGo")
    katago = KataGoPlayer(engine, budget, temp, name="KataGo")
    cfg = GameConfig(size=size, komi=args.komi)
    rng = np.random.default_rng(args.seed)
    totals = {"DataGo": [0, 0], "KataGo": [0, 0]}

    for g in range(args.games):
        players = {BLACK: datago, 3 - BLACK: katago} if g % 2 == 0 else {BLACK: katago, 3 - BLACK: datago}
        board = Board(size)
        low = {1: 0, 2: 0}
        print(f"\n=== Game {g + 1}: {players[BLACK].name} is Black. Memory holds {len(memory)} positions. ===")
        winner = None
        while board.consecutive_passes() < 2 and len(board.moves) < 2 * size * size:
            color = board.to_move
            player = players[color]
            d = player.decide(board, cfg.komi, rng)
            totals[player.name][0] += d.visits
            totals[player.name][1] += 1
            if not args.quiet:
                note = describe(d.counts, d.visits, path) if player is datago else f"{d.visits} visits"
                print(f"{len(board.moves) + 1:4d} {player.name:7s} {'B' if color == BLACK else 'W'} "
                      f"{point_to_gtp(d.point, size):5s} winrate {100 * d.winrate:5.1f}%  {note}")
            low[color] = low[color] + 1 if d.winrate < cfg.resign_threshold else 0
            if low[color] >= cfg.resign_consecutive and len(board.moves) >= size * size // 4:
                winner = players[3 - color].name + " (resignation)"
                break
            board.play(d.point)
        if winner is None:
            b, w = board.area_score()
            diff = b - w - cfg.komi
            winner = "draw" if diff == 0 else f"{players[BLACK if diff > 0 else 3 - BLACK].name} by {abs(diff):g}"
        print(board)
        print(f"Result: {winner}")
        for name, (v, n) in totals.items():
            print(f"  {name}: {v / max(n, 1):.1f} visits per move so far over {n} moves")
        print(f"  memory: {len(memory)} positions, {memory.hits} hits in {memory.lookups} lookups")
    engine.close()


if __name__ == "__main__":
    main()
