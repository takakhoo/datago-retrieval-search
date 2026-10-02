"""Check the real engine: field parsing, perspective, and throughput."""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mikiri.board import Board, point_to_gtp
from mikiri.engine import open_katago

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="b18")
ap.add_argument("--gpu", type=int, default=None)
ap.add_argument("--size", type=int, default=19)
ap.add_argument("--visits", type=int, default=200)
ap.add_argument("--positions", type=int, default=256)
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("-v", "--verbose", action="store_true")
args = ap.parse_args()

with open_katago(args.net, args.gpu, stderr=sys.stderr if args.verbose else None) as eng:
    board = Board(args.size)
    t0 = time.time()
    res = eng.search(board, args.visits, 7.5, include_policy=True)
    print(f"first search {time.time() - t0:.1f}s (includes model load)")
    print("root: winrate %.3f score %.2f visits %d raw_wr %s" % (
        res.winrate, res.score_lead, res.visits, res.raw_winrate))
    for m in res.moves[:5]:
        print("  %-4s visits %4d wr %.3f lcb %.3f prior %.3f score %.2f pv %s" % (
            point_to_gtp(m.point, args.size), m.visits, m.winrate, m.lcb, m.prior, m.score_lead,
            " ".join(point_to_gtp(p, args.size) for p in m.pv[:5])))
    board.play(res.best.point)
    reply = eng.search(board, args.visits, 7.5)
    print("after best move, side-to-move winrate %.3f (should be near 1 - %.3f)" % (
        reply.winrate, res.winrate))

    rng = np.random.default_rng(0)
    boards = []
    for _ in range(args.positions):
        b = Board(args.size)
        for _ in range(int(rng.integers(0, 60))):
            r = eng.search(b, 1, 7.5, include_policy=True)
            pol = np.maximum(np.array(r.policy[:-1]), 0)
            b.play(int(rng.choice(len(pol), p=pol / pol.sum())))
        boards.append(b)
    eng.visits_served = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda b: eng.search(b, args.visits, 7.5), boards))
    dt = time.time() - t0
    print(f"{args.net} {args.size}x{args.size}: {eng.visits_served / dt:.0f} visits/s "
          f"({args.positions} positions x {args.visits} visits, {args.workers} concurrent)")
