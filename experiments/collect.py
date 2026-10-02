"""Build the search-ladder dataset.

Step 1 (`games`): self-play at a random per-game budget, to get positions from
the distribution that actually arises in play.

Step 2 (`label`): for each sampled position, search at every rung of a visit
ladder, then score each distinct move any rung chose with an independent
forced search of fixed size. The forced value is the yardstick: every
allocation policy is judged by the forced value of the move it ends up
playing, so no policy is graded by its own search.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mikiri.board import Board, point_to_gtp
from mikiri.engine import SearchResult, open_katago
from mikiri.match import GameConfig, play_game
from mikiri.players import KataGoPlayer, Temperature


def summarize_search(res: SearchResult, size: int, top: int = 12) -> dict:
    return {
        "winrate": res.winrate, "score": res.score_lead, "visits": res.visits,
        "raw_winrate": res.raw_winrate, "raw_score": res.raw_score_lead, "raw": res.raw,
        "moves": [
            [point_to_gtp(m.point, size), m.visits, round(m.winrate, 5), round(m.lcb, 5),
             round(m.prior, 5), round(m.score_lead, 3), round(m.score_stdev, 3)]
            for m in res.moves[:top]
        ],
    }


def cmd_games(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig(size=args.size, komi=args.komi)
    rng = np.random.default_rng(args.seed)
    budgets = np.exp(rng.uniform(np.log(args.min_visits), np.log(args.max_visits), args.n)).astype(int)
    seeds = np.random.SeedSequence(args.seed).spawn(args.n)
    temp = Temperature(early=0.8, late=0.15, halflife=24)
    lock = threading.Lock()
    with open_katago(args.net, args.gpu) as eng, out.open("a") as fh:
        def one(i: int):
            p = KataGoPlayer(eng, int(budgets[i]), temp, name="self")
            q = KataGoPlayer(eng, int(budgets[i]), temp, name="self2")
            rec = play_game(p, q, cfg, np.random.default_rng(seeds[i]))
            return i, rec

        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            for done, fut in enumerate(as_completed([pool.submit(one, i) for i in range(args.n)]), 1):
                i, rec = fut.result()
                with lock:
                    fh.write(json.dumps({"game": args.game_offset + i, "visits": int(budgets[i]),
                                         "moves": rec.moves, "winner": rec.winner,
                                         "reason": rec.reason, "size": args.size,
                                         "komi": args.komi}) + "\n")
                    fh.flush()
                if done % 20 == 0:
                    print(f"{done}/{args.n} games, {eng.visits_served / (time.time() - t0):.0f} visits/s",
                          flush=True)


def cmd_label(args) -> None:
    out = Path(args.out)
    ladder = [int(x) for x in args.ladder.split(",")]
    rng = np.random.default_rng(args.seed)
    jobs = []
    for line in open(args.games):
        g = json.loads(line)
        n = len(g["moves"])
        if n < 20:
            continue
        hi = n - 1
        picks = rng.choice(np.arange(args.skip_first, hi), size=min(args.per_game, hi - args.skip_first),
                           replace=False)
        for t in sorted(int(x) for x in picks):
            jobs.append((f"g{g['game']}m{t}", g, t))
    done_ids = set()
    if out.exists():
        done_ids = {json.loads(line)["id"] for line in open(out)}
    jobs = [j for j in jobs if j[0] not in done_ids]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} positions to label ({len(done_ids)} already done)", flush=True)

    lock = threading.Lock()
    with open_katago(args.net, args.gpu) as eng, out.open("a") as fh:
        def one(job):
            pid, g, t = job
            size, komi = g["size"], g["komi"]
            board = Board(size)
            for mv in g["moves"][:t]:
                board.play_gtp(mv)
            rec = {"id": pid, "game": g["game"], "move_number": t, "size": size, "komi": komi,
                   "moves": g["moves"][:t], "ladder": {}, "forced": {}}
            probe = eng.search(board, 1, komi, include_policy=True, include_ownership=True)
            rec["policy"] = [round(x, 4) for x in probe.policy]
            rec["ownership"] = [round(x, 3) for x in probe.ownership]
            candidates: list[int] = []
            for v in ladder:
                res = eng.search(board, v, komi)
                rec["ladder"][str(v)] = summarize_search(res, size)
                if res.best.point not in candidates:
                    candidates.append(res.best.point)
            for m in res.moves[:2]:
                if m.point not in candidates:
                    candidates.append(m.point)
            for p in candidates:
                f = eng.search(board, args.forced, komi, allow=[p])
                rec["forced"][point_to_gtp(p, size)] = {
                    "winrate": f.winrate, "score": f.score_lead, "visits": f.visits}
            return rec

        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            for done, fut in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                if done % 100 == 0:
                    dt = time.time() - t0
                    print(f"{done}/{len(jobs)} positions, {eng.visits_served / dt:.0f} visits/s, "
                          f"eta {dt / done * (len(jobs) - done) / 60:.0f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("games", "label"):
        p = sub.add_parser(name)
        p.add_argument("--net", default="b18")
        p.add_argument("--gpu", type=int, default=None)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=64)
    g = sub.choices["games"]
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--size", type=int, default=19)
    g.add_argument("--komi", type=float, default=7.0)
    g.add_argument("--min-visits", type=int, default=30)
    g.add_argument("--max-visits", type=int, default=300)
    g.add_argument("--game-offset", type=int, default=0)
    lab = sub.choices["label"]
    lab.add_argument("--games", required=True)
    lab.add_argument("--per-game", type=int, default=24)
    lab.add_argument("--skip-first", type=int, default=4)
    lab.add_argument("--ladder", default="50,100,200,400,800,1600,3200")
    lab.add_argument("--forced", type=int, default=1600)
    lab.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"games": cmd_games, "label": cmd_label}[args.cmd](args)


if __name__ == "__main__":
    main()
