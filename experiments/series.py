"""Mikiri vs KataGo over a series of games from the empty board.

Each player gets its own KataGo process, so the NN evaluations each one
actually ran can be read from its log at exit. Compute is reported three ways:
continuation visits (deepest search per move), restart visits (every search
requested), and NN rows (network evaluations executed).

With --path set to a single budget and no stopper or memory, the "mikiri"
player is plain KataGo at that budget, which gives sanity checks and the
Elo-per-visits calibration.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mikiri.engine import open_katago
import numpy as np

from mikiri.match import GameConfig, load_records, run_match, sample_openings, summarize
from mikiri.memory import Memory
from mikiri.players import MikiriPlayer, KataGoPlayer, Temperature
from mikiri.series import BudgetController, Deepener, Ledger, run_series
from mikiri.stopper import Stopper


def nn_rows(log: Path) -> int | None:
    found = re.findall(r"NN rows: (\d+)", log.read_text())
    return int(found[-1]) if found else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="runs/series")
    ap.add_argument("--net", default="b18")
    ap.add_argument("--baseline-net", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--komi", type=float, default=7.0)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--budget", type=int, default=200, help="baseline visits per move")
    ap.add_argument("--grant", type=float, default=None,
                    help="visits granted per Mikiri move (default: the baseline budget)")
    ap.add_argument("--path", default="200", help="Mikiri visit ladder, comma separated")
    ap.add_argument("--stopper", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--memory-file", default=None, help="start from (and append to) this memory")
    ap.add_argument("--store-max-move", type=int, default=80)
    ap.add_argument("--deepen", type=int, default=0, help="post-game deepening budget per entry")
    ap.add_argument("--control", action="store_true",
                    help="adapt the stopper threshold so total spending tracks the baseline budget")
    ap.add_argument("--play-share", type=float, default=1.0,
                    help="share of the grant the controller aims to spend during play")
    ap.add_argument("--trace", action="store_true", help="record every move's cost and winrate")
    ap.add_argument("--paired", type=int, default=0,
                    help="instead of a series, play this many sampled openings twice with colours swapped")
    ap.add_argument("--opening-plies", type=int, default=12)
    ap.add_argument("--greedy", action="store_true",
                    help="both players always play the engine's top move (no sampling)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted series: keep the games already recorded and play the rest")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig(size=args.size, komi=args.komi, keep_trace=args.trace)
    path = [int(x) for x in args.path.split(",")]
    temp = None if args.greedy else Temperature()

    log_d, log_k = out / "mikiri.katago.log", out / "baseline.katago.log"
    with open(log_d, "w") as ed, open(log_k, "w") as ek:
        eng_d = open_katago(args.net, args.gpu, stderr=ed)
        eng_k = open_katago(args.baseline_net or args.net, args.gpu, stderr=ek)
        stopper = Stopper.load(args.stopper, args.threshold) if args.stopper else None
        memory = None
        if args.memory:
            memory = Memory(args.memory_file or (out / "memory.jsonl"))
        mikiri = MikiriPlayer(eng_d, path, stopper, memory, args.store_max_move, temp, name="mikiri")
        katago = KataGoPlayer(eng_k, args.budget, temp, name="katago")

        grant = args.grant if args.grant is not None else args.budget
        ledger = Ledger(budget_per_move=grant)
        hook = None
        if memory is not None:
            hook = Deepener("mikiri", memory, eng_d, ledger, args.deepen or 1, args.komi,
                            enabled=args.deepen > 0)
        t0 = time.time()

        def progress(done, total, rec):
            done += len(earlier)
            if done % 50 == 0 or done == total:
                s = summarize(records_so_far, "mikiri")
                print(f"{done}/{total} games  score {s['score']:.3f}  "
                      f"mikiri {s['visits_per_move']:.1f} v/move  "
                      f"mem {len(memory) if memory is not None else 0}  "
                      f"thr {stopper.threshold if stopper else 0:.5f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)

        records_so_far: list = []
        earlier: list = []
        if args.resume and not args.paired and (out / "games.jsonl").exists():
            earlier = load_records(out / "games.jsonl")
            print(f"resuming: {len(earlier)} games already recorded", flush=True)
        accountant = hook or Deepener("mikiri", Memory(), eng_d, ledger, 1, args.komi, enabled=False)
        if stopper is not None and args.control:
            controller = BudgetController(stopper, grant, share=args.play_share)
            mikiri.on_decision = controller.record

        def after(rec):
            records_so_far.append(rec)
            accountant(rec)

        was_enabled = getattr(accountant, "enabled", False)
        accountant.enabled = False  # replaying old records must not trigger new deepening
        for rec in earlier:
            after(rec)
        accountant.enabled = was_enabled
        done = {2 * r.opening + (0 if r.black == "mikiri" else 1) for r in earlier}

        if args.paired:
            openings = sample_openings(eng_k, args.paired, cfg, args.opening_plies,
                                       np.random.default_rng(args.seed), workers=args.workers)
            (out / "openings.json").write_text(json.dumps(openings))
            print(f"sampled {len(openings)} balanced openings", flush=True)
            t0 = time.time()
            records = run_match(mikiri, katago, cfg, openings, args.seed, args.workers,
                                out / "games.jsonl", progress, after)
        else:
            new = run_series(mikiri, katago, cfg, args.games, args.seed, args.workers,
                             out / "games.jsonl", after, progress, skip=done)
            records = earlier + new
        elapsed = time.time() - t0
        eng_d.close()
        eng_k.close()

    s = summarize(records, "mikiri")
    # The engine logs only cover games played by this process, so evaluations
    # per move are computed over those games.
    fresh = records[len(earlier):]
    turns_d = sum(r.turns["mikiri"] for r in fresh)
    turns_k = sum(r.turns["katago"] for r in fresh)
    rows_d, rows_k = nn_rows(log_d), nn_rows(log_k)
    s.update({
        "config": vars(args), "minutes": elapsed / 60, "resumed_from_games": len(earlier),
        "baseline_visits_per_move": sum(r.visits["katago"] for r in records) / max(
            sum(r.turns["katago"] for r in records), 1),
        "mikiri_total_visits_per_move_incl_deepening":
            (sum(r.visits["mikiri"] for r in records) + ledger.deepening) / max(
                sum(r.turns["mikiri"] for r in records), 1),
        "nn_rows_per_move": {"mikiri": rows_d / max(turns_d, 1) if rows_d else None,
                             "katago": rows_k / max(turns_k, 1) if rows_k else None},
        "ledger": {"granted": ledger.granted, "played": ledger.played,
                   "deepening": ledger.deepening, "deepened_entries": ledger.deepened},
        "memory_entries": len(memory) if memory is not None else 0,
        "final_threshold": stopper.threshold if stopper else None,
        "mean_game_length": sum(len(r.moves) for r in records) / len(records),
    })
    half = len(records) // 2
    if memory is not None:
        def hit_rate(rs):
            return sum(r.counts["mikiri"].get("hit", 0) for r in rs) / max(
                sum(r.turns["mikiri"] for r in rs), 1)
        s["hit_rate_first_half"] = hit_rate(records[:half])
        s["hit_rate_second_half"] = hit_rate(records[half:])
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1, default=str))


if __name__ == "__main__":
    main()
