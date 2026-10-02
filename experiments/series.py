"""DataGo vs KataGo over a series of games from the empty board.

Each player gets its own KataGo process, so the NN evaluations each one
actually ran can be read from its log at exit. Compute is reported three ways:
continuation visits (deepest search per move), restart visits (every search
requested), and NN rows (network evaluations executed).

With --path set to a single budget and no stopper or memory, the "datago"
player is plain KataGo at that budget, which gives sanity checks and the
Elo-per-visits calibration.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from datago.engine import open_katago
from datago.match import GameConfig, summarize
from datago.memory import Memory
from datago.players import DataGoPlayer, KataGoPlayer, Temperature
from datago.series import BudgetController, Deepener, Ledger, run_series
from datago.stopper import Stopper


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
    ap.add_argument("--path", default="200", help="DataGo visit ladder, comma separated")
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
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig(size=args.size, komi=args.komi)
    path = [int(x) for x in args.path.split(",")]
    temp = Temperature()

    log_d, log_k = out / "datago.katago.log", out / "baseline.katago.log"
    with open(log_d, "w") as ed, open(log_k, "w") as ek:
        eng_d = open_katago(args.net, args.gpu, stderr=ed)
        eng_k = open_katago(args.baseline_net or args.net, args.gpu, stderr=ek)
        stopper = Stopper.load(args.stopper, args.threshold) if args.stopper else None
        memory = None
        if args.memory:
            memory = Memory(args.memory_file or (out / "memory.jsonl"))
        datago = DataGoPlayer(eng_d, path, stopper, memory, args.store_max_move, temp, name="datago")
        katago = KataGoPlayer(eng_k, args.budget, temp, name="katago")

        ledger = Ledger(budget_per_move=args.budget)
        hook = None
        if memory is not None:
            hook = Deepener("datago", memory, eng_d, ledger, args.deepen or 1, args.komi,
                            enabled=args.deepen > 0)
        t0 = time.time()

        def progress(done, total, rec):
            if done % 50 == 0 or done == total:
                s = summarize(records_so_far, "datago")
                print(f"{done}/{total} games  score {s['score']:.3f}  "
                      f"datago {s['visits_per_move']:.1f} v/move  "
                      f"mem {len(memory) if memory is not None else 0}  "
                      f"thr {stopper.threshold if stopper else 0:.5f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)

        records_so_far: list = []
        accountant = hook or Deepener("datago", Memory(), eng_d, ledger, 1, args.komi, enabled=False)
        controller = None
        if stopper is not None and args.control:
            controller = BudgetController(stopper, ledger, share=args.play_share)

        def after(rec):
            records_so_far.append(rec)
            accountant(rec)
            if controller:
                controller(rec)

        records = run_series(datago, katago, cfg, args.games, args.seed, args.workers,
                             out / "games.jsonl", after, progress)
        elapsed = time.time() - t0
        eng_d.close()
        eng_k.close()

    s = summarize(records, "datago")
    turns_d = sum(r.turns["datago"] for r in records)
    turns_k = sum(r.turns["katago"] for r in records)
    rows_d, rows_k = nn_rows(log_d), nn_rows(log_k)
    s.update({
        "config": vars(args), "minutes": elapsed / 60,
        "baseline_visits_per_move": sum(r.visits["katago"] for r in records) / max(turns_k, 1),
        "datago_total_visits_per_move_incl_deepening":
            (sum(r.visits["datago"] for r in records) + ledger.deepening) / max(turns_d, 1),
        "nn_rows_per_move": {"datago": rows_d / max(turns_d, 1) if rows_d else None,
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
            return sum(r.counts["datago"].get("hit", 0) for r in rs) / max(
                sum(r.turns["datago"] for r in rs), 1)
        s["hit_rate_first_half"] = hit_rate(records[:half])
        s["hit_rate_second_half"] = hit_rate(records[half:])
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1, default=str))


if __name__ == "__main__":
    main()
