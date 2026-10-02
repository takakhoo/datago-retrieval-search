"""Game loop and parallel match runner with color-swapped opening pairs."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .board import BLACK, EMPTY, PASS, WHITE, Board, gtp_to_point, point_to_gtp
from .engine import AnalysisEngine
from .players import Player


@dataclass
class GameRecord:
    black: str
    white: str
    size: int
    komi: float
    opening: int
    moves: list[str]
    winner: str            # "B", "W", or "draw"
    reason: str            # "score", "resign", "move-cap"
    margin: float | None
    visits: dict[str, int] = field(default_factory=dict)
    turns: dict[str, int] = field(default_factory=dict)
    flags: dict[str, dict[str, int]] = field(default_factory=dict)
    trace: list[dict] | None = None

    def score_for(self, name: str) -> float:
        if self.winner == "draw":
            return 0.5
        return float((self.winner == "B") == (name == self.black))

    def sgf(self) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        body = []
        for i, mv in enumerate(self.moves):
            color = "B" if i % 2 == 0 else "W"
            if mv == "pass":
                body.append(f";{color}[]")
            else:
                y, x = divmod(gtp_to_point(mv, self.size), self.size)
                body.append(f";{color}[{letters[x]}{letters[y]}]")
        if self.winner == "draw":
            result = "0"
        elif self.reason == "resign":
            result = f"{self.winner}+R"
        else:
            result = f"{self.winner}+{abs(self.margin):g}"
        return (f"(;GM[1]FF[4]SZ[{self.size}]KM[{self.komi}]RU[Tromp-Taylor]"
                f"PB[{self.black}]PW[{self.white}]RE[{result}]" + "".join(body) + ")")


@dataclass
class GameConfig:
    size: int = 19
    komi: float = 7.5
    max_moves: int | None = None
    resign_threshold: float = 0.03
    resign_consecutive: int = 3
    resign_min_move_frac: float = 0.25
    keep_trace: bool = False


def play_game(black: Player, white: Player, cfg: GameConfig, rng: np.random.Generator,
              opening: Sequence[int] = (), opening_id: int = -1) -> GameRecord:
    board = Board(cfg.size)
    for p in opening:
        board.play(p)
    players = {BLACK: black, WHITE: white}
    visits = {black.name: 0, white.name: 0}
    turns = {black.name: 0, white.name: 0}
    flags: dict[str, dict[str, int]] = {black.name: {}, white.name: {}}
    low_streak = {BLACK: 0, WHITE: 0}
    trace = [] if cfg.keep_trace else None
    max_moves = cfg.max_moves or 2 * cfg.size * cfg.size
    min_resign_move = int(cfg.resign_min_move_frac * cfg.size * cfg.size)
    winner, reason, margin = None, "score", None

    while board.consecutive_passes() < 2 and len(board.moves) < max_moves:
        color = board.to_move
        player = players[color]
        d = player.decide(board, cfg.komi, rng)
        visits[player.name] += d.visits
        turns[player.name] += 1
        for k, v in d.flags.items():
            flags[player.name][k] = flags[player.name].get(k, 0) + int(bool(v))
        if trace is not None:
            trace.append({"n": len(board.moves), "p": player.name, "v": d.visits,
                          "wr": round(d.winrate, 4), "mv": point_to_gtp(d.point, cfg.size),
                          **{k: bool(v) for k, v in d.flags.items()}})
        low_streak[color] = low_streak[color] + 1 if d.winrate < cfg.resign_threshold else 0
        if low_streak[color] >= cfg.resign_consecutive and len(board.moves) >= min_resign_move:
            winner, reason = ("W" if color == BLACK else "B"), "resign"
            break
        board.play(d.point)

    if winner is None:
        b, w = board.area_score()
        margin = b - w - cfg.komi
        w_id = board.winner(cfg.komi)
        winner = "draw" if w_id == EMPTY else "B" if w_id == BLACK else "W"
        if len(board.moves) >= max_moves:
            reason = "move-cap"
    return GameRecord(
        black=black.name, white=white.name, size=cfg.size, komi=cfg.komi, opening=opening_id,
        moves=[point_to_gtp(p, cfg.size) for _, p in board.moves], winner=winner, reason=reason,
        margin=margin, visits=visits, turns=turns, flags=flags, trace=trace,
    )


def sample_openings(engine: AnalysisEngine, n: int, cfg: GameConfig, plies: int,
                    rng: np.random.Generator, visits: int = 100, temperature: float = 1.0,
                    balance: float = 0.12, check_visits: int = 400,
                    workers: int = 16) -> list[list[int]]:
    """Draw distinct, roughly even openings by sampling search visit counts.

    An opening is kept only if a check search puts the side to move within
    `balance` of 50%, so neither color starts with a decided game.
    """
    def one(seed: int) -> list[int] | None:
        r = np.random.default_rng(seed)
        board = Board(cfg.size)
        for _ in range(plies):
            res = engine.search(board, visits, cfg.komi)
            v = np.array([m.visits for m in res.moves], dtype=np.float64) ** (1.0 / temperature)
            cand = [m.point for m in res.moves]
            p = cand[int(r.choice(len(cand), p=v / v.sum()))]
            if p == PASS:
                return None
            board.play(p)
        check = engine.search(board, check_visits, cfg.komi)
        if abs(check.winrate - 0.5) > balance:
            return None
        return [p for _, p in board.moves]

    out: list[list[int]] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(workers) as pool:
        while len(out) < n:
            seeds = rng.integers(0, 2**62, size=max(workers, 2 * (n - len(out))))
            for op in pool.map(one, [int(s) for s in seeds]):
                if op is None:
                    continue
                board = Board(cfg.size)
                for p in op:
                    board.play(p)
                key = board.key(cfg.komi)[0]
                if key not in seen and len(out) < n:
                    seen.add(key)
                    out.append(op)
    return out


def run_match(a: Player, b: Player, cfg: GameConfig, openings: Sequence[Sequence[int]],
              seed: int = 0, workers: int = 32, out_path: str | Path | None = None,
              progress: Callable[[int, int, GameRecord], None] | None = None) -> list[GameRecord]:
    """Play every opening twice with colors swapped. Records are appended to out_path as JSONL."""
    jobs = []
    for i, op in enumerate(openings):
        jobs.append((i, a, b, op))
        jobs.append((i, b, a, op))
    seeds = np.random.SeedSequence(seed).spawn(len(jobs))
    lock = threading.Lock()
    records: list[GameRecord] = []
    fh = open(out_path, "a") if out_path else None

    def run(j: int) -> GameRecord:
        i, black, white, op = jobs[j]
        return play_game(black, white, cfg, np.random.default_rng(seeds[j]), op, i)

    try:
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(run, j) for j in range(len(jobs))]
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    records.append(rec)
                    if fh:
                        fh.write(json.dumps(asdict(rec)) + "\n")
                        fh.flush()
                    if progress:
                        progress(len(records), len(jobs), rec)
    finally:
        if fh:
            fh.close()
    return records


def load_records(path: str | Path) -> list[GameRecord]:
    with open(path) as f:
        return [GameRecord(**json.loads(line)) for line in f if line.strip()]


def summarize(records: Sequence[GameRecord], name: str) -> dict:
    """Score, per-opening pair scores, and compute usage for one player."""
    from .stats import paired_summary

    mine = [r for r in records if name in (r.black, r.white)]
    by_opening: dict[int, list[float]] = {}
    for r in mine:
        by_opening.setdefault(r.opening, []).append(r.score_for(name))
    pair_scores = [float(np.mean(v)) for v in by_opening.values() if len(v) == 2]
    opp_names = {r.white if r.black == name else r.black for r in mine}

    def per_move(n: str) -> float:
        v = sum(r.visits.get(n, 0) for r in mine)
        t = sum(r.turns.get(n, 0) for r in mine)
        return v / max(t, 1)

    flags: dict[str, int] = {}
    for r in mine:
        for k, v in r.flags.get(name, {}).items():
            flags[k] = flags.get(k, 0) + v
    turns = sum(r.turns.get(name, 0) for r in mine)
    out = {
        "player": name, "games": len(mine),
        "wins": sum(r.score_for(name) == 1.0 for r in mine),
        "losses": sum(r.score_for(name) == 0.0 for r in mine),
        "draws": sum(r.score_for(name) == 0.5 for r in mine),
        "visits_per_move": per_move(name),
        "opponent_visits_per_move": {n: per_move(n) for n in opp_names},
        "flag_rates": {k: v / max(turns, 1) for k, v in flags.items()},
        "reasons": {k: sum(r.reason == k for r in mine) for k in ("score", "resign", "move-cap")},
    }
    out.update(paired_summary(pair_scores))
    return out
