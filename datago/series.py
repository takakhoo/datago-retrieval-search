"""Repeated play from the empty board, with a compute ledger.

A series is how an engine is actually used: many games against the same
opponent pool, each starting from move one. Positions recur, which is what a
search memory can exploit.

The ledger keeps the comparison honest. Each DataGo move is granted the
baseline's per-move budget. Whatever memory hits and early stops leave unspent
may be used after a game to deepen stored positions that keep recurring.
Total spending never exceeds the total grant, so over the series DataGo uses
at most the baseline's compute.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .engine import AnalysisEngine
from .match import GameConfig, GameRecord, play_game
from .memory import Memory
from .players import Player


@dataclass
class Ledger:
    budget_per_move: float
    granted: float = 0.0
    played: float = 0.0
    deepening: float = 0.0
    deepened: int = 0

    @property
    def balance(self) -> float:
        return self.granted - self.played - self.deepening


class Deepener:
    """After each game, spend any ledger surplus on the most-hit shallow entries."""

    def __init__(self, name: str, memory: Memory, engine: AnalysisEngine, ledger: Ledger,
                 deep_visits: int, komi: float, min_seen: int = 2, enabled: bool = True):
        self.name, self.memory, self.engine, self.ledger = name, memory, engine, ledger
        self.deep_visits, self.komi, self.min_seen, self.enabled = deep_visits, komi, min_seen, enabled
        self._lock = threading.Lock()
        self._busy: set[str] = set()

    def __call__(self, rec: GameRecord) -> None:
        jobs = []
        with self._lock:
            self.ledger.granted += self.ledger.budget_per_move * rec.turns[self.name]
            self.ledger.played += rec.visits[self.name]
            if self.enabled:
                for entry in self.memory.shallow_by_demand(self.deep_visits, self.min_seen):
                    if self.ledger.balance < self.deep_visits:
                        break
                    if entry.key in self._busy:
                        continue
                    self._busy.add(entry.key)
                    self.ledger.deepening += self.deep_visits
                    self.ledger.deepened += 1
                    jobs.append(entry)
        for entry in jobs:
            self.memory.deepen(self.engine, entry, self.deep_visits, self.komi)
            with self._lock:
                self._busy.discard(entry.key)


def run_series(a: Player, b: Player, cfg: GameConfig, games: int, seed: int = 0,
               workers: int = 32, out_path: str | Path | None = None,
               after_game: Callable[[GameRecord], None] | None = None,
               progress: Callable[[int, int, GameRecord], None] | None = None) -> list[GameRecord]:
    """Play `games` games from the empty board, alternating colors.

    Games 2k and 2k+1 share opening id k, so pair statistics are balanced by color.
    """
    seeds = np.random.SeedSequence(seed).spawn(games)
    lock = threading.Lock()
    records: list[GameRecord] = []
    fh = open(out_path, "a") if out_path else None

    def run(i: int) -> GameRecord:
        black, white = (a, b) if i % 2 == 0 else (b, a)
        rec = play_game(black, white, cfg, np.random.default_rng(seeds[i]), (), i // 2)
        if after_game:
            after_game(rec)
        return rec

    try:
        with ThreadPoolExecutor(workers) as pool:
            for fut in as_completed([pool.submit(run, i) for i in range(games)]):
                rec = fut.result()
                with lock:
                    records.append(rec)
                    if fh:
                        fh.write(json.dumps(asdict(rec)) + "\n")
                        fh.flush()
                    if progress:
                        progress(len(records), games, rec)
    finally:
        if fh:
            fh.close()
    return records
