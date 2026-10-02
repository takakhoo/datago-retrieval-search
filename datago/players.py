"""Players: the uniform-budget KataGo baseline and DataGo.

Both go through the same engine, the same move-selection rule, and the same
visit accounting. The only difference is how DataGo spends and saves visits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .board import Board
from .engine import AnalysisEngine, SearchResult
from .features import extract
from .memory import Memory


@dataclass
class Temperature:
    """KataGo-style move sampling: sharp late, a little noise early."""
    early: float = 0.5
    late: float = 0.1
    halflife: float = 19.0
    min_share: float = 0.02

    def at(self, move_number: int, size: int) -> float:
        decay = 0.5 ** (move_number / (self.halflife * size / 19.0))
        return self.late + (self.early - self.late) * decay


def pick_move(result: SearchResult, move_number: int, size: int,
              rng: np.random.Generator, temp: Temperature | None) -> int:
    if temp is None:
        return result.best.point
    t = temp.at(move_number, size)
    visits = np.array([m.visits for m in result.moves], dtype=np.float64)
    keep = visits >= temp.min_share * visits.sum()
    keep[0] = True
    logw = np.where(keep, np.log(np.maximum(visits, 1e-9)) / t, -np.inf)
    w = np.exp(logw - logw.max())
    return result.moves[int(rng.choice(len(w), p=w / w.sum()))].point


@dataclass
class Decision:
    point: int
    visits: int
    winrate: float
    flags: dict = field(default_factory=dict)


class Player(Protocol):
    name: str

    def decide(self, board: Board, komi: float, rng: np.random.Generator) -> Decision: ...


class KataGoPlayer:
    def __init__(self, engine: AnalysisEngine, visits: int,
                 temp: Temperature | None = Temperature(), name: str | None = None):
        self.engine, self.visits, self.temp = engine, visits, temp
        self.name = name or f"katago-v{visits}"

    def decide(self, board: Board, komi: float, rng: np.random.Generator) -> Decision:
        result = self.engine.search(board, self.visits, komi)
        point = pick_move(result, len(board.moves), board.size, rng, self.temp)
        return Decision(point, result.visits, result.winrate)


Gate = Callable[[np.ndarray], bool]


class DataGoPlayer:
    """Shallow search, then a gate decides whether to pay for a deep one.

    Deep searches are written to memory. A later exact hit (same position up
    to symmetry) with at least `reuse_visits` stored visits replaces the whole
    search and costs zero visits.
    """

    def __init__(self, engine: AnalysisEngine, base_visits: int, deep_visits: int,
                 gate: Gate | None, memory: Memory | None = None,
                 reuse_visits: int | None = None, store: bool = True,
                 temp: Temperature | None = Temperature(), name: str = "datago"):
        self.engine, self.base_visits, self.deep_visits = engine, base_visits, deep_visits
        self.gate, self.memory, self.store, self.temp, self.name = gate, memory, store, temp, name
        self.reuse_visits = deep_visits if reuse_visits is None else reuse_visits

    def decide(self, board: Board, komi: float, rng: np.random.Generator) -> Decision:
        move_number = len(board.moves)
        if self.memory is not None:
            cached = self.memory.get(board, komi, self.reuse_visits)
            if cached is not None:
                point = pick_move(cached, move_number, board.size, rng, self.temp)
                return Decision(point, 0, cached.winrate, {"hit": True})

        result = self.engine.search(board, self.base_visits, komi)
        spent = result.visits
        flags = {}
        if self.gate is not None and self.gate(extract(result, board)):
            deep = self.engine.search(board, self.deep_visits, komi)
            spent += deep.visits
            flags["deep"] = True
            flags["changed"] = deep.best.point != result.best.point
            result = deep
            if self.memory is not None and self.store:
                self.memory.put(board, komi, deep)
        point = pick_move(result, move_number, board.size, rng, self.temp)
        return Decision(point, spent, result.winrate, flags)
