"""Players: the uniform-budget KataGo baseline and DataGo.

Both go through the same kind of engine, the same move-selection rule, and the
same visit accounting. The only difference is how DataGo spends and saves
visits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .board import Board
from .engine import AnalysisEngine, SearchResult
from .features import extract
from .memory import Memory
from .stopper import Stopper, trajectory


@dataclass
class Temperature:
    """KataGo-style move sampling: sharp late, a little noise early.

    Defaults are KataGo's GTP defaults (chosenMoveTemperatureEarly 0.5,
    chosenMoveTemperatureHalflife 19, chosenMoveTemperature 0.1).
    """
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
    """visits is the continuation cost (the deepest search run). Numeric
    entries in `counts` are summed per game by the match runner."""
    point: int
    visits: int
    winrate: float
    counts: dict[str, float] = field(default_factory=dict)


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
        return Decision(point, result.visits, result.winrate, {"restart_visits": result.visits})


class DataGoPlayer:
    """Visit ladder with a learned stopping rule, backed by a search memory.

    1. If the position (up to symmetry) is in memory, play from the stored
       search. Cost: zero visits.
    2. Otherwise search at path[0] visits and ask the stopper whether the
       decision is settled. If not, search at the next rung, and so on.
    3. Store the final search for positions early enough to recur.

    With stopper=None the player always stops at path[0], which makes
    DataGoPlayer([v], None, None) identical to KataGoPlayer(v).
    """

    def __init__(self, engine: AnalysisEngine, path: list[int], stopper: Stopper | None = None,
                 memory: Memory | None = None, store_max_move: int = 80,
                 temp: Temperature | None = Temperature(), name: str = "datago"):
        self.engine, self.path, self.stopper = engine, list(path), stopper
        self.memory, self.store_max_move, self.temp, self.name = memory, store_max_move, temp, name

    def search(self, board: Board, komi: float) -> tuple[SearchResult, int, int]:
        """Run the ladder. Returns (final search, restart cost, index of final rung)."""
        prev_x, prev_move, n_changes, restart = None, None, 0, 0
        for j, v in enumerate(self.path):
            res = self.engine.search(board, v, komi)
            restart += res.visits
            if j == len(self.path) - 1 or self.stopper is None:
                break
            x = extract(res, board)
            t, n_changes = trajectory(prev_x, x, prev_move, res.best.point, n_changes)
            if self.stopper.should_stop(np.concatenate([x, t]), j):
                break
            prev_x, prev_move = x, res.best.point
        return res, restart, j

    def decide(self, board: Board, komi: float, rng: np.random.Generator) -> Decision:
        move_number = len(board.moves)
        if self.memory is not None:
            cached = self.memory.get(board, komi)
            if cached is not None:
                point = pick_move(cached, move_number, board.size, rng, self.temp)
                return Decision(point, 0, cached.winrate, {"hit": 1, "restart_visits": 0})
        res, restart, j = self.search(board, komi)
        if self.memory is not None and move_number <= self.store_max_move:
            self.memory.put(board, komi, res)
        point = pick_move(res, move_number, board.size, rng, self.temp)
        return Decision(point, res.visits, res.winrate,
                        {"restart_visits": restart, "rung": j, "searched": 1})
