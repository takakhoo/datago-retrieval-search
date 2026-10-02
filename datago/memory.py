"""Search memory: exact reuse by canonical key, plus an embedding index.

An entry is a completed search stored in canonical orientation, so the eight
symmetric versions of a position share one entry and stored moves are mapped
back through the query's own symmetry on retrieval.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import BLACK, WHITE, Board
from .engine import RULES_TAG, MoveStat, SearchResult


@dataclass
class Entry:
    key: str
    size: int
    visits: int
    winrate: float
    score_lead: float
    # rows of (canonical point, visits, winrate, score_lead, prior, lcb)
    moves: list[list[float]]
    seen: int = 1
    embed: list[float] | None = None
    # One move sequence that reaches this position, so it can be searched again later.
    line: list[list[str]] | None = None

    def to_result(self, board: Board, sym: int) -> SearchResult:
        stats = [
            MoveStat(
                point=board.from_canonical(int(p), sym), visits=int(v), winrate=w,
                score_lead=s, prior=pr, lcb=lcb, utility=2 * w - 1, order=i,
            )
            for i, (p, v, w, s, pr, lcb) in enumerate(self.moves)
        ]
        return SearchResult(stats, self.winrate, self.score_lead, self.visits)


class Memory:
    def __init__(self, path: str | Path | None = None, max_moves: int = 12):
        self.path = Path(path) if path else None
        self.max_moves = max_moves
        self.entries: dict[str, Entry] = {}
        self._lock = threading.Lock()
        self._matrix: np.ndarray | None = None
        self._matrix_keys: list[str] = []
        self.lookups = self.hits = 0
        if self.path and self.path.exists():
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        e = Entry(**json.loads(line))
                        self.entries[e.key] = e

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, board: Board, komi: float, min_visits: int = 0) -> SearchResult | None:
        key, sym = board.key(komi, RULES_TAG)
        with self._lock:
            self.lookups += 1
            entry = self.entries.get(key)
            if entry is None or entry.visits < min_visits:
                return None
            self.hits += 1
            entry.seen += 1
        return entry.to_result(board, sym)

    def put(self, board: Board, komi: float, result: SearchResult,
            embed: np.ndarray | None = None) -> bool:
        """Store a search, keeping the deeper one on collision. Returns True if stored."""
        key, sym = board.key(komi, RULES_TAG)
        rows = [
            [board.to_canonical(m.point, sym), m.visits, m.winrate, m.score_lead, m.prior, m.lcb]
            for m in result.moves[: self.max_moves]
        ]
        entry = Entry(key, board.size, result.visits, result.winrate, result.score_lead, rows,
                      embed=None if embed is None else [float(x) for x in embed],
                      line=board.gtp_moves())
        with self._lock:
            old = self.entries.get(key)
            if old is not None and old.visits >= result.visits:
                return False
            if old is not None:
                entry.seen = old.seen
            self.entries[key] = entry
            self._matrix = None
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(entry.__dict__) + "\n")
        return True

    def shallow_by_demand(self, below_visits: int, min_seen: int = 2) -> list[Entry]:
        """Entries hit at least min_seen times and searched less than below_visits, most-hit first."""
        with self._lock:
            pool = [e for e in self.entries.values()
                    if e.visits < below_visits and e.seen >= min_seen and e.line is not None]
        return sorted(pool, key=lambda e: -e.seen)

    def deepen(self, engine, entry: Entry, visits: int, komi: float) -> int:
        """Replace an entry with a deeper search of the same position. Returns visits spent."""
        board = Board(entry.size)
        for color, move in entry.line:
            board.play_gtp(move, BLACK if color == "B" else WHITE)
        result = engine.search(board, visits, komi)
        self.put(board, komi, result)
        return result.visits

    def compact(self) -> None:
        """Rewrite the log with one line per key."""
        if not self.path:
            return
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w") as f:
                for e in self.entries.values():
                    f.write(json.dumps(e.__dict__) + "\n")
            tmp.replace(self.path)

    def nearest(self, embed: np.ndarray, k: int = 8) -> list[tuple[Entry, float]]:
        """Exact cosine top-k over entries that carry an embedding."""
        with self._lock:
            if self._matrix is None:
                keyed = [(k_, e) for k_, e in self.entries.items() if e.embed is not None]
                self._matrix_keys = [k_ for k_, _ in keyed]
                if keyed:
                    m = np.asarray([e.embed for _, e in keyed], dtype=np.float32)
                    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
                    self._matrix = m
                else:
                    self._matrix = np.zeros((0, len(embed)), dtype=np.float32)
            if not len(self._matrix_keys):
                return []
            q = np.asarray(embed, dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-12)
            scores = self._matrix @ q
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self.entries[self._matrix_keys[i]], float(scores[i])) for i in top]
