"""Client for KataGo's JSON analysis engine.

One engine process serves many concurrent games. Each query is an independent
search with an exact visit budget and no tree reuse, which makes compute
accounting unambiguous: a player is charged for every visit it requests.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Sequence

from .board import BLACK, Board, color_name, gtp_to_point, point_to_gtp

RULES = {
    "ko": "POSITIONAL",
    "scoring": "AREA",
    "tax": "NONE",
    "suicide": False,
    "hasButton": False,
    "whiteHandicapBonus": "0",
    "friendlyPassOk": False,
}
RULES_TAG = "area-psk"


@dataclass
class MoveStat:
    point: int
    visits: int
    winrate: float
    score_lead: float
    prior: float
    lcb: float
    utility: float
    order: int
    pv: list[int] = field(default_factory=list)


@dataclass
class SearchResult:
    """One completed search. winrate and score_lead are for the side to move."""
    moves: list[MoveStat]
    winrate: float
    score_lead: float
    visits: int
    raw_winrate: float | None = None
    raw_score_lead: float | None = None
    policy: list[float] | None = None
    ownership: list[float] | None = None

    @property
    def best(self) -> MoveStat:
        return self.moves[0]

    def stat(self, point: int) -> MoveStat | None:
        for m in self.moves:
            if m.point == point:
                return m
        return None


def parse_response(resp: dict[str, Any], size: int, to_move: int) -> SearchResult:
    sign = 1.0 if to_move == BLACK else -1.0

    def wr(x: float) -> float:
        return x if to_move == BLACK else 1.0 - x

    moves = []
    for info in resp["moveInfos"]:
        moves.append(MoveStat(
            point=gtp_to_point(info["move"], size),
            visits=int(info["visits"]),
            winrate=wr(info["winrate"]),
            score_lead=sign * info["scoreLead"],
            prior=float(info.get("prior", 0.0)),
            lcb=wr(info["lcb"]) if "lcb" in info else wr(info["winrate"]),
            utility=sign * info.get("utility", 0.0),
            order=int(info.get("order", len(moves))),
            pv=[gtp_to_point(m, size) for m in info.get("pv", [])],
        ))
    moves.sort(key=lambda m: m.order)
    root = resp["rootInfo"]
    return SearchResult(
        moves=moves,
        winrate=wr(root["winrate"]),
        score_lead=sign * root["scoreLead"],
        visits=int(root["visits"]),
        raw_winrate=wr(root["rawWinrate"]) if "rawWinrate" in root else None,
        raw_score_lead=sign * root["rawLead"] if "rawLead" in root else None,
        policy=resp.get("policy"),
        ownership=resp.get("ownership"),
    )


class AnalysisEngine:
    """Thread-safe multiplexer over one analysis-engine subprocess.

    All winrates are requested from Black's perspective and converted to the
    side to move in parse_response, so the result does not depend on the
    engine's reportAnalysisWinratesAs default.
    """

    def __init__(self, command: Sequence[str], stderr=subprocess.DEVNULL):
        self.proc = subprocess.Popen(
            list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, bufsize=1,
        )
        self._ids = itertools.count()
        self._pending: dict[str, tuple[Future, int, int]] = {}
        self._lock = threading.Lock()
        self.visits_served = 0
        self.queries_served = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            resp = json.loads(line)
            if resp.get("isDuringSearch"):
                continue
            with self._lock:
                entry = self._pending.pop(resp.get("id"), None)
            if entry is None:
                continue
            fut, size, to_move = entry
            if "error" in resp:
                fut.set_exception(RuntimeError(f"engine error: {resp['error']}"))
                continue
            if "moveInfos" not in resp:
                fut.set_exception(RuntimeError(f"unexpected response: {line[:200]}"))
                continue
            result = parse_response(resp, size, to_move)
            self.visits_served += result.visits
            self.queries_served += 1
            fut.set_result(result)
        with self._lock:
            pending, self._pending = self._pending, {}
        for fut, _, _ in pending.values():
            fut.set_exception(RuntimeError("engine process exited"))

    def submit(
        self,
        board: Board,
        visits: int,
        komi: float = 7.5,
        *,
        allow: Sequence[int] | None = None,
        avoid: Sequence[int] | None = None,
        include_policy: bool = False,
        include_ownership: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> Future:
        qid = f"q{next(self._ids)}"
        size = board.size
        query: dict[str, Any] = {
            "id": qid,
            "moves": board.gtp_moves(),
            "rules": RULES,
            "komi": komi,
            "boardXSize": size,
            "boardYSize": size,
            "maxVisits": int(visits),
            "analyzeTurns": [len(board.moves)],
            "overrideSettings": {"reportAnalysisWinratesAs": "BLACK", **(overrides or {})},
        }
        if include_policy:
            query["includePolicy"] = True
        if include_ownership:
            query["includeOwnership"] = True
        player = color_name(board.to_move)
        if allow is not None:
            query["allowMoves"] = [{
                "player": player,
                "moves": [point_to_gtp(p, size) for p in allow],
                "untilDepth": 1,
            }]
        if avoid:
            query["avoidMoves"] = [{
                "player": player,
                "moves": [point_to_gtp(p, size) for p in avoid],
                "untilDepth": 1,
            }]
        fut: Future = Future()
        with self._lock:
            self._pending[qid] = (fut, size, board.to_move)
            self.proc.stdin.write(json.dumps(query) + "\n")
            self.proc.stdin.flush()
        return fut

    def search(self, board: Board, visits: int, komi: float = 7.5, **kw) -> SearchResult:
        return self.submit(board, visits, komi, **kw).result()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def katago_command(binary: str, model: str, config: str, extra: Sequence[str] = ()) -> list[str]:
    return [binary, "analysis", "-model", model, "-config", config, *extra]
