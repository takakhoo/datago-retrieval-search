"""Client for KataGo's JSON analysis engine.

One engine process serves many concurrent games. Each query is an independent
search with an exact visit budget and no tree reuse, which makes compute
accounting unambiguous: a player is charged for every visit it requests.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
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
RAW_KEYS = ("rawStWrError", "rawStScoreError", "rawVarTimeLeft", "rawScoreSelfplayStdev")


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
    score_stdev: float = 0.0


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
    # KataGo's own raw-net error predictions: rawStWrError, rawStScoreError, rawVarTimeLeft.
    raw: dict[str, float] = field(default_factory=dict)

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
        # Root symmetry pruning reports mirrored copies of one searched move.
        if "isSymmetryOf" in info:
            continue
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
            score_stdev=float(info.get("scoreStdev", 0.0)),
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
        ownership=[sign * x for x in resp["ownership"]] if "ownership" in resp else None,
        raw={k: float(root[k]) for k in RAW_KEYS if k in root},
    )


class AnalysisEngine:
    """Thread-safe multiplexer over one analysis-engine subprocess.

    The engine config must set reportAnalysisWinratesAs = BLACK (as
    configs/analysis.cfg does). parse_response converts to the side to move.
    """

    def __init__(self, command: Sequence[str], stderr=subprocess.DEVNULL, env=None):
        self.proc = subprocess.Popen(
            list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, bufsize=1, env=env,
        )
        self._ids = itertools.count()
        self._pending: dict[str, tuple[Future, int, int]] = {}
        # Separate locks: the reader must never wait on a writer that is blocked
        # on a full stdin pipe, or the engine's stdout fills and both sides stall.
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.visits_served = 0
        self.queries_served = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("isDuringSearch") or "warning" in resp:
                continue
            with self._lock:
                entry = self._pending.pop(resp.get("id"), None)
            if entry is None:
                continue
            fut, size, to_move = entry
            try:
                if "error" in resp:
                    raise RuntimeError(f"engine error: {resp['error']}")
                result = parse_response(resp, size, to_move)
            except Exception as exc:
                fut.set_exception(RuntimeError(f"{exc!r} for response {line[:300]}"))
                continue
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
        }
        if overrides:
            query["overrideSettings"] = overrides
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
        payload = json.dumps(query) + "\n"
        with self._lock:
            self._pending[qid] = (fut, size, board.to_move)
        with self._write_lock:
            self.proc.stdin.write(payload)
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


NETS = {
    "b6": "kata1-b6c96-s175395328-d26788732.txt.gz",
    "b10": "kata1-b10c128-s1141046784-d204142634.txt.gz",
    "b18": "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
    "b28": "kata1-b28c512nbt-s13255194368-d5935380940.bin.gz",
}


def open_katago(net: str = "b18", gpu: int | None = None, config: str | None = None,
                threads: int | None = None, stderr=subprocess.DEVNULL) -> AnalysisEngine:
    """Start `katago analysis`. Paths come from $KATAGO and $DG (see README)."""
    root = Path(os.environ.get("DG", "."))
    binary = os.environ.get("KATAGO", "katago")
    model = NETS.get(net, net)
    if not os.path.isabs(model):
        model = str(root / "nets" / model)
    config = config or str(Path(__file__).resolve().parent.parent / "configs" / "analysis.cfg")
    cmd = [binary, "analysis", "-model", model, "-config", config]
    if threads:
        cmd += ["-override-config", f"numAnalysisThreads={threads}"]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return AnalysisEngine(cmd, stderr=stderr or subprocess.DEVNULL, env=env)
