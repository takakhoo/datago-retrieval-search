"""A stand-in for `katago analysis` that speaks the same JSON protocol.

Used by the test suite and CI, where no GPU or network weights exist. Each
legal move has a hidden true value (a one-ply area heuristic). A search
observes those values with noise that shrinks as 1/sqrt(visits), so deeper
searches pick better moves, which is the only property the pipeline tests need.
Nothing measured against this engine says anything about Go strength.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from .board import BLACK, PASS, Board, point_to_gtp

NOISE = 0.6


def _rng(*parts) -> np.random.Generator:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def true_values(board: Board, komi: float = 7.5) -> dict[int, float]:
    """Hidden win probability for the side to move after each legal move.

    The value is a squashed one-ply area margin plus a small symmetric jitter,
    so it is tied to the real outcome of the game and equal across the eight
    symmetric versions of a position.
    """
    key, sym = board.key()
    rng = _rng("truth", key)
    side = 1.0 if board.to_move == BLACK else -1.0
    legal = board.legal_moves()
    canon = sorted(board.to_canonical(p, sym) for p in legal if p != PASS)
    jitter = dict(zip(canon, 0.02 * rng.standard_normal(len(canon))))

    def squash(b: Board) -> float:
        black, white = b.area_score()
        return 1.0 / (1.0 + np.exp(-0.35 * side * (black - white - komi)))

    out = {}
    for p in legal:
        if p == PASS:
            out[p] = float(np.clip(squash(board) - 0.03, 0.001, 0.999))
            continue
        nxt = board.copy()
        nxt.play(p)
        out[p] = float(np.clip(squash(nxt) + jitter[board.to_canonical(p, sym)], 0.001, 0.999))
    return out


def analyze(query: dict) -> dict:
    size = query["boardXSize"]
    board = Board(size)
    for color, move in query["moves"]:
        board.play_gtp(move, BLACK if color == "B" else 2)
    visits = int(query["maxVisits"])
    truth = true_values(board, float(query.get("komi", 7.5)))
    points = list(truth)
    for spec in query.get("allowMoves", []):
        allowed = {m.upper() for m in spec["moves"]}
        points = [p for p in points if point_to_gtp(p, size).upper() in allowed]
    for spec in query.get("avoidMoves", []):
        banned = {m.upper() for m in spec["moves"]}
        points = [p for p in points if point_to_gtp(p, size).upper() not in banned]

    rng = _rng("search", board.key()[0], visits, query["id"])
    tv = np.array([truth[p] for p in points])
    prior_logits = 4.0 * tv + 0.8 * _rng("prior", board.key()[0]).standard_normal(len(points))
    prior = np.exp(prior_logits - prior_logits.max())
    prior /= prior.sum()
    k = min(len(points), max(1, int(np.sqrt(visits))))
    top = np.argsort(-prior)[:k]
    share = prior[top] / prior[top].sum()
    v = np.maximum(1, np.floor(share * visits)).astype(int)
    v[0] += max(0, visits - int(v.sum()))
    obs = np.clip(tv[top] + NOISE * rng.standard_normal(k) / np.sqrt(v), 0.001, 0.999)
    order = np.argsort(-(obs - 0.3 / np.sqrt(v)))

    black_to_move = board.to_move == BLACK

    def black_wr(x: float) -> float:
        return float(x if black_to_move else 1.0 - x)

    sign = 1.0 if black_to_move else -1.0
    infos = []
    for rank, j in enumerate(order):
        i = top[j]
        infos.append({
            "move": point_to_gtp(points[i], size),
            "visits": int(v[j]),
            "winrate": black_wr(obs[j]),
            "scoreLead": sign * float(40 * (obs[j] - 0.5)),
            "prior": float(prior[i]),
            "lcb": black_wr(obs[j] - 0.3 / np.sqrt(v[j])),
            "utility": sign * float(2 * obs[j] - 1),
            "order": rank,
            "pv": [point_to_gtp(points[i], size)],
        })
    root_wr = float(np.average(obs, weights=v))
    resp = {
        "id": query["id"],
        "turnNumber": len(query["moves"]),
        "moveInfos": infos,
        "rootInfo": {
            "winrate": black_wr(root_wr),
            "scoreLead": sign * 40 * (root_wr - 0.5),
            "visits": int(v.sum()),
            "rawWinrate": black_wr(float(np.clip(tv.mean(), 0.01, 0.99))),
            "rawLead": sign * 40 * (float(tv.mean()) - 0.5),
        },
    }
    if query.get("includePolicy"):
        full = np.full(size * size + 1, -1.0)
        for p, pr in zip(points, prior):
            full[size * size if p == PASS else p] = pr
        resp["policy"] = full.tolist()
    return resp


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        query = json.loads(line)
        try:
            out = analyze(query)
        except Exception as exc:
            out = {"id": query.get("id"), "error": repr(exc)}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
