"""Offline evaluation of visit-allocation policies on the ladder dataset.

Each position was searched at every rung of a geometric visit ladder, and
every move chosen at any rung was scored by an independent forced search.
A policy is a rule for which rung to stop at. Its quality is the mean forced
winrate of the moves it plays, reported as regret against the best scored
move, and its cost is the visits it used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import Board, gtp_to_point
from .engine import MoveStat, SearchResult
from .features import FEATURE_NAMES, extract


@dataclass
class Ladder:
    rungs: np.ndarray        # (K,) visit budgets
    ids: list[str]
    game: np.ndarray         # (N,) game index, used for train/test splits
    move_number: np.ndarray  # (N,)
    feats: np.ndarray        # (N, K, F) features of the search at each rung
    move: np.ndarray         # (N, K) point chosen at each rung
    q: np.ndarray            # (N, K) forced winrate of that move
    score: np.ndarray        # (N, K) forced score lead of that move
    regret: np.ndarray       # (N, K) best forced winrate minus q
    policy: np.ndarray | None = None
    ownership: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, mask: np.ndarray) -> "Ladder":
        idx = np.flatnonzero(mask)
        return Ladder(
            self.rungs, [self.ids[i] for i in idx], self.game[idx], self.move_number[idx],
            self.feats[idx], self.move[idx], self.q[idx], self.score[idx], self.regret[idx],
            None if self.policy is None else self.policy[idx],
            None if self.ownership is None else self.ownership[idx],
        )


def result_from_summary(summary: dict, size: int) -> SearchResult:
    moves = [
        MoveStat(point=gtp_to_point(mv, size), visits=v, winrate=w, score_lead=s, prior=pr,
                 lcb=lcb, utility=2 * w - 1, order=i, score_stdev=sd)
        for i, (mv, v, w, lcb, pr, s, sd) in enumerate(summary["moves"])
    ]
    return SearchResult(moves, summary["winrate"], summary["score"], summary["visits"],
                        summary.get("raw_winrate"), summary.get("raw_score"),
                        raw=summary.get("raw") or {})


def load(path: str | Path, keep_maps: bool = False) -> Ladder:
    ids, game, move_number, feats, move, q, score = [], [], [], [], [], [], []
    policy, ownership = [], []
    rungs = None
    for line in open(path):
        rec = json.loads(line)
        size = rec["size"]
        keys = sorted(rec["ladder"], key=int)
        if rungs is None:
            rungs = np.array([int(k) for k in keys])
        board = Board(size, superko=False)
        for mv in rec["moves"]:
            board.play_gtp(mv)
        f_row, m_row, q_row, s_row = [], [], [], []
        for k in keys:
            res = result_from_summary(rec["ladder"][k], size)
            f_row.append(extract(res, board))
            best = rec["ladder"][k]["moves"][0][0]
            m_row.append(gtp_to_point(best, size))
            q_row.append(rec["forced"][best]["winrate"])
            s_row.append(rec["forced"][best]["score"])
        ids.append(rec["id"])
        game.append(rec["game"])
        move_number.append(rec["move_number"])
        feats.append(f_row)
        move.append(m_row)
        q.append(q_row)
        score.append(s_row)
        if keep_maps:
            policy.append(rec["policy"])
            ownership.append(rec["ownership"])
    q = np.array(q)
    best = np.array([
        max(f["winrate"] for f in json.loads(line)["forced"].values())
        for line in open(path)
    ])
    return Ladder(
        rungs, ids, np.array(game), np.array(move_number), np.array(feats), np.array(move),
        q, np.array(score), best[:, None] - q,
        np.array(policy, dtype=np.float32) if keep_maps else None,
        np.array(ownership, dtype=np.float32) if keep_maps else None,
    )


def uniform_curve(lad: Ladder) -> np.ndarray:
    """Mean regret of always stopping at each rung."""
    return lad.regret.mean(axis=0)


def equivalent_visits(lad: Ladder, regret: float) -> float:
    """Uniform budget whose mean regret equals `regret`, interpolating in log visits."""
    curve = uniform_curve(lad)
    logv = np.log(lad.rungs)
    order = np.argsort(curve)
    return float(np.exp(np.interp(regret, curve[order], logv[order])))


def evaluate_stops(lad: Ladder, stop: np.ndarray, path: list[int] | None = None) -> dict:
    """Score a policy given the rung index each position stops at.

    `path` lists the rung indices a sequential policy searches in order. The
    restart cost charges every rung searched up to the stopping one. The
    continuation cost charges only the final rung, which is what an engine
    that extends its tree in place would pay.
    """
    n = len(lad)
    rows = np.arange(n)
    regret = lad.regret[rows, stop]
    cont = lad.rungs[stop].astype(float)
    if path is None:
        restart = cont
    else:
        path_arr = np.array(path)
        cum = np.cumsum(lad.rungs[path_arr])
        pos = {r: i for i, r in enumerate(path)}
        restart = np.array([cum[pos[int(s)]] for s in stop], dtype=float)
    mean_regret = float(regret.mean())
    out = {
        "regret": mean_regret,
        "blunder_rate": float((regret > 0.02).mean()),
        "cost_restart": float(restart.mean()),
        "cost_continue": float(cont.mean()),
        "equivalent_visits": equivalent_visits(lad, mean_regret),
    }
    out["gain_restart"] = out["equivalent_visits"] / out["cost_restart"]
    out["gain_continue"] = out["equivalent_visits"] / out["cost_continue"]
    return out


def sequential_stops(lad: Ladder, path: list[int], scores: np.ndarray, threshold: float) -> np.ndarray:
    """Stop at the first rung on `path` whose continue-score is below threshold.

    scores[i, k] is the predicted value of searching past rung k for position i.
    """
    n = len(lad)
    stop = np.full(n, path[-1])
    active = np.ones(n, dtype=bool)
    for k in path[:-1]:
        halt = active & (scores[:, k] < threshold)
        stop[halt] = k
        active &= ~halt
    return stop


__all__ = ["Ladder", "load", "uniform_curve", "equivalent_visits", "evaluate_stops",
           "sequential_stops", "FEATURE_NAMES"]
