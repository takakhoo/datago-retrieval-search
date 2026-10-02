"""Features of a finished shallow search that predict whether more search pays.

Everything here is computed from the search the player already ran, so the
gate adds no engine cost.
"""
from __future__ import annotations

import math

import numpy as np

from .board import Board
from .engine import SearchResult

FEATURE_NAMES = [
    "top1_share", "top2_share", "share_gap", "visit_entropy", "n_contenders",
    "best_prior", "max_prior", "best_is_not_max_prior", "policy_surprise",
    "q_gap", "lcb_margin", "visit_value_disagree", "root_winrate", "undecided",
    "value_surprise", "abs_score_lead", "score_spread", "phase", "log_visits",
]


def extract(result: SearchResult, board: Board) -> np.ndarray:
    moves = result.moves
    visits = np.array([m.visits for m in moves], dtype=np.float64)
    total = max(visits.sum(), 1.0)
    share = visits / total
    by_visits = np.argsort(-visits)
    best = moves[0]
    top1 = share[by_visits[0]]
    top2 = share[by_visits[1]] if len(moves) > 1 else 0.0

    nz = share[share > 0]
    entropy = float(-(nz * np.log(nz)).sum() / math.log(len(nz))) if len(nz) > 1 else 0.0

    priors = np.array([m.prior for m in moves], dtype=np.float64)
    max_prior = float(priors.max()) if len(priors) else 0.0
    pn = np.clip(priors / max(priors.sum(), 1e-9), 1e-6, None)
    surprise = float((nz * np.log(nz / pn[share > 0])).sum())

    searched = [m for m in moves if m.visits >= 2]
    others = [m for m in searched if m is not best]
    q_gap = best.winrate - max((m.winrate for m in others), default=best.winrate - 1.0)
    lcb_margin = best.lcb - max((m.winrate for m in others), default=best.lcb - 1.0)
    most_visited = moves[by_visits[0]]
    highest_q = max(searched, key=lambda m: m.winrate, default=best)
    disagree = float(most_visited.point != highest_q.point)

    raw = result.raw_winrate if result.raw_winrate is not None else result.winrate
    scores = [m.score_lead for m in searched[:5]]
    return np.array([
        top1, top2, top1 - top2, entropy, float((share >= 0.05).sum()),
        best.prior, max_prior, float(best.prior < max_prior - 1e-9), surprise,
        float(np.clip(q_gap, -1, 1)), float(np.clip(lcb_margin, -1, 1)), disagree,
        result.winrate, 1.0 - 2.0 * abs(result.winrate - 0.5),
        abs(result.winrate - raw), abs(result.score_lead),
        float(np.std(scores)) if len(scores) > 1 else 0.0,
        board.num_stones() / (board.size * board.size), math.log(max(result.visits, 1)),
    ])
