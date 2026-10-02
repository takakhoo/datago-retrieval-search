"""Match statistics: Elo with confidence intervals, and a multi-player fit."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def elo_from_score(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def paired_summary(pair_scores: Sequence[float], seed: int = 0, boots: int = 10000) -> dict:
    """Score and Elo for one side, resampling opening pairs.

    pair_scores holds, per opening, that side's mean score over the two
    color-swapped games (each game scores 1, 0.5, or 0). Resampling pairs
    rather than games keeps the opening's bias out of the interval.
    """
    x = np.asarray(pair_scores, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"pairs": 0}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(boots, n))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "pairs": n, "score": p, "score_lo": float(lo), "score_hi": float(hi),
        "elo": elo_from_score(p), "elo_lo": elo_from_score(float(lo)),
        "elo_hi": elo_from_score(float(hi)),
        "z": (p - 0.5) / se if se and se > 0 else float("nan"),
        "p_superior": float((means > 0.5).mean()),
    }


def bradley_terry(results: Iterable[tuple[str, str, float]], anchor: str | None = None,
                  iters: int = 2000, prior_games: float = 0.5) -> dict[str, float]:
    """Maximum-likelihood Elo from (player_a, player_b, score_of_a) records.

    A weak symmetric prior (prior_games of a draw per pairing) keeps ratings
    finite for players with perfect records.
    """
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    players: set[str] = set()
    for a, b, s in results:
        players |= {a, b}
        wins[(a, b)] += s
        wins[(b, a)] += 1 - s
        games[(a, b)] += 1
        games[(b, a)] += 1
    for (a, b) in list(games):
        wins[(a, b)] += prior_games / 2
        games[(a, b)] += prior_games / 2
    names = sorted(players)
    gamma = {p: 1.0 for p in names}
    for _ in range(iters):
        new = {}
        for p in names:
            w = sum(wins[(p, q)] for q in names if (p, q) in games)
            d = sum(games[(p, q)] / (gamma[p] + gamma[q]) for q in names if (p, q) in games)
            new[p] = w / d if d > 0 else gamma[p]
        norm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / norm for p, v in new.items()}
        delta = max(abs(math.log(new[p] / gamma[p])) for p in names)
        gamma = new
        if delta < 1e-10:
            break
    elo = {p: 400.0 * math.log10(g) for p, g in gamma.items()}
    shift = elo[anchor] if anchor else 0.0
    return {p: e - shift for p, e in elo.items()}


def bradley_terry_ci(results: Sequence[tuple[str, str, float]], anchor: str,
                     boots: int = 500, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Elo with a 95% bootstrap interval, resampling games."""
    point = bradley_terry(results, anchor)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    n = len(results)
    for _ in range(boots):
        idx = rng.integers(0, n, n)
        fit = bradley_terry([results[i] for i in idx], anchor, iters=300)
        for p, e in fit.items():
            samples[p].append(e)
    out = {}
    for p, e in point.items():
        lo, hi = np.quantile(samples[p], [0.025, 0.975]) if samples[p] else (e, e)
        out[p] = (e, float(lo), float(hi))
    return out
