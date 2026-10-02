"""End-to-end checks against the fake engine. These verify plumbing and
accounting. They say nothing about playing strength."""
import sys

import numpy as np
import pytest

from datago.board import BLACK, PASS, Board, symmetry_tables
from datago.engine import AnalysisEngine
from datago.features import FEATURE_NAMES, extract
from datago.match import GameConfig, play_game, run_match, sample_openings, summarize
from datago.memory import Memory
from datago.players import DataGoPlayer, KataGoPlayer, Temperature, pick_move
from datago.stats import bradley_terry, elo_from_score, paired_summary


@pytest.fixture(scope="module")
def engine():
    with AnalysisEngine([sys.executable, "-m", "datago.fake_engine"]) as e:
        yield e


def test_search_respects_budget_and_perspective(engine):
    board = Board(7)
    res = engine.search(board, 64, 7.5, include_policy=True)
    assert res.visits >= 60 and res.moves[0].order == 0
    assert len(res.policy) == 50
    board.play(res.best.point)
    reply = engine.search(board, 64, 7.5)
    assert 0 < reply.winrate < 1
    assert all(board.is_legal(m.point) for m in reply.moves)


def test_allow_moves_restricts_root(engine):
    board = Board(7)
    only = [10, 11]
    res = engine.search(board, 32, 7.5, allow=only)
    assert {m.point for m in res.moves} <= set(only)


def test_features_are_finite_and_named(engine):
    board = Board(7)
    f = extract(engine.search(board, 100, 7.5), board)
    assert f.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(f))


def test_memory_hit_is_free_and_symmetry_aware(engine, tmp_path):
    mem = Memory(tmp_path / "mem.jsonl")
    always = lambda feats: True
    player = DataGoPlayer(engine, 16, 128, always, mem, temp=None)
    rng = np.random.default_rng(0)
    board = Board(7)
    board.play(9)
    first = player.decide(board, 7.5, rng)
    assert first.visits >= 16 + 120 and first.flags.get("deep")

    to_sym, _ = symmetry_tables(7)
    for s in range(8):
        twin = Board(7)
        twin.play(int(to_sym[s][9]))
        again = player.decide(twin, 7.5, rng)
        assert again.visits == 0 and again.flags == {"hit": True}
        assert again.point == (PASS if first.point == PASS else int(to_sym[s][first.point]))

    reloaded = Memory(tmp_path / "mem.jsonl")
    assert len(reloaded) == 1 and reloaded.get(board, 7.5, 100) is not None
    assert reloaded.get(board, 7.5, 10_000) is None
    assert reloaded.get(board, 6.5) is None


def test_memory_keeps_the_deeper_search(engine):
    mem = Memory()
    board = Board(7)
    deep, shallow = engine.search(board, 200, 7.5), engine.search(board, 20, 7.5)
    assert mem.put(board, 7.5, deep) and not mem.put(board, 7.5, shallow)
    assert mem.get(board, 7.5).visits == deep.visits


def test_game_terminates_and_accounts_visits(engine):
    cfg = GameConfig(size=5, komi=0.5, max_moves=60, keep_trace=True)
    a, b = KataGoPlayer(engine, 20, name="a"), KataGoPlayer(engine, 40, name="b")
    rec = play_game(a, b, cfg, np.random.default_rng(1))
    assert rec.winner in ("B", "W", "draw") and rec.reason in ("score", "resign", "move-cap")
    assert rec.visits["a"] == sum(t["v"] for t in rec.trace if t["p"] == "a")
    assert rec.turns["a"] + rec.turns["b"] == len(rec.trace)
    assert rec.sgf().startswith("(;GM[1]") and rec.sgf().count(";B[") >= 1
    replay = Board(5)
    for mv in rec.moves:
        replay.play_gtp(mv)


def test_match_is_color_balanced_and_accounts_compute(engine, tmp_path):
    cfg = GameConfig(size=5, komi=0.5, max_moves=50)
    rng = np.random.default_rng(3)
    openings = sample_openings(engine, 12, cfg, plies=2, rng=rng, visits=30,
                               balance=0.5, check_visits=30, workers=4)
    assert len({tuple(o) for o in openings}) == 12
    weak, strong = KataGoPlayer(engine, 2, name="weak"), KataGoPlayer(engine, 2000, name="strong")
    out = tmp_path / "games.jsonl"
    recs = run_match(strong, weak, cfg, openings, seed=5, workers=4, out_path=out)
    assert len(recs) == 24 and len(out.read_text().splitlines()) == 24
    assert sum(r.black == "strong" for r in recs) == 12
    s = summarize(recs, "strong")
    assert s["pairs"] == 12 and s["wins"] + s["losses"] + s["draws"] == 24
    assert s["visits_per_move"] > 20 * s["opponent_visits_per_move"]["weak"]
    assert 0.0 <= s["score_lo"] <= s["score"] <= s["score_hi"] <= 1.0


def test_fake_engine_picks_better_moves_with_more_visits(engine):
    from datago.fake_engine import true_values
    rng = np.random.default_rng(0)
    gain = []
    for _ in range(40):
        board = Board(7)
        for _ in range(int(rng.integers(0, 12))):
            board.play(int(rng.choice([p for p in board.legal_moves() if p != PASS])))
        truth = true_values(board, 7.5)
        shallow = engine.search(board, 4, 7.5).best.point
        deep = engine.search(board, 4000, 7.5).best.point
        gain.append(truth[deep] - truth[shallow])
    assert np.mean(gain) > 0.01


def test_pick_move_is_greedy_without_temperature_and_samples_with_it(engine):
    board = Board(7)
    res = engine.search(board, 300, 7.5)
    rng = np.random.default_rng(0)
    assert pick_move(res, 0, 7, rng, None) == res.best.point
    hot = Temperature(early=2.0, late=2.0, min_share=0.0)
    assert len({pick_move(res, 0, 7, rng, hot) for _ in range(200)}) > 1
    cold = Temperature(early=0.01, late=0.01)
    top = max(res.moves, key=lambda m: m.visits).point
    assert {pick_move(res, 0, 7, rng, cold) for _ in range(50)} == {top}


def test_elo_helpers():
    assert elo_from_score(0.5) == pytest.approx(0.0)
    assert elo_from_score(0.75) == pytest.approx(190.85, abs=0.1)
    s = paired_summary([1, 1, 0.5, 1, 0.5, 1, 1, 0.5])
    assert s["elo_lo"] > 0 and s["score_lo"] <= s["score"] <= s["score_hi"]
    games = [("a", "b", 1.0)] * 30 + [("a", "b", 0.0)] * 10 + [("b", "c", 1.0)] * 30 + [("b", "c", 0.0)] * 10
    elo = bradley_terry(games, anchor="b")
    assert elo["a"] > 150 and elo["c"] < -150 and elo["b"] == 0
