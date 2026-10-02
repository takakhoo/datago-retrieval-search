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
    player = DataGoPlayer(engine, [128], None, mem, temp=None)
    rng = np.random.default_rng(0)
    board = Board(7)
    board.play(9)
    first = player.decide(board, 7.5, rng)
    assert first.visits == 128 and first.counts["searched"] == 1

    to_sym, _ = symmetry_tables(7)
    for s in range(8):
        twin = Board(7)
        twin.play(int(to_sym[s][9]))
        again = player.decide(twin, 7.5, rng)
        assert again.visits == 0 and again.counts == {"hit": 1, "restart_visits": 0}
        assert again.point == (PASS if first.point == PASS else int(to_sym[s][first.point]))

    reloaded = Memory(tmp_path / "mem.jsonl")
    assert len(reloaded) == 1 and reloaded.get(board, 7.5, 100) is not None
    assert reloaded.get(board, 7.5, 10_000) is None
    assert reloaded.get(board, 6.5) is None


def test_ladder_player_matches_baseline_without_a_stopper(engine):
    board = Board(7)
    rng = np.random.default_rng(0)
    base = KataGoPlayer(engine, 64, temp=None).decide(board, 7.5, rng)
    same = DataGoPlayer(engine, [64, 256], None, None, temp=None).decide(board, 7.5, rng)
    assert (same.point, same.visits) == (base.point, base.visits)


def test_stopper_controls_ladder_depth_and_costs(engine):
    from datago.stopper import STOPPER_FEATURES, Stopper, TreeEnsemble
    leaf = {"left": [-1], "right": [-1], "feature": [0], "threshold": [0.0], "value": [0.0]}
    board = Board(7)
    rng = np.random.default_rng(0)
    never = Stopper(TreeEnsemble(1.0, 1.0, [leaf]), 0.5, [16, 64, 256])
    always = Stopper(TreeEnsemble(0.0, 1.0, [leaf]), 0.5, [16, 64, 256])
    deep = DataGoPlayer(engine, [16, 64, 256], never, None, temp=None).decide(board, 7.5, rng)
    assert deep.visits == 256 and deep.counts["restart_visits"] == 16 + 64 + 256 and deep.counts["rung"] == 2
    quick = DataGoPlayer(engine, [16, 64, 256], always, None, temp=None).decide(board, 7.5, rng)
    assert quick.visits == 16 and quick.counts["restart_visits"] == 16 and quick.counts["rung"] == 0
    assert len(STOPPER_FEATURES) == len(extract(engine.search(board, 16, 7.5), board)) + 6


def test_tree_ensemble_matches_sklearn(tmp_path):
    sklearn = pytest.importorskip("sklearn.ensemble")
    from datago.stopper import STOPPER_FEATURES, Stopper, TreeEnsemble
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(STOPPER_FEATURES)))
    y = np.maximum(0, X[:, 0] * X[:, 3] + 0.3 * X[:, 5]) + 0.05 * rng.normal(size=400)
    gbr = sklearn.GradientBoostingRegressor(n_estimators=40, max_depth=3).fit(X, y)
    ens = TreeEnsemble.from_sklearn(gbr)
    assert np.allclose(ens.raw_batch(X[:50]), gbr.predict(X[:50]), atol=1e-9)
    gbc = sklearn.GradientBoostingClassifier(n_estimators=40, max_depth=3).fit(X, y > 0.3)
    ens_c = TreeEnsemble.from_sklearn(gbc)
    assert np.allclose(ens_c.raw_batch(X[:50]), gbc.decision_function(X[:50]), atol=1e-9)
    Stopper(ens, 0.1, [50, 200]).save(tmp_path / "s.json")
    again = Stopper.load(tmp_path / "s.json")
    assert again.score(X[0]) == pytest.approx(gbr.predict(X[:1])[0], abs=1e-9)


def test_series_ledger_never_overspends_and_memory_gets_hits(engine):
    from datago.series import Deepener, Ledger, run_series
    cfg = GameConfig(size=5, komi=0.5, max_moves=40)
    mem = Memory()
    dg = DataGoPlayer(engine, [24], None, mem, name="dg")
    kg = KataGoPlayer(engine, 24, name="kg")
    ledger = Ledger(budget_per_move=24)
    hook = Deepener("dg", mem, engine, ledger, deep_visits=96, komi=0.5)
    recs = run_series(dg, kg, cfg, 24, seed=1, workers=4, after_game=hook)
    assert len(recs) == 24 and sum(r.black == "dg" for r in recs) == 12
    assert ledger.balance >= 0 and ledger.granted == 24 * sum(r.turns["dg"] for r in recs)
    hits = sum(r.counts["dg"].get("hit", 0) for r in recs)
    assert hits > 0 and ledger.deepened > 0
    assert max(e.visits for e in mem.entries.values()) == 96
    s = summarize(recs, "dg")
    assert s["visits_per_move"] < 24 and s["per_move"]["hit"] > 0


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
    assert rec.visits["a"] == sum(t["v"] for t in rec.trace if t["p"] == "a") == rec.restart_visits["a"]
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


def test_many_large_concurrent_responses_do_not_deadlock(engine):
    """Regression: 64 threads requesting policy-sized responses once stalled the pipe."""
    from concurrent.futures import ThreadPoolExecutor
    board = Board(19)
    for p in (72, 288, 60, 300, 110):
        board.play(p)
    with ThreadPoolExecutor(64) as pool:
        futs = [pool.submit(engine.search, board, 8, 7.0, include_policy=True) for _ in range(128)]
        results = [f.result(timeout=120) for f in futs]
    assert all(len(r.policy) == 362 for r in results)


def test_budget_controller_tracks_the_grant():
    from datago.series import BudgetController, Ledger
    from datago.stopper import Stopper, TreeEnsemble
    leaf = {"left": [-1], "right": [-1], "feature": [0], "threshold": [0.0], "value": [0.0]}
    calib = [{"threshold": t, "cost_continue": c} for t, c in ((0.9, 50), (0.5, 100), (0.2, 200), (0.1, 400))]
    st = Stopper(TreeEnsemble(0.0, 1.0, [leaf]), 0.5, [50, 200, 800], meta={"calibration": calib})
    ledger = Ledger(budget_per_move=200)
    ctl = BudgetController(st, ledger, gain=0.5)
    assert st.threshold == pytest.approx(0.2)
    for _ in range(40):
        ledger.granted += 200 * 100
        ledger.played += 100 * 100
        ctl()
    assert ctl.target > 300 and st.threshold < 0.2
    for _ in range(200):
        ledger.granted += 200 * 100
        ledger.played += 600 * 100
        ctl()
    assert ctl.target < 150 and st.threshold > 0.2
