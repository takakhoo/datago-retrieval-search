import numpy as np
import pytest

from mikiri.board import (BLACK, EMPTY, PASS, WHITE, Board, IllegalMove, gtp_to_point,
                          point_to_gtp, symmetry_tables)


def play(board, moves):
    for m in moves.split():
        board.play_gtp(m)


def test_gtp_roundtrip_skips_i():
    assert point_to_gtp(gtp_to_point("J1", 19), 19) == "J1"
    assert gtp_to_point("A19", 19) == 0
    assert gtp_to_point("T1", 19) == 19 * 19 - 1
    assert point_to_gtp(PASS, 9) == "pass"
    with pytest.raises(ValueError):
        gtp_to_point("I5", 19)


def test_capture_removes_stones_and_restores_liberties():
    b = Board(9)
    play(b, "A1 A2 pass B1")
    assert b.stones[gtp_to_point("A1", 9)] == EMPTY
    assert b.num_stones() == 2


def test_suicide_is_illegal_but_capturing_into_no_liberties_is_legal():
    b = Board(9)
    play(b, "pass A2 pass B1")
    assert not b.is_legal(gtp_to_point("A1", 9))
    with pytest.raises(IllegalMove):
        b.play_gtp("A1")
    c = Board(9)
    play(c, "A2 A1 B1")
    assert c.stones[gtp_to_point("A1", 9)] == EMPTY


def test_simple_ko_is_banned_for_one_turn():
    b = Board(9)
    play(b, "D5 E5 E4 F4 E6 F6 pass G5 F5")
    assert b.stones[gtp_to_point("E5", 9)] == EMPTY
    assert b.ko == gtp_to_point("E5", 9)
    assert not b.is_legal(gtp_to_point("E5", 9))
    play(b, "A1 A9")
    assert b.is_legal(gtp_to_point("E5", 9))


def test_positional_superko_blocks_repetition():
    b = Board(9)
    play(b, "D5 E5 E4 F4 E6 F6 pass G5 F5 pass pass")
    # White retaking now is not a simple-ko violation, but recreates an earlier position.
    assert b.ko == PASS
    assert not b.is_legal(gtp_to_point("E5", 9))
    relaxed = Board(9, superko=False)
    play(relaxed, "D5 E5 E4 F4 E6 F6 pass G5 F5 pass pass")
    assert relaxed.is_legal(gtp_to_point("E5", 9))


def test_area_score_counts_territory_and_ignores_neutral_regions():
    b = Board(5)
    play(b, "C1 D1 C2 D2 C3 D3 C4 D4 C5 D5")
    assert b.area_score() == (15, 10)
    assert b.winner(0.5) == BLACK
    assert b.winner(7.5) == WHITE
    assert Board(5).area_score() == (0, 0)


def test_symmetry_tables_are_inverse_permutations():
    to_sym, from_sym = symmetry_tables(7)
    for s in range(8):
        assert sorted(to_sym[s]) == list(range(49))
        assert np.array_equal(from_sym[s][to_sym[s]], np.arange(49))
    assert len({tuple(t) for t in to_sym}) == 8


def test_all_eight_symmetric_games_share_one_key_and_moves_map_back():
    rng = np.random.default_rng(0)
    base = Board(9)
    for _ in range(30):
        legal = [p for p in base.legal_moves() if p != PASS]
        base.play(int(rng.choice(legal)))
    to_sym, _ = symmetry_tables(9)
    key0, sym0 = base.key(7.5)
    target = base.moves[-1][1]
    for s in range(8):
        b = Board(9)
        for _, p in base.moves:
            b.play(int(to_sym[s][p]))
        key, sym = b.key(7.5)
        assert key == key0
        canon = base.to_canonical(target, sym0)
        assert b.from_canonical(canon, sym) == int(to_sym[s][target])
        assert b.to_canonical(b.from_canonical(canon, sym), sym) == canon


def test_key_separates_side_to_move_ko_and_komi():
    c = Board(9)
    c.play_gtp("E5", BLACK)
    d = c.copy()
    d.to_move = BLACK
    assert c.key()[0] != d.key()[0]
    assert c.key(7.5)[0] != c.key(6.5)[0]
    ko = Board(9)
    play(ko, "D5 E5 E4 F4 E6 F6 pass G5 F5")
    no_ko = ko.copy()
    no_ko.ko = PASS
    assert ko.key()[0] != no_ko.key()[0]


def test_copy_is_independent():
    a = Board(9)
    a.play_gtp("C3")
    b = a.copy()
    b.play_gtp("D4")
    assert len(a.moves) == 1 and a.num_stones() == 1 and b.num_stones() == 2
