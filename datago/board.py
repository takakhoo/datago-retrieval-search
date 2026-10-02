"""Go rules, Tromp-Taylor scoring, and symmetry-canonical position keys.

The board is the source of truth for position identity. KataGo remains the
authority on move legality during play (players only choose among moves the
engine returns), but memory keys must reflect captures and ko exactly, so the
rules here are complete: captures, suicide, simple ko, positional superko.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
PASS = -1
GTP_COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def opponent(color: int) -> int:
    return BLACK + WHITE - color


def color_name(color: int) -> str:
    return "B" if color == BLACK else "W"


@lru_cache(maxsize=None)
def _neighbors(size: int) -> tuple[tuple[int, ...], ...]:
    out = []
    for i in range(size * size):
        y, x = divmod(i, size)
        nb = []
        if y > 0:
            nb.append(i - size)
        if y < size - 1:
            nb.append(i + size)
        if x > 0:
            nb.append(i - 1)
        if x < size - 1:
            nb.append(i + 1)
        out.append(tuple(nb))
    return tuple(out)


@lru_cache(maxsize=None)
def symmetry_tables(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (to_sym, from_sym), each of shape (8, size*size).

    to_sym[s][i] is where point i lands under symmetry s.
    from_sym[s][j] is the point that lands on j, so board[from_sym[s]] is the
    transformed board.
    """
    idx = np.arange(size * size).reshape(size, size)
    from_sym = []
    for s in range(8):
        a = np.rot90(idx, s % 4)
        if s >= 4:
            a = np.fliplr(a)
        from_sym.append(a.reshape(-1).copy())
    from_sym = np.stack(from_sym)
    to_sym = np.empty_like(from_sym)
    for s in range(8):
        to_sym[s][from_sym[s]] = np.arange(size * size)
    return to_sym, from_sym


def point_to_gtp(point: int, size: int) -> str:
    if point == PASS:
        return "pass"
    y, x = divmod(point, size)
    return f"{GTP_COLS[x]}{size - y}"


def gtp_to_point(move: str, size: int) -> int:
    move = move.strip().upper()
    if move == "PASS":
        return PASS
    x = GTP_COLS.index(move[0])
    y = size - int(move[1:])
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"move {move} is off a {size}x{size} board")
    return y * size + x


class IllegalMove(ValueError):
    pass


class Board:
    __slots__ = ("size", "stones", "to_move", "ko", "moves", "_seen", "superko")

    def __init__(self, size: int = 19, superko: bool = True):
        self.size = size
        self.stones = np.zeros(size * size, dtype=np.int8)
        self.to_move = BLACK
        self.ko = PASS
        self.moves: list[tuple[int, int]] = []
        self.superko = superko
        self._seen = {self.stones.tobytes()}

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.size = self.size
        b.stones = self.stones.copy()
        b.to_move = self.to_move
        b.ko = self.ko
        b.moves = list(self.moves)
        b.superko = self.superko
        b._seen = set(self._seen)
        return b

    def _group(self, start: int) -> tuple[list[int], bool]:
        """Flood-fill the group at start. Returns (stones, has_liberty)."""
        nbrs = _neighbors(self.size)
        color = self.stones[start]
        stack, seen = [start], {start}
        has_lib = False
        while stack:
            p = stack.pop()
            for q in nbrs[p]:
                c = self.stones[q]
                if c == EMPTY:
                    has_lib = True
                elif c == color and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return list(seen), has_lib

    def _try(self, point: int, color: int):
        """Apply a stone placement on a scratch copy. Returns (stones, captured)."""
        if self.stones[point] != EMPTY:
            raise IllegalMove("point occupied")
        saved = self.stones
        self.stones = saved.copy()
        try:
            self.stones[point] = color
            captured: list[int] = []
            for q in _neighbors(self.size)[point]:
                if self.stones[q] == opponent(color):
                    group, has_lib = self._group(q)
                    if not has_lib:
                        captured.extend(group)
                        self.stones[group] = EMPTY
            if not captured and not self._group(point)[1]:
                raise IllegalMove("suicide")
            return self.stones, captured
        finally:
            self.stones = saved

    def is_legal(self, point: int, color: int | None = None) -> bool:
        color = self.to_move if color is None else color
        if point == PASS:
            return True
        if point == self.ko and color == self.to_move:
            return False
        try:
            stones, _ = self._try(point, color)
        except IllegalMove:
            return False
        return not (self.superko and stones.tobytes() in self._seen)

    def legal_moves(self) -> list[int]:
        empties = np.flatnonzero(self.stones == EMPTY)
        return [int(p) for p in empties if self.is_legal(int(p))] + [PASS]

    def play(self, point: int, color: int | None = None) -> None:
        color = self.to_move if color is None else color
        if point == PASS:
            self.ko = PASS
        else:
            if point == self.ko and color == self.to_move:
                raise IllegalMove("ko")
            stones, captured = self._try(point, color)
            if self.superko and stones.tobytes() in self._seen:
                raise IllegalMove("superko")
            self.stones = stones
            self.ko = PASS
            if len(captured) == 1:
                group, _ = self._group(point)
                if len(group) == 1:
                    libs = [q for q in _neighbors(self.size)[point] if stones[q] == EMPTY]
                    if libs == [captured[0]]:
                        self.ko = captured[0]
            self._seen.add(stones.tobytes())
        self.moves.append((color, point))
        self.to_move = opponent(color)

    def play_gtp(self, move: str, color: int | None = None) -> None:
        self.play(gtp_to_point(move, self.size), color)

    def gtp_moves(self) -> list[list[str]]:
        return [[color_name(c), point_to_gtp(p, self.size)] for c, p in self.moves]

    def consecutive_passes(self) -> int:
        n = 0
        for _, p in reversed(self.moves):
            if p != PASS:
                break
            n += 1
        return n

    def num_stones(self) -> int:
        return int(np.count_nonzero(self.stones))

    def area_score(self) -> tuple[int, int]:
        """Tromp-Taylor area: stones plus empty regions bordering one color only."""
        nbrs = _neighbors(self.size)
        black = int(np.count_nonzero(self.stones == BLACK))
        white = int(np.count_nonzero(self.stones == WHITE))
        seen = np.zeros(self.size * self.size, dtype=bool)
        for start in np.flatnonzero(self.stones == EMPTY):
            if seen[start]:
                continue
            stack, region, borders = [int(start)], 0, set()
            seen[start] = True
            while stack:
                p = stack.pop()
                region += 1
                for q in nbrs[p]:
                    c = self.stones[q]
                    if c == EMPTY:
                        if not seen[q]:
                            seen[q] = True
                            stack.append(q)
                    else:
                        borders.add(int(c))
            if borders == {BLACK}:
                black += region
            elif borders == {WHITE}:
                white += region
        return black, white

    def winner(self, komi: float) -> int:
        """BLACK, WHITE, or EMPTY for a tie under area scoring."""
        black, white = self.area_score()
        diff = black - white - komi
        return BLACK if diff > 0 else WHITE if diff < 0 else EMPTY

    def canonical(self) -> tuple[bytes, int]:
        """Lexicographically smallest encoding over the 8 symmetries.

        The encoding covers stones, the ko point, and the side to move, so two
        positions share it exactly when they are the same game state up to
        symmetry (ignoring superko history).
        """
        _, from_sym = symmetry_tables(self.size)
        state = self.stones.copy()
        if self.ko != PASS:
            state[self.ko] = 3
        best, best_sym = None, 0
        for s in range(8):
            enc = state[from_sym[s]].tobytes()
            if best is None or enc < best:
                best, best_sym = enc, s
        return bytes([self.size, self.to_move]) + best, best_sym

    def key(self, komi: float = 7.5, rules: str = "tromp-taylor") -> tuple[str, int]:
        """(hex key, symmetry index mapping this board onto the canonical one)."""
        enc, sym = self.canonical()
        digest = hashlib.blake2b(enc + f"|{komi:.1f}|{rules}".encode(), digest_size=16)
        return digest.hexdigest(), sym

    def to_canonical(self, point: int, sym: int) -> int:
        return PASS if point == PASS else int(symmetry_tables(self.size)[0][sym][point])

    def from_canonical(self, point: int, sym: int) -> int:
        return PASS if point == PASS else int(symmetry_tables(self.size)[1][sym][point])

    def __str__(self) -> str:
        chars = ".XO"
        rows = []
        for y in range(self.size):
            row = " ".join(chars[self.stones[y * self.size + x]] for x in range(self.size))
            rows.append(f"{self.size - y:2d} {row}")
        rows.append("   " + " ".join(GTP_COLS[: self.size]))
        return "\n".join(rows)
