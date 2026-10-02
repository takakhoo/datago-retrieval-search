"""Does a deeper search of a position reuse the shallower one's NN evaluations?

Runs fresh engines on one midgame position and compares the number of NN rows
evaluated for (a) a 400-visit search alone and (b) a 200-visit search followed
by a 400-visit search. If the 200-visit tree is a prefix of the 400-visit one,
(b) costs the same rows as (a).
"""
import json
import re
import sys
import tempfile

from datago.board import Board
from datago.engine import open_katago

gpu = int(sys.argv[1])
games = [json.loads(l) for l in open(sys.argv[2])][:6]


def rows(schedule, moves, size, komi):
    with tempfile.TemporaryFile("w+") as err:
        with open_katago("b18", gpu, stderr=err) as eng:
            b = Board(size)
            for mv in moves:
                b.play_gtp(mv)
            best = [eng.search(b, v, komi).best.point for v in schedule]
        err.seek(0)
        return int(re.findall(r"NN rows: (\d+)", err.read())[-1]), best


for g in games:
    moves = g["moves"][: min(80, len(g["moves"]) - 2)]
    a, ba = rows([400], moves, g["size"], g["komi"])
    b, bb = rows([200, 400], moves, g["size"], g["komi"])
    c, _ = rows([200], moves, g["size"], g["komi"])
    print(f"400 alone: {a} rows | 200 alone: {c} rows | 200 then 400: {b} rows | "
          f"same final move: {ba[-1] == bb[-1]}")
