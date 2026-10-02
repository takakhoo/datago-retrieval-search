"""Replay recorded games on both our Board and KataGo's GTP board and compare stones.

Position keys are only as good as the rules engine underneath them, so this
checks captures and kos against KataGo itself over full games.

Usage: python check_board_vs_katago.py KATAGO MODEL GTP_CONFIG games.jsonl [n_games]
"""
import json
import subprocess
import sys

import numpy as np

from mikiri.board import BLACK, EMPTY, WHITE, Board

katago, model, config, games_path = sys.argv[1:5]
n_games = int(sys.argv[5]) if len(sys.argv) > 5 else 20
proc = subprocess.Popen([katago, "gtp", "-model", model, "-config", config],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def gtp(cmd: str) -> str:
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()
    lines = []
    while True:
        line = proc.stdout.readline()
        if line.strip() == "" and lines:
            break
        if line.strip():
            lines.append(line.rstrip("\n"))
    if not lines[0].startswith("="):
        raise RuntimeError(f"{cmd}: {lines}")
    return "\n".join(lines)


def katago_stones(size: int) -> np.ndarray:
    grid = np.zeros(size * size, dtype=np.int8)
    for line in gtp("showboard").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            row = size - int(parts[0])
            cells = parts[1]
            # Cells are two characters wide: the stone, then a marker or space.
            for x in range(size):
                c = cells[2 * x]
                grid[row * size + x] = BLACK if c == "X" else WHITE if c == "O" else EMPTY
    return grid


games = [json.loads(l) for l in open(games_path)][:n_games]
checked = captures = 0
for g in games:
    size = g["size"]
    gtp(f"boardsize {size}")
    gtp("clear_board")
    gtp("kata-set-rules tromp-taylor")
    board = Board(size)
    for i, mv in enumerate(g["moves"]):
        color = "B" if i % 2 == 0 else "W"
        before = board.num_stones()
        board.play_gtp(mv)
        captures += board.num_stones() < before + (mv != "pass")
        gtp(f"play {color} {mv}")
        if i % 25 == 24 or i == len(g["moves"]) - 1:
            if not np.array_equal(board.stones, katago_stones(size)):
                print(f"MISMATCH in game {g['game']} after move {i + 1}")
                print(board)
                print(gtp("showboard"))
                sys.exit(1)
            checked += 1
gtp("quit")
print(f"{len(games)} games, {sum(len(g['moves']) for g in games)} moves, "
      f"{captures} capturing moves, {checked} board comparisons: all identical")
