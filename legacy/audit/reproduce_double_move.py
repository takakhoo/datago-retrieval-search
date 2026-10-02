"""Reproduce the v1 match-runner flaw with v1's own GTP controller.

v1's RecursiveDataGoPlayer analysed a position by sending KataGo
`kata-genmove_analyze b`. That command does not only analyse: it also plays
the generated move on KataGo's board. v1 called it once for the standard
search and once more for every deep search, so each deep search put an extra
black stone on the board before White was allowed to reply.

Usage: python reproduce_double_move.py KATAGO MODEL GTP_CONFIG
"""
import importlib.util
import sys
from pathlib import Path

# Load v1's controller file directly so the rest of the v1 package is not needed.
_path = Path(__file__).resolve().parents[1] / "v1" / "datago" / "src" / "bot" / "gtp_controller.py"
_spec = importlib.util.spec_from_file_location("v1_gtp_controller", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GTPController = _mod.GTPController

katago, model, config = sys.argv[1:4]
gtp = GTPController([katago, "gtp", "-model", model, "-config", config])
gtp.boardsize(19)
gtp.clear_board()
gtp.komi(7.5)

gtp.set_max_visits(50)
first, _ = gtp.genmove_analyze("b")       # v1 "standard search"
gtp.set_max_visits(100)
second, _ = gtp.genmove_analyze("b")      # v1 "deep search" of the same position
third, _ = gtp.genmove_analyze("b")       # v1 recursive search of a child
white = gtp.genmove("w")                  # the opponent finally moves, at 100 visits

ok, board = gtp.send_command("showboard")
print(board)
rows = [line.split(None, 1)[1] for line in board.splitlines()
        if line.split() and line.split()[0].isdigit()]
black_stones = sum(r.count("X") for r in rows)
white_stones = sum(r.count("O") for r in rows)
print(f"\nBlack 'searches' returned: {first}, {second}, {third}. White then played: {white}.")
print(f"Stones on KataGo's board: {black_stones} black, {white_stones} white.")
print("One black turn placed three stones." if black_stones == 3 else "Unexpected stone count.")
gtp.quit()
