"""One-off checks of engine behavior that the design depends on."""
import sys
import numpy as np
from datago.board import Board, point_to_gtp
from datago.engine import open_katago

with open_katago("b18", int(sys.argv[1])) as eng:
    b = Board(19)
    for komi in (5.5, 6.0, 6.5, 7.0, 7.5):
        r = eng.search(b, 800, komi)
        print(f"komi {komi}: black winrate {r.winrate:.3f} score {r.score_lead:+.2f}")
    r = eng.search(b, 200, 7.5, include_policy=True, include_ownership=True)
    print("moves listed after symmetry dedup:", len(r.moves), "visit sum", sum(m.visits for m in r.moves), "root", r.visits)
    print("raw:", r.raw, "ownership len", len(r.ownership), "policy len", len(r.policy))
    for mv in ("Q16", "K10", "A1"):
        from datago.board import gtp_to_point
        f = eng.search(b, 200, 7.5, allow=[gtp_to_point(mv, 19)])
        print(f"forced {mv}: moves {[point_to_gtp(m.point,19) for m in f.moves]} root wr {f.winrate:.3f} "
              f"move wr {f.best.winrate:.3f} visits {f.visits}/{f.best.visits}")
    a = [eng.search(b, 100, 7.5).winrate for _ in range(5)]
    print("repeat-search winrate spread at 100 visits:", np.round(a, 4))
