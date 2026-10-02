"""Can similar (non-identical) stored positions stand in for search?

Embeds each position with KataGo's raw policy and ownership maps, aligned over
the eight board symmetries, and asks two things of a position's nearest
neighbours from other games:

1. Proposal: when the shallow search picks a different move from the deep
   search, does the neighbour's deep move (mapped through the aligning
   symmetry) point at the deep move here? Compared against simply taking the
   raw policy's top moves.
2. Gating: does the neighbours' average regret predict this position's regret?
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from datago.board import symmetry_tables
from datago.ladder import load

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--shallow", type=int, default=200)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--maps", default=None, help="separate policy/ownership file, if not in the data")
ap.add_argument("--out", default=None)
args = ap.parse_args()

lad = load(args.data, keep_maps=True, maps_path=args.maps)
n = len(lad)
size = int(round(np.sqrt(lad.ownership.shape[1])))
to_sym, from_sym = symmetry_tables(size)
ks = int(np.flatnonzero(lad.rungs == args.shallow)[0])

pol = np.maximum(lad.policy[:, :-1], 0)
own = lad.ownership


def embed(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    e = np.concatenate([np.sqrt(p), 0.5 * o], axis=1)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


base = embed(pol, own)
# Query in all eight orientations against the stored (unrotated) database.
sims = np.full((n, n), -1.0, dtype=np.float32)
best_sym = np.zeros((n, n), dtype=np.int8)
for s in range(8):
    q = embed(pol[:, from_sym[s]], own[:, from_sym[s]])
    sim = q @ base.T
    better = sim > sims
    sims[better] = sim[better]
    best_sym[better] = s
same_game = lad.game[:, None] == lad.game[None, :]
sims[same_game] = -1.0

order = np.argsort(-sims, axis=1)[:, : args.k]
nn_sim = np.take_along_axis(sims, order, axis=1)

deep_move = lad.move[:, -1]
shallow_move = lad.move[:, ks]
wrong = shallow_move != deep_move
policy_rank = np.argsort(-pol, axis=1)

report = {"positions": n, "shallow": args.shallow, "k": args.k,
          "shallow_differs_from_deep": float(wrong.mean())}

# 1. Proposal quality on positions where the shallow move differs from the deep one.
hit_nn1 = hit_nnk = hit_pol2 = hit_polk = 0
exact_dup = 0
for i in np.flatnonzero(wrong):
    props = []
    for j_rank in range(args.k):
        j = order[i, j_rank]
        s = best_sym[i, j]
        # Query was rotated by s to match j, so map j's move back with the inverse.
        props.append(int(from_sym[s][deep_move[j]]))
    hit_nn1 += props[0] == deep_move[i]
    hit_nnk += deep_move[i] in props
    cand = [int(p) for p in policy_rank[i] if p != shallow_move[i]]
    hit_pol2 += cand[0] == deep_move[i]
    hit_polk += deep_move[i] in cand[: args.k]
    exact_dup += nn_sim[i, 0] > 0.999
m = int(wrong.sum())
report["proposal"] = {
    "positions": m,
    "nearest_neighbour_deep_move_is_right": hit_nn1 / m,
    f"any_of_{args.k}_neighbours_right": hit_nnk / m,
    "policy_top_alternative_is_right": hit_pol2 / m,
    f"any_of_policy_top_{args.k}_alternatives_right": hit_polk / m,
    "nearest_neighbour_is_near_duplicate": exact_dup / m,
}

# How does neighbour agreement depend on similarity? (all positions)
agree = np.array([int(from_sym[best_sym[i, order[i, 0]]][deep_move[order[i, 0]]]) == deep_move[i]
                  for i in range(n)])
bins = [(-1, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 0.999), (0.999, 1.01)]
report["agreement_by_similarity"] = [
    {"similarity": f"{lo:.3f}-{hi:.3f}", "positions": int(((nn_sim[:, 0] >= lo) & (nn_sim[:, 0] < hi)).sum()),
     "neighbour_deep_move_matches": float(agree[(nn_sim[:, 0] >= lo) & (nn_sim[:, 0] < hi)].mean())
     if ((nn_sim[:, 0] >= lo) & (nn_sim[:, 0] < hi)).any() else None}
    for lo, hi in bins]
report["median_nn_similarity_by_move"] = {
    f"{a}-{b}": float(np.median(nn_sim[(lad.move_number >= a) & (lad.move_number < b), 0]))
    for a, b in ((0, 20), (20, 60), (60, 120), (120, 400))
    if ((lad.move_number >= a) & (lad.move_number < b)).any()}

# 2. Gating: neighbours' regret as a predictor of this position's regret.
reg = lad.regret[:, ks]
w = np.maximum(nn_sim, 0) + 1e-6
knn_regret = (reg[order] * w).sum(1) / w.sum(1)
pos = (reg > 0.005).astype(int)
lcb = lad.feats[:, ks, 10]
report["gating_auc"] = {"knn_regret": float(roc_auc_score(pos, knn_regret)),
                        "lcb_margin_feature": float(roc_auc_score(pos, -lcb))}

print(json.dumps(report, indent=1))
if args.out:
    json.dump(report, open(args.out, "w"), indent=1)
