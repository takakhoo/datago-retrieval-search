"""Where does the regret that survives a shallow search come from?

For each position with regret at a given rung, look up how much attention that
rung's search paid to the move that turned out best. If the search barely
looked at it, no statistic of that search can reveal the problem, which bounds
what any stopping rule built on search statistics can recover.
"""
import argparse
import gzip
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--out", default=None)
args = ap.parse_args()

opener = gzip.open if args.data.endswith(".gz") else open
report = {}
recs = [json.loads(l) for l in opener(args.data, "rt")]
for rung in ("50", "200", "800"):
    total = 0.0
    buckets = {"not listed or under 1% of visits": 0.0, "1-10% of visits": 0.0,
               "10-40% of visits": 0.0, "over 40% of visits": 0.0}
    low_prior = 0.0
    n_regret = 0
    for r in recs:
        forced = r["forced"]
        best_move = max(forced, key=lambda m: forced[m]["winrate"])
        search = r["ladder"][rung]
        chosen = search["moves"][0][0]
        regret = forced[best_move]["winrate"] - forced[chosen]["winrate"]
        if regret <= 0.005:
            continue
        n_regret += 1
        total += regret
        visits = {m[0]: m[1] for m in search["moves"]}
        priors = {m[0]: m[4] for m in search["moves"]}
        share = visits.get(best_move, 0) / max(search["visits"], 1)
        key = ("not listed or under 1% of visits" if share < 0.01 else "1-10% of visits" if share < 0.10
               else "10-40% of visits" if share < 0.40 else "over 40% of visits")
        buckets[key] += regret
        if priors.get(best_move, 0.0) < 0.05:
            low_prior += regret
    report[rung] = {
        "positions_with_regret_over_0.5pct": n_regret,
        "share_of_regret_by_attention_to_best_move": {k: v / total for k, v in buckets.items()},
        "share_of_regret_where_best_move_prior_under_5pct": low_prior / total,
    }
print(json.dumps(report, indent=1))
if args.out:
    json.dump(report, open(args.out, "w"), indent=1)
