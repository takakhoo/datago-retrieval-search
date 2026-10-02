"""Prior stopping and budget rules, re-implemented on the ladder dataset.

Every rule sees what Mikiri's stopper sees, root statistics of the search at
each rung, and is scored the same way: by the forced value of the move it ends
up playing and by the visits it used. Threshold-style rules are swept over
their threshold, so each baseline is shown at its best operating point for
every budget.

Published rules and how they are adapted (details in the paper):
  UNST / CLOSE / BEHIND   Huang et al. 2010; Baier and Winands 2016. One planned
                          budget, extended to the next rung when the condition holds.
  STOP / smart pruning    Baier and Winands 2016; Leela Chess Zero. Stop when
                          (Nmax - n) * p < N1 - N2.
  KLD gain                Leela Chess Zero. KL(old || new) of root visit shares per new visit.
  V-MCTS                  Ye et al. 2022. L1 distance between virtually expanded
                          policies at k and k/2, with r = 0.2.
  DS-MCTS                 Lan et al. 2021. Learned probability that the current or any
                          later choice is worse than the final one by at least eps.
  VOI stop                Hay et al. 2012, Eqs. 7-11.
  BAI stop                Kaufmann and Koolen 2017, root arms only.
  state only              Muppidi et al. 2026 in spirit: a budget chosen before
                          search from raw network outputs.
Each threshold rule is run flat, as published, and with Mikiri's rate rule.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from mikiri.board import gtp_to_point
from mikiri.ladder import FEATURE_NAMES, evaluate_stops, load, sequential_stops
from mikiri.stopper import Stopper
from train_stopper import folds_by_game, path_features

GBM = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=30, random_state=0)
PHI = 8 * (np.sqrt(2) - 1) ** 2


def build_arrays(recs, policy, rungs, width=64):
    """Per position: candidate moves (visited at any rung, plus top priors), and per rung N, Q, LCB."""
    n, K = len(recs), len(rungs)
    P = np.zeros((n, width))
    N = np.zeros((n, K, width))
    Q = np.zeros((n, K, width))
    for i, r in enumerate(recs):
        size = r["size"]
        idx: dict[int, int] = {}
        for v in rungs:
            for m in r["ladder"][str(v)]["moves"]:
                p = size * size if m[0] == "pass" else gtp_to_point(m[0], size)
                idx.setdefault(p, len(idx))
        for p in np.argsort(-policy[i]):
            if len(idx) >= width:
                break
            idx.setdefault(int(p), len(idx))
        for p, j in idx.items():
            if j < width:
                P[i, j] = max(policy[i, p], 0.0)
        for k, v in enumerate(rungs):
            listed = r["ladder"][str(v)]["moves"]
            qmin = min(m[2] for m in listed)
            Q[i, k, :] = qmin
            for m in listed:
                p = size * size if m[0] == "pass" else gtp_to_point(m[0], size)
                j = idx[p]
                if j < width:
                    N[i, k, j], Q[i, k, j] = m[1], m[2]
    return P, N, Q


def virtual_policy(P, N, Q, total: int) -> np.ndarray:
    """Ye et al. Algorithm 2: fill the remaining budget by root UCB with Q frozen."""
    c1, c2 = 1.25, 19652.0
    Nh = N.copy()
    rows = np.arange(len(N))
    spent = int(round(float(np.median(N.sum(axis=1)))))
    for _ in range(max(total - spent, 0)):
        s = Nh.sum(axis=1, keepdims=True)
        ucb = Q + P * np.sqrt(s) / (1 + Nh) * (c1 + np.log((s + c2 + 1) / c2))
        Nh[rows, ucb.argmax(axis=1)] += 1
    return Nh / Nh.sum(axis=1, keepdims=True)


def frontier(lad, path, scores, points=200):
    out = []
    for t in np.unique(np.quantile(scores[:, path[:-1]].ravel(), np.linspace(0, 1, points))):
        stop = sequential_stops(lad, path, scores, t)
        r = evaluate_stops(lad, stop, path)
        r["stop"], r["threshold"] = stop, float(t)
        out.append(r)
    return out


def bootstrap(lad, stop, boots=400, seed=0):
    rng = np.random.default_rng(seed)
    games = np.unique(lad.game)
    members = {g: np.flatnonzero(lad.game == g) for g in games}
    logv = np.log(lad.rungs)
    vals = []
    for _ in range(boots):
        idx = np.concatenate([members[g] for g in rng.choice(games, len(games))])
        curve = lad.regret[idx].mean(axis=0)
        order = np.argsort(curve)
        reg = lad.regret[idx, stop[idx]].mean()
        vals.append(np.exp(np.interp(reg, curve[order], logv[order])) / lad.rungs[stop[idx]].mean())
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def paired_diff(lad, stop_a, stop_b, boots=1000, seed=0) -> dict:
    """Bootstrap over games of multiplier(a) - multiplier(b), resampling the same games for both."""
    rng = np.random.default_rng(seed)
    games = np.unique(lad.game)
    members = {g: np.flatnonzero(lad.game == g) for g in games}
    logv = np.log(lad.rungs)
    diffs = []
    for _ in range(boots):
        idx = np.concatenate([members[g] for g in rng.choice(games, len(games))])
        curve = lad.regret[idx].mean(axis=0)
        order = np.argsort(curve)

        def mult(stop):
            reg = lad.regret[idx, stop[idx]].mean()
            return np.exp(np.interp(reg, curve[order], logv[order])) / lad.rungs[stop[idx]].mean()

        diffs.append(mult(stop_a) - mult(stop_b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"difference": float(np.mean(diffs)), "lo": float(lo), "hi": float(hi),
            "p_a_better": float(np.mean(np.array(diffs) > 0))}


def best_point(points: list[dict], budget: float) -> dict | None:
    ok = [p for p in points if 0.8 * budget < p["cost_continue"] <= budget]
    return min(ok, key=lambda r: r["regret"]) if ok else None


def best_at(lad, points: list[dict], budget: float) -> dict | None:
    """Best operating point (lowest regret) with mean cost in (0.8, 1.0] of the budget."""
    ok = [p for p in points if 0.8 * budget < p["cost_continue"] <= budget]
    if not ok:
        return None
    p = min(ok, key=lambda r: r["regret"])
    lo, hi = bootstrap(lad, p["stop"])
    return {"cost": p["cost_continue"], "regret": p["regret"], "multiplier": p["gain_continue"],
            "lo": lo, "hi": hi, "multiplier_restart": p["gain_restart"], "setting": p.get("setting", "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ladder19.jsonl.gz")
    ap.add_argument("--maps", default="data/ladder19_maps.jsonl.gz")
    ap.add_argument("--out", default="results/baselines.json")
    ap.add_argument("--eps", type=float, default=0.005, help="DS-MCTS label: regret that counts as uncertain")
    args = ap.parse_args()

    lad = load(args.data, keep_maps=True, maps_path=args.maps)
    recs = [json.loads(l) for l in gzip.open(args.data, "rt")]
    n, K = lad.regret.shape
    rungs = [int(v) for v in lad.rungs]
    f = {name: lad.feats[:, :, i] for i, name in enumerate(FEATURE_NAMES)}
    folds = folds_by_game(lad.game, 5, 0)
    budgets = [100.0, 200.0, 400.0, 800.0]
    full = list(range(K))
    quad = [0, 2, 4, 6]
    rows = np.arange(n)

    P, N, Q = build_arrays(recs, lad.policy, rungs)
    share = N / np.maximum(N.sum(axis=2, keepdims=True), 1)
    order = np.argsort(-N, axis=2)
    n1 = np.take_along_axis(N, order[:, :, :1], axis=2)[:, :, 0]
    n2 = np.take_along_axis(N, order[:, :, 1:2], axis=2)[:, :, 0]
    visited = N > 0
    qv = np.where(visited, Q, -np.inf)
    best_q = qv.argmax(axis=2)
    most_visited = order[:, :, 0]

    points: dict[str, list[dict]] = {}

    def add(name: str, stop: np.ndarray, path: list[int], setting: str = "") -> None:
        r = evaluate_stops(lad, stop, path)
        r["stop"], r["setting"] = stop, setting
        points.setdefault(name, []).append(r)

    # --- extension rules: one planned rung, extend to the next when the condition holds ---
    wr_best = np.take_along_axis(Q, most_visited[:, :, None], axis=2)[:, :, 0]
    conds = {
        "UNST": most_visited != best_q,
        "CLOSE (d=0.4)": (n1 - n2) / np.maximum(n1, 1) < 0.4,
        "BEHIND (v=0.6)": wr_best < 0.6,
    }
    for name, cond in conds.items():
        for b in range(K - 1):
            stop = np.where(cond[:, b], b + 1, b)
            add(name, stop, [b, b + 1], f"planned {rungs[b]}")
            if b + 2 < K:  # loop variant: one more extension if still true
                stop2 = np.where(cond[:, b] & cond[:, b + 1], b + 2, stop)
                add(name + ", looped", stop2, [b, b + 1, b + 2], f"planned {rungs[b]}")

    # --- STOP / smart pruning: stop when the runner-up cannot catch up ---
    for p_stop, label in ((1.0, "STOP (p=1)"), (0.75, "smart pruning (factor 1.33)"), (0.45, "STOP (p=0.45)")):
        for top in range(1, K):
            stop = np.full(n, top)
            active = np.ones(n, dtype=bool)
            for k in range(top):
                halt = active & ((rungs[top] - rungs[k]) * p_stop < n1[:, k] - n2[:, k])
                stop[halt] = k
                active &= ~halt
            add(label, stop, list(range(top + 1)), f"Nmax {rungs[top]}")

    # --- threshold rules on consecutive rungs ---
    prior = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-9)
    kld = np.zeros((n, K))
    for k in range(K):
        old = prior if k == 0 else share[:, k - 1]
        new = np.maximum(share[:, k], 1e-4)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(old > 0, old * np.log(old / new), 0.0)
        kld[:, k] = term.sum(axis=1) / (rungs[k] - (rungs[k - 1] if k else 0))

    # Hay et al.: stop when both value-of-information bounds are below c.
    voi = np.zeros((n, K))
    bai = np.zeros((n, K))
    for k in range(K):
        a = best_q[:, k]
        xa, na = Q[rows, k, a], np.maximum(N[rows, k, a], 1)
        others = visited[:, k].copy()
        others[rows, a] = False
        xo = np.where(others, Q[:, k], -np.inf)
        b = xo.argmax(axis=1)
        xb = np.where(others.any(axis=1), Q[rows, k, b], 0.0)
        t1 = xb / na * 2 * np.exp(-PHI * (xa - xb) ** 2 * na)
        ni = np.maximum(N[:, k], 1)
        t2 = np.where(others, (1 - xa)[:, None] / ni * 2 * np.exp(-PHI * (xa[:, None] - Q[:, k]) ** 2 * ni), 0.0)
        voi[:, k] = np.maximum(t1, t2.max(axis=1))
        beta = np.log(np.log(np.e * np.maximum(N[:, k], 1)) / 0.1)
        rad = np.sqrt(np.maximum(beta, 0) / (2 * np.maximum(N[:, k], 1)))
        upper = np.where(others, Q[:, k] + rad, -np.inf).max(axis=1)
        lower = xa - rad[rows, a]
        bai[:, k] = np.where(others.any(axis=1), upper - lower, 0.0)

    # --- V-MCTS: virtually expanded policies at k and k/2 for each maximum budget ---
    vm_points = []
    for top in range(2, K):
        total = rungs[top]
        checks = [k for k in range(1, top) if rungs[k] >= 0.2 * total]
        dist = {}
        for k in checks:
            a = virtual_policy(P, N[:, k], Q[:, k], total)
            b = virtual_policy(P, N[:, k - 1], Q[:, k - 1], total)
            dist[k] = np.abs(a - b).sum(axis=1)
        allv = np.concatenate(list(dist.values()))
        for eps in sorted(set(np.quantile(allv, np.linspace(0, 1, 60)).tolist() + [0.1])):
            stop = np.full(n, top)
            active = np.ones(n, dtype=bool)
            for k in checks:
                halt = active & (dist[k] < eps)
                stop[halt] = k
                active &= ~halt
            # V-MCTS searches straight through to the checkpoint, so only continuation cost applies.
            r = evaluate_stops(lad, stop, None)
            r["stop"] = stop
            r["setting"] = f"N {total}, eps {eps:.3f}" + (" (published)" if abs(eps - 0.1) < 1e-12 else "")
            vm_points.append(r)
    points["V-MCTS"] = vm_points
    points["V-MCTS (r=0.2, eps=0.1)"] = [p for p in vm_points if "published" in p["setting"]]

    # --- learned signals, cross-fitted by game ---
    q_final = lad.q[:, -1]
    later_worst = np.maximum.accumulate((q_final[:, None] - lad.q)[:, ::-1], axis=1)[:, ::-1]
    uncertain = (later_worst >= args.eps).astype(int)   # U(s, n) of Lan et al.

    pol = np.maximum(lad.policy[:, :-1], 1e-9)
    pol /= pol.sum(axis=1, keepdims=True)
    entropy = -(pol * np.log(pol)).sum(axis=1)
    raw_wr = np.array([r["ladder"][str(rungs[0])]["raw_winrate"] for r in recs])
    state = np.stack([entropy, pol.max(axis=1), 1 - 2 * np.abs(raw_wr - 0.5), f["raw_wr_error"][:, 0],
                      f["raw_score_error"][:, 0], f["raw_var_time_left"][:, 0], f["phase"][:, 0]], axis=1)

    summary: dict = {"positions": n, "eps": args.eps, "budgets": budgets, "ladders": {}}
    for pname, path in (("50,200,800,3200", quad), ("doubling", full)):
        Pn = len(path)
        X = path_features(lad, path)
        regret = lad.regret[:, path]
        steps = np.diff(lad.rungs[path]).astype(float)
        rate = np.ones(K)
        rate[path[:-1]] = steps[0] / steps
        learned = {name: np.zeros((n, K)) for name in ("Mikiri", "DS-MCTS", "state only")}
        flat = lambda a, m: a[m][:, :-1].reshape(-1, *a.shape[2:])  # noqa: E731
        for fold in np.unique(folds):
            tr, te = folds != fold, folds == fold
            m1 = GradientBoostingRegressor(**GBM).fit(flat(X, tr), np.sqrt(flat(regret, tr)))
            learned["Mikiri"][np.ix_(te, path[:-1])] = (
                np.maximum(m1.predict(flat(X, te)), 0) ** 2).reshape(-1, Pn - 1)
            m2 = GradientBoostingClassifier(**GBM).fit(flat(X, tr), flat(uncertain[:, path], tr))
            learned["DS-MCTS"][np.ix_(te, path[:-1])] = m2.predict_proba(flat(X, te))[:, 1].reshape(-1, Pn - 1)
            lv = np.log(lad.rungs[path[:-1]])
            S_tr = np.concatenate([np.repeat(state[tr], Pn - 1, axis=0), np.tile(lv, tr.sum())[:, None]], axis=1)
            S_te = np.concatenate([np.repeat(state[te], Pn - 1, axis=0), np.tile(lv, te.sum())[:, None]], axis=1)
            m3 = GradientBoostingRegressor(**GBM).fit(S_tr, np.sqrt(flat(regret, tr)))
            learned["state only"][np.ix_(te, path[:-1])] = (
                np.maximum(m3.predict(S_te), 0) ** 2).reshape(-1, Pn - 1)

        signals = {
            "KLD gain": kld,
            "VOI stop": voi,
            "BAI stop": bai,
            "DS-MCTS": learned["DS-MCTS"],
            "state only": learned["state only"],
            "LCB margin": np.maximum(-f["lcb_margin"], 0) + 1e-6 * f["visit_entropy"],
            "Mikiri": learned["Mikiri"],
            "oracle": lad.regret,
        }
        table = {}
        for name, s in signals.items():
            table[f"{name} | flat"] = {str(int(b)): best_at(lad, frontier(lad, path, s), b) for b in budgets}
            if name != "KLD gain":
                table[f"{name} | rate"] = {
                    str(int(b)): best_at(lad, frontier(lad, path, s * rate[None, :]), b) for b in budgets}
        if pname == "doubling":
            for name, pts in points.items():
                table[f"{name} | as published"] = {str(int(b)): best_at(lad, pts, b) for b in budgets}
            ds01 = evaluate_stops(lad, sequential_stops(lad, path, learned["DS-MCTS"], 0.1), path)
            table["DS-MCTS (threshold 0.1) | as published"] = {
                "own": {"cost": ds01["cost_continue"], "regret": ds01["regret"],
                        "multiplier": ds01["gain_continue"], "multiplier_restart": ds01["gain_restart"]}}
            fronts = {"Mikiri": frontier(lad, path, learned["Mikiri"] * rate[None, :]),
                      "DS-MCTS flat": frontier(lad, path, learned["DS-MCTS"]),
                      "DS-MCTS rate": frontier(lad, path, learned["DS-MCTS"] * rate[None, :]),
                      "KLD gain": frontier(lad, path, kld), "V-MCTS": points["V-MCTS"],
                      "V-MCTS published": points["V-MCTS (r=0.2, eps=0.1)"]}
            paired = {}
            for b in budgets:
                ours = best_point(fronts["Mikiri"], b)
                for other in ("V-MCTS", "V-MCTS published", "DS-MCTS flat", "DS-MCTS rate", "KLD gain"):
                    theirs = best_point(fronts[other], b)
                    if ours and theirs:
                        paired[f"Mikiri minus {other} at {int(b)}"] = paired_diff(lad, ours["stop"], theirs["stop"])
            summary["paired"] = paired
            summary["point_rules"] = {
                name: [{"setting": q["setting"], "cost": q["cost_continue"], "multiplier": q["gain_continue"],
                        "multiplier_restart": q["gain_restart"]} for q in pts]
                for name, pts in points.items() if not name.startswith("V-MCTS")}
        summary["ladders"][pname] = table
        print(f"\n=== ladder {pname}: multiplier by continuation cost [95% interval] (restart) ===", flush=True)
        for name, row in table.items():
            cells = []
            for b, v in row.items():
                if v is None:
                    continue
                ci = f" [{v['lo']:.2f},{v['hi']:.2f}]" if "lo" in v else ""
                cells.append(f"{b}@{v['cost']:.0f}: x{v['multiplier']:.2f}{ci} ({v['multiplier_restart']:.2f})")
            print(f"  {name:42s} " + "  ".join(cells), flush=True)

    print("\n=== point rules: every configuration at its own cost ===")
    for name, pts in summary["point_rules"].items():
        print(f"  {name:32s} " + "  ".join(f"{q['cost']:.0f}: x{q['multiplier']:.2f}" for q in pts[:7]))
    print("\n=== paired differences in multiplier (bootstrap over games) ===")
    for name, d in summary["paired"].items():
        print(f"  {name:40s} {d['difference']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]  P(Mikiri better) {d['p_a_better']:.3f}")
    Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
