"""Write the paper's LaTeX tables straight from the JSON results, so no number is retyped."""
from __future__ import annotations

import json
from pathlib import Path

RES = Path("results")
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


def at_cost(front: list[dict], budget: float) -> dict | None:
    ok = [r for r in front if r["cost_continue"] <= budget]
    return max(ok, key=lambda r: r["cost_continue"]) if ok else None


def uniform_table() -> None:
    r = json.loads((RES / "ladder/report.json").read_text())
    conc = {c["visits"]: c for c in r["concentration"]}
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Visits & Mean regret & Positions with & Regret held by & Move differs from \\",
             r" & (\% winrate) & no regret (\%) & worst 10\% (\%) & 3{,}200-visit move (\%) \\",
             r"\midrule"]
    for u in r["uniform"]:
        v = u["visits"]
        visits = f"{v:,}".replace(",", "{,}")
        lines.append(f"{visits} & {100 * u['regret']:.2f} & {100 * (1 - u['share_with_any_regret']):.0f} & "
                     f"{100 * conc[v]['top_10pct_of_positions_hold']:.0f} & "
                     f"{100 * u['move_differs_from_3200']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("uniform", "\n".join(lines) + "\n")


def frontier_table(policy: str = "50,200,800,3200:sqrt") -> None:
    r = json.loads((RES / "stopper/report.json").read_text())
    import numpy as np
    rungs, curve = np.array(r["rungs"]), np.array(r["uniform_regret"])
    pol = r["policies"][policy]
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"Mean & Uniform & Stopper & Equivalent & \multicolumn{2}{c}{Multiplier} & Oracle \\",
             r"visits & regret (\%) & regret (\%) & uniform visits & continuation & restart & multiplier \\",
             r"\midrule"]
    for b in (100, 150, 200, 300, 400, 600, 800):
        pt, orc = at_cost(pol["frontier"], b), at_cost(pol["oracle"], b)
        uni = float(np.interp(np.log(b), np.log(rungs), curve))
        lines.append(f"{b} & {100 * uni:.2f} & {100 * pt['regret']:.2f} & {pt['equivalent_visits']:.0f} & "
                     f"{pt['gain_continue']:.2f} & {pt['gain_restart']:.2f} & {orc['gain_continue']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("frontier", "\n".join(lines) + "\n")


def rules_table() -> None:
    r = json.loads((RES / "ladder/report.json").read_text())
    rows = r["heuristic_rules"]["rows"]
    names = ["random", "unstable (visits vs value)", "v1 entropy gate", "lcb margin"]
    label = {"random": "Random stopping", "unstable (visits vs value)": "Most-visited is not best-valued",
             "v1 entropy gate": "v1 entropy gate", "lcb margin": "LCB margin"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Flat threshold} & \multicolumn{2}{c}{Rate rule} \\",
             r"Stopping signal & 200 visits & 400 visits & 200 visits & 400 visits \\", r"\midrule"]
    for n in names:
        cell = {(x["scaling"], x["budget"]): x["multiplier"] for x in rows if x["rule"] == n}
        lines.append(f"{label[n]} & {cell[('flat', 200.0)]:.2f} & {cell[('flat', 400.0)]:.2f} & "
                     f"{cell[('rate', 200.0)]:.2f} & {cell[('rate', 400.0)]:.2f} \\\\")
    ab = RES / "stopper/ablation.json"
    if ab.exists():
        a = json.loads(ab.read_text())["variants"]
        if "flat threshold (no rate rule)" in a and "full model" in a:
            f, m = a["flat threshold (no rate rule)"], a["full model"]
            lines.append(r"\midrule")
            lines.append(f"Learned stopper & {f['200']['continue']:.2f} & {f['400']['continue']:.2f} & "
                         f"{m['200']['continue']:.2f} & {m['400']['continue']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("rules", "\n".join(lines) + "\n")


def ablation_table() -> None:
    ab = RES / "stopper/ablation.json"
    if not ab.exists():
        return
    a = json.loads(ab.read_text())["variants"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Variant & 100 & 200 & 400 & 800 \\", r"\midrule"]
    for name, v in a.items():
        cells = " & ".join(f"{v[b]['continue']:.2f} ({v[b]['restart']:.2f})" if b in v else "--"
                           for b in ("100", "200", "400", "800"))
        shown = (name[0].upper() + name[1:]).replace(",", ", ")
        lines.append(f"{shown} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("ablation", "\n".join(lines) + "\n")


def auc_table() -> None:
    r = json.loads((RES / "ladder/report.json").read_text())
    rungs = r["rungs"]
    keep = ["top1_share (low)", "lcb_margin (low)", "v1 entropy gate", "policy_surprise",
            "visit_value_disagree", "katago rawStWrError", "value_surprise"]
    label = {"top1_share (low)": "Top move's visit share", "lcb_margin (low)": "LCB margin",
             "v1 entropy gate": "v1 entropy gate", "policy_surprise": "KL(visits, prior)",
             "visit_value_disagree": "Most-visited is not best-valued",
             "katago rawStWrError": "KataGo's predicted winrate error",
             "value_surprise": "Searched minus raw winrate"}
    idx = [rungs.index(v) for v in (50, 200, 800)]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Signal & 50 & 200 & 800 \\", r"\midrule"]
    for k in keep:
        lines.append(f"{label[k]} & " + " & ".join(f"{r['auc'][k][i]:.2f}" for i in idx) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("auc", "\n".join(lines) + "\n")


def matches_table() -> None:
    p = RES / "matches/table.json"
    if not p.exists():
        return
    rows = [r for r in json.loads(p.read_text()) if not r["run"].startswith("pilot")]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Player & Opponent & Games & W--L--D & Elo (95\% interval) & Visits & Restart & Evals \\",
             r"\midrule"]
    for r in rows:
        ev = f"{r['rows']:.0f}/{r['base_rows']:.0f}" if r["rows"] else "--"
        lines.append(f"{r['player']} & {r['baseline']} & {r['games']} & {r['wld'].replace('-', '--')} & "
                     f"${r['elo']:+.0f}$ (${r['elo_lo']:+.0f}$ to ${r['elo_hi']:+.0f}$) & "
                     f"{r['visits']:.0f} & {r['restart']:.0f} & {ev} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("matches", "\n".join(lines) + "\n")


def prediction_table() -> None:
    p = RES / "match_analysis.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())["prediction"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & Mean visits & Offline regret & Equivalent & Predicted & Measured Elo \\",
             r"Ladder & per move & (\% winrate) & uniform visits & Elo & (95\% interval) \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r.get('ladder', 'four-rung')} & {r['visits']:.0f} & {100 * r['offline_regret']:.2f} & "
                     f"{r['equivalent_uniform_visits']:.0f} & "
                     f"${r['predicted_elo']:+.0f}$ & ${r['measured_elo']:+.0f}$ "
                     f"(${r['measured_lo']:+.0f}$ to ${r['measured_hi']:+.0f}$) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("prediction", "\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Ladder & Visits & Equiv. & Forecast & Measured (95\% interval) \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r.get('ladder', 'four-rung')} & {r['visits']:.0f} & {r['equivalent_uniform_visits']:.0f} & "
                     f"${r['predicted_elo']:+.0f}$ & ${r['measured_elo']:+.0f}$ "
                     f"(${r['measured_lo']:+.0f}$, ${r['measured_hi']:+.0f}$) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("prediction_compact", "\n".join(lines) + "\n")


def targets_table() -> None:
    p = RES / "stopper/targets.json"
    if not p.exists():
        return
    lad = json.loads(p.read_text())["ladders"]
    four, six, seven = (lad[k] for k in ("50,200,800,3200", "50,200,400,800,1600,3200",
                                         "50,100,200,400,800,1600,3200"))
    rows = [("$r_j$", "1 (flat)", "regret / flat"),
            ("$r_j$ (rate rule)", "$v_{j+1}-v_j$", "regret / rate"),
            ("$r_j$", "$v_m-v_j$", "regret / to_top"),
            ("$r_j-r_{j+1}$", "$v_{j+1}-v_j$", "step / rate"),
            ("$r_j-r_m$", "$v_{j+1}-v_j$", "top / rate"),
            ("$\\max_{k>j}\\frac{r_j-r_k}{v_k-v_j}$", "built in", "index / flat")]

    def cell(table, key, budget):
        v = table.get(key, {}).get(budget)
        return f"{v['gain']:.2f}" if v else "--"

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & & Oracle, & \multicolumn{3}{c}{Learned, 200 visits} \\",
             r"Scored & Divided by & 100 visits & 4 & 6 & 7 rungs \\", r"\midrule"]
    for name, price, key in rows:
        lines.append(f"{name} & {price} & {cell(four, 'oracle ' + key, '100')} & {cell(four, 'learned ' + key, '200')} & "
                     f"{cell(six, 'learned ' + key, '200')} & {cell(seven, 'learned ' + key, '200')} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{2}}{{l}}{{Hindsight optimum}} & {cell(four, 'hindsight optimum', '100')} & -- & -- & -- \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("targets", "\n".join(lines) + "\n")


COMPACT = [
    ("uni_100_vs_200", "KataGo, 100 visits"), ("uni_283_vs_200", "KataGo, 283 visits"),
    ("uni_400_vs_200", "KataGo, 400 visits"), None,
    ("double_200", "Mikiri, seven-rung ladder"),
    ("long_200", "Mikiri, six-rung ladder"), ("main_200", "Mikiri, four-rung ladder"),
    ("main_stopper_160", "Mikiri, four-rung, 160-visit grant"), ("main_140", "Mikiri, four-rung, 140-visit grant"),
    ("paired_greedy_200", "Mikiri, four-rung, no move sampling"), None,
    ("vmcts_200", "V-MCTS rule (Ye et al.)"), ("dsmcts_200", "DS-MCTS-style rule (Lan et al.)"),
    ("rule_lcb_200", "LCB margin with rate rule, no learning"), ("v1gate_200", "Entropy gate, flat threshold"), None,
    ("mem_only_200", "Memory only"), ("full_200", "Mikiri four-rung + memory"),
]
OTHER = [
    ("budget_100", "100"), ("budget_400", "400"), ("budget_800", "800"),
    ("b28_200", "200, b28 network"),
    ("size13_200", "200, 13$\\times$13"), ("size9_200", "200, 9$\\times$9"),
]


def compact_tables() -> None:
    p = RES / "matches/table.json"
    if not p.exists():
        return
    rows = {r["run"]: r for r in json.loads(p.read_text())}

    def line(label: str, r: dict | None) -> str:
        if r is None:
            return f"{label} & \\multicolumn{{6}}{{c}}{{(running)}} \\\\"
        ev = f"{r['rows']:.0f} : {r['base_rows']:.0f}" if r["rows"] else "--"
        return (f"{label} & {r['games']:,} & {r['wld'].replace('-', '--')} & "
                f"${r['elo']:+.0f}$ (${r['elo_lo']:+.0f}$, ${r['elo_hi']:+.0f}$) & "
                f"{r['visits']:.0f} & {r['restart']:.0f} & {ev} \\\\").replace(",", "{,}", 0)

    head = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Player (vs.\ KataGo at 200 visits) & Games & W--L--D & Elo (95\% interval) & Visits & Restart & Evals \\",
            r"\midrule"]
    body = [r"\midrule" if item is None else line(item[1], rows.get(item[0])) for item in COMPACT]
    write("matches_compact", "\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    head2 = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Opponent visits & Games & W--L--D & Elo (95\% interval) \\", r"\midrule"]
    body2 = []
    for run, label in OTHER:
        r = rows.get(run)
        body2.append(f"{label} & \\multicolumn{{3}}{{c}}{{(running)}} \\\\" if r is None else
                     f"{label} & {r['games']} & {r['wld'].replace('-', '--')} & "
                     f"${r['elo']:+.0f}$ (${r['elo_lo']:+.0f}$, ${r['elo_hi']:+.0f}$) \\\\")
    write("matches_other", "\n".join(head2 + body2 + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def baselines_table() -> None:
    p = RES / "baselines.json"
    if not p.exists():
        return
    b = json.loads(p.read_text())
    d = b["ladders"]["doubling"]

    def cell(key: str, budget: str) -> str:
        v = d.get(key, {}).get(budget)
        return f"{v['multiplier']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" if v else "--"

    def point(rule: str, lo: float, hi: float) -> str:
        pts = [x for x in b["point_rules"][rule] if lo <= x["cost"] <= hi]
        return f"{pts[0]['multiplier']:.2f} at {pts[0]['cost']:.0f}" if pts else "--"

    rows = [
        ("VOI stop \\cite{hay2012selecting}", cell("VOI stop | flat", "200"), cell("VOI stop | flat", "400"), cell("VOI stop | rate", "200")),
        ("BAI stop \\cite{kaufmann2017monte}", cell("BAI stop | flat", "200"), cell("BAI stop | flat", "400"), cell("BAI stop | rate", "200")),
        ("BEHIND \\cite{huang2010time}", point("BEHIND (v=0.6)", 150, 260), point("BEHIND (v=0.6)", 300, 520), "--"),
        ("STOP, $p=0.45$ \\cite{baier2016time}", point("STOP (p=0.45)", 120, 270), point("STOP (p=0.45)", 300, 520), "--"),
        ("Smart pruning \\cite{lc0_options}", point("smart pruning (factor 1.33)", 150, 260), point("smart pruning (factor 1.33)", 300, 520), "--"),
        ("CLOSE \\cite{baier2016time}", point("CLOSE (d=0.4)", 150, 260), point("CLOSE (d=0.4)", 300, 520), "--"),
        ("UNST \\cite{huang2010time}", point("UNST", 150, 260), point("UNST", 300, 520), "--"),
        ("KLD gain \\cite{lc0_options}", cell("KLD gain | flat", "200"), cell("KLD gain | flat", "400"), "--"),
        ("State only \\cite{muppidi2026finding}", cell("state only | flat", "200"), cell("state only | flat", "400"), cell("state only | rate", "200")),
        ("DS-MCTS \\cite{lan2021learning}", cell("DS-MCTS | flat", "200"), cell("DS-MCTS | flat", "400"), cell("DS-MCTS | rate", "200")),
        ("V-MCTS \\cite{ye2022spending}", cell("V-MCTS (r=0.2, eps=0.1) | as published", "200"),
         cell("V-MCTS (r=0.2, eps=0.1) | as published", "400"), "--"),
        ("LCB margin", cell("LCB margin | flat", "200"), cell("LCB margin | flat", "400"), cell("LCB margin | rate", "200")),
        ("\\textbf{Mikiri}", cell("Mikiri | flat", "200"), cell("Mikiri | flat", "400"),
         "\\textbf{" + cell("Mikiri | rate", "200") + "}"),
    ]
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Rule & As published, 200 & As published, 400 & With rate rule, 200 \\", r"\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("baselines", "\n".join(lines) + "\n")
    pr = b["paired"]
    extra = ["% paired differences"]
    for key, name in (("Mikiri minus V-MCTS published at 200", "pairedVmctsTwo"),
                      ("Mikiri minus V-MCTS published at 400", "pairedVmctsFour"),
                      ("Mikiri minus DS-MCTS flat at 200", "pairedDsTwo"),
                      ("Mikiri minus DS-MCTS rate at 400", "pairedDsRateFour")):
        v = pr[key]
        extra.append("\\newcommand{\\" + name + "}{" + f"{v['difference']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" + "}")
    (OUT / "baseline_numbers.tex").write_text("\n".join(extra) + "\n")


RUNS = {
    "main": "main_200", "strict": "main_stopper_160", "lean": "main_140", "paired": "paired_greedy_200",
    "memonly": "mem_only_200", "full": "full_200", "long": "long_200",
    "bhundred": "budget_100", "bfour": "budget_400", "beight": "budget_800",
    "bignet": "b28_200", "thirteen": "size13_200", "nine": "size9_200",
    "unihalf": "uni_100_vs_200", "unisame": "uni_200_vs_200", "uniroot": "uni_283_vs_200",
    "unidouble": "uni_400_vs_200", "pilot": "pilot2_stopper_200",
    "rulelcb": "rule_lcb_200", "vonegate": "v1gate_200", "vmcts": "vmcts_200", "dsmcts": "dsmcts_200",
    "seven": "double_200",
}


def numbers() -> None:
    """One macro per quoted match number. A run that has not finished prints as ??."""
    p = RES / "matches/table.json"
    rows = {r["run"]: r for r in json.loads(p.read_text())} if p.exists() else {}
    summaries = {q.parent.name: json.loads(q.read_text()) for q in (RES / "matches").glob("*/summary.json")}
    out = ["% Generated by experiments/paper_tables.py. Do not edit."]
    missing = "\\textbf{??}"

    def cmd(name: str, value: str) -> None:
        out.append("\\newcommand{\\" + name + "}{" + value + "}")

    for key, run in RUNS.items():
        r, s = rows.get(run), summaries.get(run)
        fields = {
            "Games": lambda: f"{r['games']:,}".replace(",", "{,}"),
            "Record": lambda: r["wld"].replace("-", "--"),
            "Score": lambda: f"{100 * r['score']:.1f}",
            "Elo": lambda: f"{r['elo']:+.0f}",
            "EloLo": lambda: f"{r['elo_lo']:+.0f}",
            "EloHi": lambda: f"{r['elo_hi']:+.0f}",
            "Visits": lambda: f"{r['visits']:.0f}",
            "Restart": lambda: f"{r['restart']:.0f}",
            "Evals": lambda: f"{r['rows']:.0f}",
            "BaseEvals": lambda: f"{r['base_rows']:.0f}",
            "HitRate": lambda: f"{100 * r['hit_rate']:.1f}",
            "HitLate": lambda: f"{100 * s['hit_rate_second_half']:.1f}",
            "Memory": lambda: f"{s['memory_entries']:,}".replace(",", "{,}"),
        }
        for suffix, fn in fields.items():
            try:
                value = fn() if r is not None else missing
            except (KeyError, TypeError):
                value = missing
            cmd(key + suffix, value)
    Path("paper/numbers.tex").write_text("\n".join(out) + "\n")
    done = sorted(k for k, run in RUNS.items() if run in rows)
    print("wrote paper/numbers.tex; finished runs:", ", ".join(done))


if __name__ == "__main__":
    numbers()
    uniform_table()
    frontier_table()
    rules_table()
    ablation_table()
    auc_table()
    matches_table()
    prediction_table()
    targets_table()
    compact_tables()
    baselines_table()
