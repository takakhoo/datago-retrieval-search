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


RUNS = {
    "main": "main_200", "strict": "main_stopper_160", "lean": "main_140", "paired": "paired_greedy_200",
    "memonly": "mem_only_200", "full": "full_200", "long": "long_200",
    "bhundred": "budget_100", "bfour": "budget_400", "beight": "budget_800",
    "bignet": "b28_200", "thirteen": "size13_200", "nine": "size9_200",
    "unihalf": "uni_100_vs_200", "unisame": "uni_200_vs_200", "uniroot": "uni_283_vs_200",
    "unidouble": "uni_400_vs_200", "pilot": "pilot2_stopper_200",
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
