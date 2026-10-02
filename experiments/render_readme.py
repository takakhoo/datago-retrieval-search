"""Fill README.template.md from results/. Numbers for unfinished runs print as 'pending'."""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "experiments")
from paper_tables import RUNS  # noqa: E402

rows = {r["run"]: r for r in json.loads(Path("results/matches/table.json").read_text())}
text = Path("experiments/README.template.md").read_text()


def value(key: str, field: str) -> str:
    r = rows.get(RUNS[key])
    if r is None:
        return "pending"
    return {
        "games": lambda: f"{r['games']:,}", "score": lambda: f"{100 * r['score']:.1f}",
        "elo": lambda: f"{r['elo']:+.0f}", "elo_lo": lambda: f"{r['elo_lo']:+.0f}",
        "elo_hi": lambda: f"{r['elo_hi']:+.0f}", "visits": lambda: f"{r['visits']:.0f}",
        "elo_abs": lambda: f"{abs(r['elo']):.0f}", "wld": lambda: r["wld"],
        "evals": lambda: f"{r['rows']:.0f}", "base_evals": lambda: f"{r['base_rows']:.0f}",
        "hit": lambda: f"{100 * r['hit_rate']:.0f}",
    }[field]()


text = re.sub(r"\{\{(\w+)\.(\w+)\}\}", lambda m: value(m.group(1), m.group(2)), text)
table = Path("results/matches/TABLE.md").read_text()
table = "\n".join(l for l in table.splitlines() if "pilot" not in l.lower())
text = text.replace("{{table}}", table.strip())
demo = Path("results/demo/game.json")
excerpt = "(demo game pending)"
if demo.exists():
    out = subprocess.run([sys.executable, "-m", "mikiri.demo", "--replay", str(demo)],
                         capture_output=True, text=True).stdout.splitlines()
    moves = [l for l in out if l[:4].strip().isdigit()]
    k = next((i for i, l in enumerate(moves) if "3200" in l),
             next((i for i, l in enumerate(moves) if "800" in l), len(moves) // 2))
    excerpt = "\n".join([out[0]] + moves[:6] + ["   ..."] + moves[max(k - 4, 6): k + 5] + ["   ..."] + out[-3:])
text = text.replace("{{demo_excerpt}}", excerpt)
train = Path("results/stopper/training.json")
sentence = "(training curve pending)"
if train.exists():
    dc = json.loads(train.read_text())["data_curve"]
    first, last = dc[0], dc[-1]
    sentence = (f"with only {first['positions']:,} labeled positions the stopper already reaches "
                f"{first['multiplier']:.2f}x, and with {last['positions']:,} it reaches {last['multiplier']:.2f}x. "
                "The band shows the spread over three random draws of training games.")
text = text.replace("{{training_sentence}}", sentence)
pending_runs = [k for k in ("memonly", "full", "long", "bhundred", "bfour", "beight", "bignet", "thirteen", "nine", "lean")
                if RUNS[k] not in rows]
note = ("*More runs are still in progress and will be added to this table: head-to-head matches for the two strongest "
        "prior rules, other budgets, the b28 network, and smaller boards.*\n\n" if pending_runs else "")
text = text.replace("{{progress_note}}", note)
total = sum(r["games"] for name, r in rows.items() if not name.startswith("pilot"))
text = text.replace("{{total_games}}", f"{total:,}")
base = Path("results/baselines.json")
if base.exists():
    b = json.loads(base.read_text())
    d = b["ladders"]["doubling"]
    pr = {x["cost"]: x for x in sum(b["point_rules"].values(), [])}

    def cell(key, budget="200"):
        v = d.get(key, {}).get(budget)
        return f"{v['multiplier']:.2f}x ({v['lo']:.2f} to {v['hi']:.2f})" if v else "n/a"

    def point(rule, lo=150, hi=260):
        pts = [x for x in b["point_rules"][rule] if lo <= x["cost"] <= hi]
        return f"{pts[0]['multiplier']:.2f}x at {pts[0]['cost']:.0f} visits" if pts else "n/a"

    lines = ["| Rule | Source | Multiplier at 200 visits |", "|---|---|---|",
             f"| Value-of-information stop | Hay et al. 2012 | {cell('VOI stop | flat')} |",
             f"| Best-arm-identification stop | Kaufmann and Koolen 2017 | {cell('BAI stop | flat')} |",
             f"| Think longer when behind | Huang et al. 2010 | {point('BEHIND (v=0.6)')} |",
             f"| Stop when the runner-up cannot catch up | Baier and Winands 2016 | {point('STOP (p=0.45)', 120, 270)} |",
             f"| Smart pruning | Leela Chess Zero | {point('smart pruning (factor 1.33)')} |",
             f"| Extend when the top two are close | Baier and Winands 2016 | {point('CLOSE (d=0.4)')} |",
             f"| Extend when most-visited is not best-valued | Huang et al. 2010 | {point('UNST')} |",
             f"| KL-gain stop | Leela Chess Zero | {cell('KLD gain | flat')} |",
             f"| Budget chosen before search | in the spirit of Muppidi et al. 2026 | {cell('state only | flat')} |",
             f"| Learned move-stability classifier | DS-MCTS, Lan et al. 2021 | {cell('DS-MCTS | flat')} |",
             f"| Virtual expansion, published settings | V-MCTS, Ye et al. 2022 | {cell('V-MCTS (r=0.2, eps=0.1) | as published')} |",
             f"| **Mikiri** | this work | **{cell('Mikiri | rate')}** |"]
    text = text.replace("{{baselines_table}}", "\n".join(lines))
    pv = b["paired"]["Mikiri minus V-MCTS published at 200"]
    text = text.replace("{{paired_vmcts}}", f"{pv['difference']:.2f} ({pv['lo']:.2f} to {pv['hi']:.2f})")
    text = text.replace("{{ds_flat}}", f"{d['DS-MCTS | flat']['200']['multiplier']:.2f}")
    text = text.replace("{{ds_rate}}", f"{d['DS-MCTS | rate']['200']['multiplier']:.2f}")
Path("README.md").write_text(text)
print("wrote README.md;", text.count("pending"), "values pending")
