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
    }[field]()


text = re.sub(r"\{\{(\w+)\.(\w+)\}\}", lambda m: value(m.group(1), m.group(2)), text)
table = Path("results/matches/TABLE.md").read_text()
table = "\n".join(l for l in table.splitlines() if "pilot" not in l.lower() or "Player" in l)
text = text.replace("{{table}}", table.strip())
demo = Path("results/demo/game.json")
excerpt = "(demo game pending)"
if demo.exists():
    out = subprocess.run([sys.executable, "-m", "datago.demo", "--replay", str(demo)],
                         capture_output=True, text=True).stdout.splitlines()
    moves = [l for l in out if l[:4].strip().isdigit()]
    k = next((i for i, l in enumerate(moves) if "800" in l), len(moves) // 2)
    excerpt = "\n".join([out[0]] + moves[:6] + ["   ..."] + moves[max(k - 3, 6): k + 5] + ["   ..."] + out[-3:])
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
Path("README.md").write_text(text)
print("wrote README.md;", text.count("pending"), "values pending")
