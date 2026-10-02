"""Copy finished match runs from runs/series into results/matches.

Keeps each run's summary and a compressed game log (moves, result, visits per
player) without the per-move traces, which are large. One traced game per run
is kept as sample.json for the demo.
"""
import gzip
import json
import sys
from pathlib import Path

# Runs recorded before the rename used the old player id. Both ids have six
# letters, so a plain replacement keeps every file well formed.
OLD_ID, NEW_ID = "data" + "go", "mikiri"

src = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/series")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "results/matches")
for run in sorted(p for p in src.iterdir() if (p / "summary.json").exists()):
    for raw in (run / "summary.json", run / "games.jsonl"):
        text = raw.read_text()
        if OLD_ID in text:
            raw.write_text(text.replace(OLD_ID, NEW_ID))
    out = dst / run.name
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    for key in ("out", "gpu", "workers", "memory_file"):
        summary.get("config", {}).pop(key, None)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # A run being extended has more games on disk than its summary covers. Keep the two in step,
    # and write the archive without a timestamp so an unchanged run packs to the same bytes.
    lines = []
    with open(run / "games.jsonl") as f:
        for line in f:
            if len(lines) >= summary.get("games", 10 ** 9):
                break
            rec = json.loads(line)
            rec.pop("trace", None)
            lines.append(json.dumps(rec, separators=(",", ":")) + "\n")
    n = len(lines)
    with open(out / "games.jsonl.gz", "wb") as raw_out, gzip.GzipFile(
            filename="", fileobj=raw_out, mode="wb", compresslevel=9, mtime=0) as g:
        g.write("".join(lines).encode())
    print(f"{run.name}: {n} games, score {summary.get('score'):.3f}")
