#!/bin/sh
# Pull finished match runs from the lab machine over an existing SSH control socket.
# Usage: pull_runs.sh user@host:/path/to/runs/series
exec rsync -az --include '*/' --include 'summary.json' --include 'games.jsonl' --include 'memory.jsonl' --exclude '*' \
  -e "ssh -S $HOME/.ssh/cm-lisplab2" "$1/" runs/series/
