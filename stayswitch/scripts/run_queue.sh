#!/usr/bin/env bash
# Run several routing configs one after another on the same tasks (one Harbor job at a time).
# Usage: scripts/run_queue.sh <dataset> <task_list> <config> [<config> ...]
# Waits for any Harbor job already running, reads each config only when its turn comes
# (so later configs can be tuned from earlier results), and skips configs marked TODO.
set -uo pipefail
dataset="$1"; task_list="$2"; shift 2
cd "$(dirname "$0")/.."
while pgrep -f "harbor run --dataset" >/dev/null; do sleep 60; done
for config in "$@"; do
  if grep -q "TODO" "$config"; then
    echo "$(date '+%F %T') skip $config (still has TODO)"
    continue
  fi
  echo "$(date '+%F %T') start $config"
  scripts/run_model.sh "$dataset" "$task_list" "$config" 4001 3 | tail -30
  echo "$(date '+%F %T') done $config"
done
