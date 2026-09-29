#!/usr/bin/env bash
# Always-strong and always-weak baselines on the same Terminal-Bench 2 tasks, run side by side.
# Usage: scripts/tb2_baseline.sh [task_list=configs/tb2_tasks.txt] [concurrency_per_model=2]
# Output: runs/tb2_{strong,weak}/calls.jsonl, jobs/tb2_{strong,weak}/, then a summary table.
set -euo pipefail
task_list="${1:-configs/tb2_tasks.txt}"
conc="${2:-2}"
dataset="${TB2_DATASET:-terminal-bench/terminal-bench-2}"
cd "$(dirname "$0")/.."
mkdir -p runs

colima status >/dev/null 2>&1 || colima start

tasks=($(grep -v '^#' "$task_list" | grep -v '^$'))
include=()
for t in "${tasks[@]}"; do include+=(-i "${dataset%%/*}/$t"); done  # registry names are org/task

# Fetch task files and build their images once, before two jobs race on the same cache.
uv run harbor datasets download "$dataset" --cache > runs/tb2_download.log 2>&1 \
  || echo "dataset download incomplete (see runs/tb2_download.log); continuing with cached tasks"
uv run python scripts/prebuild_images.py -j 3 --only "${tasks[@]}"

trap "pkill -f 'litellm --config litellm_config.yaml --port 400[12]' || true" EXIT

start_proxy() {  # config port
  scripts/start_proxy.sh "configs/$1.toml" "$2" > "runs/proxy_$1.log" 2>&1 &
  for _ in $(seq 1 90); do curl -s "localhost:$2/health/liveliness" >/dev/null && return 0; sleep 1; done
  echo "proxy $1 did not come up; see runs/proxy_$1.log" >&2
  exit 1
}

start_proxy tb2_strong 4001
start_proxy tb2_weak 4002

STAYSWITCH_PORT=4001 OPENAI_API_KEY=sk-dummy scripts/run_harbor.sh "$dataset" jobs/tb2_strong "${include[@]}" -n "$conc" \
  > runs/harbor_tb2_strong.log 2>&1 &
job_strong=$!
STAYSWITCH_PORT=4002 OPENAI_API_KEY=sk-dummy scripts/run_harbor.sh "$dataset" jobs/tb2_weak "${include[@]}" -n "$conc" \
  > runs/harbor_tb2_weak.log 2>&1 &
job_weak=$!
wait "$job_strong" || echo "strong job exited non-zero (see runs/harbor_tb2_strong.log)"
wait "$job_weak" || echo "weak job exited non-zero (see runs/harbor_tb2_weak.log)"

uv run python scripts/summarize_runs.py tb2_strong tb2_weak
