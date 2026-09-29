#!/usr/bin/env bash
# One routing config over a list of Harbor tasks: start its proxy, run the job, stop the proxy, summarise.
# Usage: scripts/run_model.sh <dataset> <task_list> <config> [port=4001] [concurrency=2]
#   e.g. scripts/run_model.sh swe-bench/swe-bench-verified configs/swe_tasks.txt configs/swe_strong.toml
# The run id (config [run].id) names both runs/<id>/ (call log) and jobs/<id>/ (Harbor trials).
set -euo pipefail
dataset="$1"; task_list="$2"; config="$3"; port="${4:-4001}"; conc="${5:-2}"
cd "$(dirname "$0")/.."
run_id="$(uv run python -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['run']['id'])" "$config")"
mkdir -p runs

colima status >/dev/null 2>&1 || colima start

include=()
for t in $(sed 's/#.*//' "$task_list" | awk 'NF {print $1}'); do include+=(-i "${dataset%%/*}/$t"); done

scripts/start_proxy.sh "$config" "$port" > "runs/proxy_$run_id.log" 2>&1 &
trap "pkill -f 'litellm --config litellm_config.yaml --port $port' || true" EXIT
for _ in $(seq 1 90); do curl -s "localhost:$port/health/liveliness" >/dev/null && break; sleep 1; done
curl -s "localhost:$port/health/liveliness" >/dev/null || { echo "proxy did not come up; see runs/proxy_$run_id.log" >&2; exit 1; }

STAYSWITCH_PORT="$port" OPENAI_API_KEY=sk-dummy scripts/run_harbor.sh "$dataset" "jobs/$run_id" "${include[@]}" -n "$conc" \
  > "runs/harbor_$run_id.log" 2>&1 || echo "harbor exited non-zero (see runs/harbor_$run_id.log)"

uv run python scripts/summarize_runs.py "$run_id"
