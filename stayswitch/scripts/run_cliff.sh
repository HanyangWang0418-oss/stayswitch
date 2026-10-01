#!/usr/bin/env bash
# Agent -> CliffCompaction proxy (compaction) -> StaySwitch LiteLLM proxy (routing, logging, cost) -> provider.
# Usage: scripts/run_cliff.sh <task_list> <config> fixed <threshold_tokens>
#        scripts/run_cliff.sh <task_list> <config> eoq <extra_steps>
# <config> is a StaySwitch router config (normally a fixed model, no [context]); its run id names
# runs/<id>/ and jobs/<id>/. The agent's own summariser is disabled: compaction is CliffCompaction's job.
set -euo pipefail
task_list="$1"; config="$2"; mode="$3"; arg="$4"
dataset="${DATASET:-swe-bench/swe-bench-verified}"
router_port="${ROUTER_PORT:-4001}"; cliff_port="${CLIFF_PORT:-8257}"; keep="${KEEP_RECENT:-3}"; conc="${CONC:-3}"
cd "$(dirname "$0")/.."
run_id="$(uv run python -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['run']['id'])" "$config")"
model="$(uv run python -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['policy'].get('model','strong'))" "$config")"
mkdir -p "runs/$run_id"
colima status >/dev/null 2>&1 || colima start

cleanup() {
  pkill -f "litellm --config litellm_config.yaml --port $router_port" || true
  pkill -f "serve --port $cliff_port" || true
}
trap cleanup EXIT

scripts/start_proxy.sh "$config" "$router_port" > "runs/proxy_$run_id.log" 2>&1 &
for _ in $(seq 1 90); do curl -s "localhost:$router_port/health/liveliness" >/dev/null && break; sleep 1; done

serve=(serve --port "$cliff_port" --keep-recent "$keep" --openai-upstream "http://127.0.0.1:$router_port")
if [ "$mode" = fixed ]; then
  uv run --group cliff cliff "${serve[@]}" --threshold "$arg" > "runs/$run_id/cliff.log" 2>&1 &
else
  uv run --group cliff python scripts/cliff_eoq.py --model "$model" --extra-steps "$arg" \
    --log "runs/$run_id/cliff_thresholds.jsonl" -- "${serve[@]}" > "runs/$run_id/cliff.log" 2>&1 &
fi
for _ in $(seq 1 60); do curl -s -o /dev/null "localhost:$cliff_port/" && break; sleep 1; done

include=()
for t in $(sed 's/#.*//' "$task_list" | awk 'NF {print $1}'); do include+=(-i "${dataset%%/*}/$t"); done
STAYSWITCH_API_BASE="http://127.0.0.1:$cliff_port/v1" STAYSWITCH_MAX_INPUT_TOKENS=1000000 OPENAI_API_KEY=sk-dummy \
  scripts/run_harbor.sh "$dataset" "jobs/$run_id" "${include[@]}" -n "$conc" > "runs/harbor_$run_id.log" 2>&1 \
  || echo "harbor exited non-zero (see runs/harbor_$run_id.log)"

uv run python scripts/summarize_runs.py "$run_id"
