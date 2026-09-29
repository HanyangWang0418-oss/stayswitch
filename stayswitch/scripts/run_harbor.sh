#!/usr/bin/env bash
# Run a Harbor job whose agent talks to the StaySwitch proxy.
# Usage: scripts/run_harbor.sh <dataset@version> <jobs-dir> [extra harbor args...]
# terminus-2 sends X-Session-ID on every call, which the router keys trajectories on.
# Network workarounds for this machine:
#  - verifiers download uv from GitHub Releases: served by scripts/gh_release_cache.py (started here if absent)
#  - agent setup apt-installs tmux in the container, which can be slow: setup/verifier timeouts x5
# The virtual model "stayswitch" is unknown to LiteLLM, so terminus-2 would assume a 1M context and never
# summarise; model_info gives it the pool's real budget (Tinker: 64K incl. output) so proactive
# summarisation fires ~8K below STAYSWITCH_MAX_INPUT_TOKENS.
set -euo pipefail
dataset="$1"; jobs="$2"; shift 2
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cache_port="${GH_CACHE_PORT:-8765}"
if ! curl -s -o /dev/null "http://127.0.0.1:$cache_port/"; then
  mkdir -p runs
  nohup python3 scripts/gh_release_cache.py "$cache_port" > runs/gh_cache.log 2>&1 &
  sleep 1
fi
export LITELLM_LOCAL_MODEL_COST_MAP=True  # the remote cost map on GitHub times out from here; calls are priced by stayswitch
exec uv run harbor run \
  --dataset "$dataset" \
  --agent "${STAYSWITCH_AGENT:-stayswitch.agents:StaySwitchTerminus}" \
  --model openai/stayswitch \
  --ak api_base=http://127.0.0.1:${STAYSWITCH_PORT:-4000} \
  --ak "model_info={\"max_input_tokens\": ${STAYSWITCH_MAX_INPUT_TOKENS:-56000}, \"max_output_tokens\": 8192, \"input_cost_per_token\": 0, \"output_cost_per_token\": 0}" \
  --jobs-dir "$jobs" \
  --verifier-timeout-multiplier "${VERIFIER_TIMEOUT_MULT:-5}" \
  --agent-setup-timeout-multiplier "${SETUP_TIMEOUT_MULT:-5}" \
  --ve "UV_INSTALLER_GITHUB_BASE_URL=http://host.docker.internal:$cache_port" \
  "$@"
