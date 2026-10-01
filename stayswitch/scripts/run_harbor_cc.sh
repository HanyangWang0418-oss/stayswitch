#!/usr/bin/env bash
# Run a Harbor job with Claude Code as the agent, talking to a StaySwitch (or CliffCompaction) proxy.
# Usage: scripts/run_harbor_cc.sh <dataset> <jobs-dir> [extra harbor args...]
#
# Claude Code runs inside the task container, so it reaches the proxy on the host via
# host.docker.internal. It requests the virtual model "stayswitch" for both its main and its
# small/fast model; the proxy routes and logs them. It identifies sessions with the
# x-claude-code-session-id header, which the proxy reads.
#   CC_BASE_URL      proxy URL as seen from the container (default http://host.docker.internal:$STAYSWITCH_PORT)
#   CC_COMPACT_PCT   Claude Code's own auto-compact trigger, percent of its assumed 200K window
#                    (default 25 = ~50K, under Tinker's 64K limit). Irrelevant when a compacting
#                    proxy keeps the prompt small. The variable Claude Code 2.1.286 reads is
#                    CLAUDE_AUTOCOMPACT_PCT_OVERRIDE (checked in the binary); CLAUDE_CODE_AUTOCOMPACT_PCT_OVERRIDE
#                    is ignored. DISABLE_AUTO_COMPACT=1 turns it off entirely.
set -euo pipefail
dataset="$1"; jobs="$2"; shift 2
cd "$(dirname "$0")/.."
cache_port="${GH_CACHE_PORT:-8765}"
if ! curl -s -o /dev/null "http://127.0.0.1:$cache_port/"; then
  mkdir -p runs
  nohup python3 scripts/gh_release_cache.py "$cache_port" > runs/gh_cache.log 2>&1 &
  sleep 1
fi
export LITELLM_LOCAL_MODEL_COST_MAP=True
exec uv run harbor run \
  --dataset "$dataset" \
  --agent "${STAYSWITCH_CC_AGENT:-stayswitch.agents:StaySwitchClaudeCode}" \
  --model stayswitch \
  --ae "ANTHROPIC_BASE_URL=${CC_BASE_URL:-http://host.docker.internal:${STAYSWITCH_PORT:-4000}}" \
  --ae "ANTHROPIC_API_KEY=sk-stayswitch-local" \
  --ae "ANTHROPIC_DEFAULT_HAIKU_MODEL=stayswitch" \
  --ae "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=${CC_COMPACT_PCT:-25}" \
  ${DISABLE_AUTO_COMPACT:+--ae DISABLE_AUTO_COMPACT=1} \
  --ae "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" \
  --jobs-dir "$jobs" \
  --verifier-timeout-multiplier "${VERIFIER_TIMEOUT_MULT:-5}" \
  --agent-setup-timeout-multiplier "${SETUP_TIMEOUT_MULT:-5}" \
  --ve "UV_INSTALLER_GITHUB_BASE_URL=http://host.docker.internal:$cache_port" \
  "$@"
