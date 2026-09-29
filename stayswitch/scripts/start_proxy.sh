#!/usr/bin/env bash
# Usage: scripts/start_proxy.sh configs/<router>.toml [port]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export STAYSWITCH_CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
port="${2:-4000}"
# Provider credentials live in the gitignored .env (GATEWAY_TOKEN, GATEWAY_BASE, GATEWAY_LIMIT_PRICE).
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; fi
export GATEWAY_AUTH_HEADER="Bearer ${GATEWAY_TOKEN:-}"
cd "$root/proxy"
export LITELLM_LOCAL_MODEL_COST_MAP=True  # the remote cost map on GitHub times out from here; calls are priced by stayswitch
exec uv run litellm --config litellm_config.yaml --port "$port"
