# StaySwitch

**使用手册(中文):[docs/MANUAL.md](docs/MANUAL.md)** — setup, datasets and images, running experiments, config reference, outputs, forking, troubleshooting.

Cache-aware step-level model routing for LLM agents. See `../proposal.md` for the research plan.

## Layout
| Path | What |
|---|---|
| `src/stayswitch/pricing.py` | Price table + usage split (fresh / cache_read / cache_write / output) |
| `src/stayswitch/cache.py` | `CacheSemantics` (provider cache rules: TTL, read/write multipliers, min prefix, cross-model) + `CacheLedger` (what each model has cached for a trajectory) |
| `src/stayswitch/costmodel.py` | Analytic cost of the next call, switch premium, return premium, break-even steps under given semantics |
| `src/stayswitch/session.py` | Per-trajectory state keyed by `X-Session-ID`; task key = hash of first user message |
| `src/stayswitch/policy.py` | `fixed`, `random_segment`, `fork` (replay prefix → option for H calls → base) |
| `src/stayswitch/accounting.py` | Re-price a trajectory under cache semantics (E0) |
| `proxy/stayswitch_hook.py` | LiteLLM pre-call hook (routes) + success/failure logger (JSONL call log) |
| `proxy/litellm_config.yaml` | Model pool: `strong`=Qwen3.5-397B-A17B, `weak`=Qwen3.5-9B on Tinker (OpenAI-compatible); `ds-*` DeepSeek V4 (company network); `mock-*` offline |
| `src/stayswitch/agents.py` | `StaySwitchTerminus`: terminus-2 that really sends `X-Session-ID` and points container apt at a mirror |
| `scripts/gh_release_cache.py` | Local retrying cache for GitHub release assets (verifiers' uv install) |
| `configs/*.toml` | Router configs; `prices.toml` must be checked against the provider price page |

Two environments because LiteLLM-proxy and Harbor pin incompatible `rich` versions:
root `.venv` = Harbor + our package; `proxy/.venv` = LiteLLM proxy + our package (editable).

## Setup
```bash
uv sync                 # Harbor env
(cd proxy && uv sync)   # proxy env
uv run pytest -q
```

## Run
```bash
# credentials: gitignored .env with TINKER_API_KEY (and GATEWAY_* for the ds pool); start_proxy.sh loads it
scripts/start_proxy.sh configs/fixed_strong.toml 4000
# another terminal; needs Docker running (colima start)
OPENAI_API_KEY=sk-dummy scripts/run_harbor.sh terminal-bench/terminal-bench-2 jobs/fixed_strong -n 4
```
Forks: point `configs/fork_example.toml` at the base run's `calls.jsonl`, restart the proxy with it, rerun the same tasks.
Every call lands in `runs/<run>/calls.jsonl` with session, step, model, switch flag, replay flag, token buckets and cost.

Offline smoke test (no keys): `configs/mock_fixed.toml` then `configs/mock_fork.toml` with `scripts/smoke_client.py`.

## Known gaps
- Network notes (this Mac): Docker Hub is blocked → colima uses registry mirrors (`~/.colima/default/colima.yaml`: docker.m.daocloud.io, mirror.gcr.io); GitHub Releases is slow → verifier timeout x5; the `ds-*` gateway is only reachable on its private network.
- `prices.toml`: Tinker rates are list prices (cached prefill 0.2x, no write premium); `ds-*` rates are placeholders.
- Qwen3.5 on Tinker ignores `enable_thinking: false`; every step carries reasoning tokens.
- Tinker's Cloudflare blocks some default User-Agents (error 1010); the pool sets one explicitly.
- Cache semantics observed on this service: automatic prefix cache per model, persistent across switches (switching back re-reads the old prefix), no cache-write premium. `reprice(..., semantics=TINKER)` (or `ttl_s=None`) models this.
- Replay divergence hashes the prompt with container hostnames normalised; other nondeterministic tool output (timestamps, PIDs) will still trip it.
- Terminus-2 context summarisation calls carry derived session ids (`<id>-summarization-*`) and are routed as separate sessions.

## Verified (2026-09-28)
`harbor/hello-world`: base run (weak) reward 1 in 57 s; fork (replay call 0, strong from call 1) reward 1, replayed prefix matched (no divergence), the switch shows a cold cache on the new model and a hit on the next call.
