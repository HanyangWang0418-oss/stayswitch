"""Re-price runs with the CliffCompaction paper's "perfect-caching" cost model (arXiv 2609.26779, appendix A.1).

The paper ignores provider-reported cache fields and derives cache hits from prompt lengths alone, so that
identical prompt sequences cost the same on every provider. For one trajectory with prompt lengths P_1..P_N:
the first call is a cold start (all P_1 tokens uncached); for k > 1, if P_k >= P_{k-1} the previous prompt is
fully cached (P_{k-1} cached, the rest uncached); if P_k < P_{k-1} the drop is a compaction that invalidates
the cache, so all P_k tokens are uncached. Output is priced per call. Cached tokens use the model's cache-read
price and uncached ones its input price (no cache-write premium, as in the paper's Table 9).

Also reports the provider-metered cost, cache hit rates, mean peak context (used to match a ~45K budget) and
the number of prompt drops. Usage: uv run python scripts/perfect_cache_cost.py <run_id> [<run_id> ...]
A trajectory is one session; terminus-2 summariser calls (``-summarization-`` ids) belong to their parent and
Claude Code subagents (``~sub-`` ids) are their own trajectory, because a subagent has its own prompt.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from stayswitch.pricing import PriceTable

ROOT = Path(__file__).resolve().parent.parent
PRICES = PriceTable.load(ROOT / "configs" / "prices.toml")


def trajectory_key(session_id: str) -> str:
    return re.split(r"-summarization-", session_id, maxsplit=1)[0]


def perfect_cache_cost(calls: list[dict]) -> tuple[float, int, int, int]:
    """(cost, cached tokens, prompt tokens, prompt drops) of one trajectory's calls in step order."""
    cost, cached_total, prompt_total, drops, prev = 0.0, 0, 0, 0, None
    for r in calls:
        u = r["usage"]
        prompt = u["fresh_input"] + u["cache_read"] + u["cache_write"]
        if prev is None:
            cached = 0
        elif prompt >= prev:
            cached = prev
        else:
            cached, drops = 0, drops + 1
        price = PRICES[r["model"]]
        per_m = 1_000_000
        cost += ((prompt - cached) * price.input + cached * price.cache_read + u["output"] * price.output) / per_m
        cached_total += cached
        prompt_total += prompt
        prev = prompt
    return cost, cached_total, prompt_total, drops


def solved(run_id: str) -> tuple[int, int]:
    done = [json.loads(p.read_text()) for p in (ROOT / "jobs" / run_id).glob("*/*/result.json")]
    rewards = [(d.get("verifier_result") or {}).get("rewards", {}).get("reward") for d in done]
    scored = [r for r in rewards if r is not None]
    return sum(1 for r in scored if r >= 1), len(scored)


def report(run_id: str) -> dict | None:
    path = ROOT / "runs" / run_id / "calls.jsonl"
    if not path.exists():
        return None
    by_traj: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("ok") and not r.get("replayed") and r["model"] in PRICES:
            by_traj[trajectory_key(r["session_id"])].append(r)
    rows = []
    for calls in by_traj.values():
        calls.sort(key=lambda r: (r["step"], r["ts"]))
        cost, cached, prompt, drops = perfect_cache_cost(calls)
        prompts = [r["usage"]["fresh_input"] + r["usage"]["cache_read"] + r["usage"]["cache_write"] for r in calls]
        real_cached = sum(r["usage"]["cache_read"] for r in calls)
        rows.append({
            "n": len(calls), "perfect": cost, "real": sum(r["cost"] for r in calls), "cached": cached, "prompt": prompt,
            "real_cached": real_cached, "peak": max(prompts), "mean_prompt": prompt / len(calls), "drops": drops,
        })
    if not rows:
        return None
    ok, scored = solved(run_id)
    mean = lambda key: statistics.fmean(r[key] for r in rows)  # noqa: E731
    return {
        "run": run_id, "traj": len(rows), "solved": f"{ok}/{scored}", "calls": mean("n"),
        "perfect": mean("perfect"), "real": mean("real"),
        "hit_perfect": 100 * sum(r["cached"] for r in rows) / sum(r["prompt"] for r in rows),
        "hit_real": 100 * sum(r["real_cached"] for r in rows) / sum(r["prompt"] for r in rows),
        "peak": mean("peak"), "mean_prompt": mean("mean_prompt"), "drops": mean("drops"),
    }


def main() -> None:
    reports = [r for r in map(report, sys.argv[1:]) if r]
    print(f"{'run':24s} {'traj':>4s} {'solved':>7s} {'calls':>6s} {'$/traj':>8s} {'$/traj':>8s} {'hit%':>5s} {'hit%':>5s} {'peak':>7s} {'mean':>7s} {'drops':>5s}")
    print(f"{'':24s} {'':>4s} {'':>7s} {'':>6s} {'perfect':>8s} {'real':>8s} {'perf':>5s} {'real':>5s} {'ctx':>7s} {'ctx':>7s} {'/traj':>5s}")
    for r in reports:
        print(f"{r['run']:24s} {r['traj']:4d} {r['solved']:>7s} {r['calls']:6.1f} {r['perfect']:8.3f} {r['real']:8.3f} "
              f"{r['hit_perfect']:5.0f} {r['hit_real']:5.0f} {r['peak']:7.0f} {r['mean_prompt']:7.0f} {r['drops']:5.1f}")


if __name__ == "__main__":
    main()
