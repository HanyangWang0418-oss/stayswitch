"""Re-price the Claude Code arms under different cache price sheets, from their logged prompt sequences.

  uv run python scripts/price_regimes.py native=cc_mid,cc_mid_r2 cliff45=... eoq1=...

Billing follows the CliffCompaction paper's prompt-length rule (perfect caching from prompt lengths, appendix A.1)
extended with the two knobs that differ between providers: a cache-write premium on the uncached part of a growing
prompt, and a cache TTL after which the whole prompt is re-written (timestamps from the call log). Cost per task is
reported for each regime, per arm, plus the EOQ trigger the price sheet implies for the observed growth and
post-compaction size. Prices are relative to the mid model's input price so regimes differ only in cache semantics.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_arms import load, trial_index  # noqa: E402
from stayswitch.eoq_trigger import eoq_l_star  # noqa: E402
from stayswitch.pricing import PER_M, Price, PriceTable  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MID = PriceTable.load(ROOT / "configs" / "prices.toml")["mid"]
# name -> (cache-read multiplier, cache-write multiplier, TTL seconds or None)
REGIMES = {
    "tinker 0.2x read, no write premium, persistent": (0.2, 1.0, None),
    "anthropic-like 0.1x read, 1.25x write, 5-min TTL": (0.1, 1.25, 300.0),
    "deepseek-like 0.1x read, no premium, persistent": (0.1, 1.0, None),
    "openai-like 0.5x read, no premium, 5-min TTL": (0.5, 1.0, 300.0),
}


def regime_price(read_mult: float, write_mult: float) -> Price:
    return Price(input=MID.input, output=MID.output, cache_read=MID.input * read_mult, cache_write=MID.input * write_mult)


def bill(calls: list[dict], price: Price, ttl: float | None) -> float:
    cost, prev, prev_ts = 0.0, None, None
    for r in calls:
        u = r["usage"]
        prompt = u["fresh_input"] + u["cache_read"] + u["cache_write"]
        expired = ttl is not None and prev_ts is not None and r["ts"] - prev_ts > ttl
        if prev is None or prompt < prev or expired:
            cached, written = 0, prompt
        else:
            cached, written = prev, prompt - prev
        cost += (cached * price.cache_read + written * price.cache_write + u["output"] * price.output) / PER_M
        prev, prev_ts = prompt, r["ts"]
    return cost


def arm_calls(run_id: str) -> list[list[dict]]:
    """Main-trajectory call lists of a run's latest trials, in step order (subagents priced separately below)."""
    import json
    from collections import defaultdict

    index = trial_index(run_id)
    sessions: dict[str, list[dict]] = defaultdict(list)
    for line in (ROOT / "runs" / run_id / "calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("ok") and not r.get("replayed") and r["model"] == "mid" and r.get("note") != "budget_stop":
            sessions[r["session_id"]].append(r)
    out = []
    for sid, calls in sessions.items():
        root = sid.split("~sub-")[0]
        if root in index:
            out.append(sorted(calls, key=lambda r: (r["step"], r["ts"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()
    arms = {spec.split("=")[0]: spec.split("=", 1)[1].split(",") for spec in args.arms}
    per_arm = {name: [c for rid in ids if (ROOT / "runs" / rid / "calls.jsonl").exists() for c in arm_calls(rid)] for name, ids in arms.items()}
    n_tasks = {name: sum(1 for rid in ids if (ROOT / "jobs" / rid).exists() for _ in trial_index(rid)) for name, ids in arms.items()}
    print(f"{'regime':52s} " + " ".join(f"{n:>9s}" for n in arms) + "   EOQ L* (K)")
    for regime, (rm, wm, ttl) in REGIMES.items():
        price = regime_price(rm, wm)
        cells = []
        for name, trajs in per_arm.items():
            total = sum(bill(c, price, ttl) for c in trajs)
            cells.append(f"{total / max(n_tasks[name], 1):9.3f}")
        # Implied trigger for a typical Claude Code trajectory on this pool: head ~18K, post-compaction ~21K, 1.25K/call.
        star = eoq_l_star(price, l0=21000, head=18000, growth=1250, extra_steps=2.3, ceiling=10**9) / 1000
        print(f"{regime:52s} " + " ".join(cells) + f"   {star:6.1f}")
    print("\ncost per task (USD) under each regime; the last column is the compaction trigger the EOQ rule derives from that price sheet.")
    base = next(iter(per_arm))
    print(f"\nRelative to {base}:")
    for regime, (rm, wm, ttl) in REGIMES.items():
        price = regime_price(rm, wm)
        b = sum(bill(c, price, ttl) for c in per_arm[base]) / max(n_tasks[base], 1)
        rel = [f"{100 * (sum(bill(c, price, ttl) for c in trajs) / max(n_tasks[name], 1) / b - 1):+8.1f}%" for name, trajs in per_arm.items()]
        print(f"{regime:52s} " + " ".join(rel))


if __name__ == "__main__":
    main()
