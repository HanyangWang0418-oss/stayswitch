"""E0: re-price recorded trajectories under different cache rules and model assignments.

Keeps every trajectory's token sequence as recorded and only changes which model
each call is billed to, then prices the result under several providers' cache
semantics. Reproduces the table in proposal section 11 with the ledger-based
``reprice`` (compactions and summaries now make every model cold) and prints the
old algorithm's numbers next to it.

Usage: python scripts/e0_reprice.py tb2_strong [more runs] [--strong strong] [--weak weak]
                                   [--segment 20] [--seeds 5] [--json out.json]
Reads runs/<run>/calls.jsonl; summarisation calls are folded into their parent trajectory.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from stayswitch.accounting import Bill, Call, reprice
from stayswitch.cache import ANTHROPIC, DEEPSEEK, OBLIVIOUS, TINKER, CacheLedger, CacheSemantics
from stayswitch.calllog import read_calls
from stayswitch.costmodel import CostModel
from stayswitch.pricing import PER_M, PriceTable

ROOT = Path(__file__).resolve().parent.parent
RULES = [TINKER, DEEPSEEK, ANTHROPIC]


def load_trajectories(run: str) -> dict[str, list[Call]]:
    """Successful calls of each parent trajectory, in time order."""
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for rec in read_calls(ROOT / "runs" / run / "calls.jsonl"):
        if rec.get("ok") and not rec.get("replayed"):
            by_parent[rec["session_id"].split("-summarization-", 1)[0]].append(rec)
    return {sid: [Call.from_record(r) for r in sorted(recs, key=lambda r: r["ts"])] for sid, recs in by_parent.items()}


# ---- model assignments: same tokens, different model per call ----

def all_of(model: str):
    return lambda calls, rng: [replace(c, model=model) for c in calls]


def random_step(weak: str, strong: str, p_weak: float):
    return lambda calls, rng: [replace(c, model=weak if rng.random() < p_weak else strong) for c in calls]


def random_segment(weak: str, strong: str, p_weak: float, length: int):
    def assign(calls, rng):
        out, model = [], strong
        for i, c in enumerate(calls):
            if i % length == 0:
                model = weak if rng.random() < p_weak else strong
            out.append(replace(c, model=model))
        return out
    return assign


def legacy_reprice(calls: list[Call], prices: PriceTable, ttl_s: float | None) -> Bill:
    """The pre-ledger algorithm: a model's prefix is its last prompt, never invalidated by a rewrite."""
    aware = oblivious = 0.0
    switches = cold = 0
    cached: dict[str, tuple[int, float | None]] = {}
    prev = None
    for c in calls:
        p = prices[c.model]
        prefix = 0
        if c.model in cached:
            length, last = cached[c.model]
            alive = ttl_s is None or c.ts is None or last is None or c.ts - last <= ttl_s
            prefix = min(length, c.prompt_tokens) if alive else 0
        cold += prefix == 0
        aware += (prefix * p.cache_read + (c.prompt_tokens - prefix) * p.cache_write + c.output_tokens * p.output) / PER_M
        oblivious += (c.prompt_tokens * p.input + c.output_tokens * p.output) / PER_M
        switches += prev is not None and prev != c.model
        cached[c.model] = (c.prompt_tokens, c.ts)
        prev = c.model
    return Bill(aware, oblivious, switches, cold)


def total(trajs: dict[str, list[Call]], assign, prices: PriceTable, sem: CacheSemantics, seed: int, *, legacy=False) -> Bill:
    rng = random.Random(seed)
    bills = []
    for calls in trajs.values():
        assigned = assign(calls, rng)
        if legacy:
            bills.append(legacy_reprice(assigned, prices.__class__({m: sem.price(prices[m]) for m in prices.names()}), sem.ttl_s))
        else:
            bills.append(reprice(assigned, prices, sem))
    return Bill(
        sum(b.cache_aware for b in bills), sum(b.cache_oblivious for b in bills), sum(b.switches for b in bills),
        sum(b.cold_calls for b in bills), sum(b.read_cost for b in bills), sum(b.resets for b in bills),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--strong", default="strong")
    ap.add_argument("--weak", default="weak")
    ap.add_argument("--segment", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--prices", default=str(ROOT / "configs" / "prices.toml"))
    ap.add_argument("--json")
    args = ap.parse_args()

    prices = PriceTable.load(args.prices)
    trajs: dict[str, list[Call]] = {}
    for run in args.runs:
        trajs.update({f"{run}/{sid}": calls for sid, calls in load_trajectories(run).items()})
    n_calls = sum(len(c) for c in trajs.values())
    print(f"{len(trajs)} trajectories, {n_calls} calls (summaries folded in)")

    # 1. The recorded run as it was: how much of the bill is cache, how far off cache-oblivious counting is.
    base = all_of(args.strong)
    print(f"\n## all-{args.strong}: the recorded trajectories under each cache rule")
    print(f"{'rule':12s} {'aware $':>9s} {'oblivious $':>11s} {'obliv/aware':>11s} {'cache-read share':>16s} {'cold':>5s} {'resets':>6s} {'legacy aware $':>14s}")
    for sem in RULES:
        b = total(trajs, base, prices, sem, 0)
        old = total(trajs, base, prices, sem, 0, legacy=True)
        print(f"{sem.name:12s} {b.cache_aware:9.2f} {b.cache_oblivious:11.2f} {b.cache_oblivious / b.cache_aware:11.2f}x"
              f" {b.read_cost / b.cache_aware:16.0%} {b.cold_calls:5d} {b.resets:6d} {old.cache_aware:14.2f}")

    # 2. Routing policies on fixed tokens: saving vs all-strong, counted without and with caching.
    policies = {
        f"step 20% {args.weak}": random_step(args.weak, args.strong, 0.2),
        f"step 50% {args.weak}": random_step(args.weak, args.strong, 0.5),
        f"seg{args.segment} 50% {args.weak}": random_segment(args.weak, args.strong, 0.5, args.segment),
        f"all-{args.weak} (same tokens)": all_of(args.weak),
    }
    results = {}
    print(f"\n## saving vs all-{args.strong}: oblivious -> cache-aware (mean over {args.seeds} seeds; legacy algorithm in brackets)")
    print(f"{'rule':12s}" + "".join(f" | {name:>28s}" for name in policies))
    for sem in RULES:
        ref = total(trajs, base, prices, sem, 0)
        ref_old = total(trajs, base, prices, sem, 0, legacy=True)
        cells = []
        for name, assign in policies.items():
            obl, aware, aware_old, sw, cold = [], [], [], [], []
            for seed in range(args.seeds):
                b = total(trajs, assign, prices, sem, seed)
                o = total(trajs, assign, prices, sem, seed, legacy=True)
                obl.append(1 - b.cache_oblivious / ref.cache_oblivious)
                aware.append(1 - b.cache_aware / ref.cache_aware)
                aware_old.append(1 - o.cache_aware / ref_old.cache_aware)
                sw.append(b.switches); cold.append(b.cold_calls)
            m = {k: statistics.mean(v) for k, v in dict(obl=obl, aware=aware, aware_old=aware_old, switches=sw, cold=cold).items()}
            results[(sem.name, name)] = m
            cells.append(f"{m['obl']:5.1%} -> {m['aware']:5.1%} [{m['aware_old']:5.1%}]")
        print(f"{sem.name:12s}" + "".join(f" | {c:>28s}" for c in cells))
    print("\nswitches / cold calls per policy (tinker rule):")
    for name in policies:
        m = results[("tinker", name)]
        print(f"  {name:28s} switches {m['switches']:7.1f}  cold {m['cold']:7.1f}")

    # 3. Break-even steps at the median context of these trajectories, per rule.
    ctx = statistics.median(c.prompt_tokens for calls in trajs.values() for c in calls)
    new = statistics.median(max(b.prompt_tokens - a.prompt_tokens, 0) for calls in trajs.values() for a, b in zip(calls, calls[1:])) or 800
    out = statistics.median(c.output_tokens for calls in trajs.values() for c in calls)
    print(f"\n## break-even steps for {args.strong} -> {args.weak} at median ctx {ctx:,.0f}, +{new:,.0f} new, {out:,.0f} out per step")
    print(f"{'rule':12s} {'permanent':>10s} {'dip, back in TTL':>17s} {'dip, back after TTL':>20s}")
    for sem in RULES + [OBLIVIOUS]:
        cm = CostModel(prices, sem)
        kw = dict(ctx_tokens=int(ctx), new_tokens=int(new), output_tokens=int(out))
        f = lambda v: "inf" if v == float("inf") else f"{v:.1f}"
        print(f"{sem.name:12s} {f(cm.breakeven_steps(args.strong, args.weak, permanent=True, **kw)):>10s}"
              f" {f(cm.breakeven_steps(args.strong, args.weak, gap_s=0, **kw)):>17s}"
              f" {f(cm.breakeven_steps(args.strong, args.weak, gap_s=10**9, **kw)):>20s}")

    if args.json:
        Path(args.json).write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in results.items()}, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
