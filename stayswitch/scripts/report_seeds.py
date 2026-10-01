"""Aggregate the Claude Code compaction arms across seeds (docs/CLIFF_ALIGNMENT.md).

  uv run python scripts/report_seeds.py native=cc_mid,cc_mid_r2,cc_mid_r3 cliff45=cc_mid_cliff45,cc_mid_cliff45_r2,... [--json out]

The first arm is the baseline. Per arm: each seed's resolve rate, calls per task, cost per task (the paper's
perfect-cache model) and compactions per task, then the mean +- standard deviation across seeds, and the pooled
resolve rate over every scored trajectory with its binomial standard error. Against the baseline, over (task, seed)
pairs both arms finished: mean differences +- standard error across pairs, and e_hat (extra calls per compaction).
Runs are joined to trials exactly as in report_arms.py (latest trial per task, total-budget stops excluded).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_arms import load  # noqa: E402

PER_TASK_KEYS = ("calls", "perfect", "real", "mean_prompt", "peak", "drops")


def seed_metrics(tasks: dict[str, dict]) -> dict:
    scored = [t for t in tasks.values() if t["reward"] is not None]
    out = {"tasks": len(tasks), "scored": len(scored), "solved": sum(1 for t in scored if t["reward"] >= 1)}
    out["rate"] = 100 * out["solved"] / len(scored) if scored else float("nan")
    for k in PER_TASK_KEYS:
        out[k] = statistics.fmean(t[k] for t in tasks.values()) if tasks else float("nan")
    return out


def mean_sd(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan"))


def fmt(m: float, s: float, p: int = 1) -> str:
    return f"{m:.{p}f}" + (f"±{s:.{p}f}" if s == s else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=run_id[,run_id...]; the first is the baseline")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    arms: dict[str, list[dict[str, dict]]] = {}
    for spec in args.arms:
        name, ids = spec.split("=", 1)
        arms[name] = [load(r) for r in ids.split(",")]

    print(f"{'arm':10s} {'seeds':>5s} {'traj':>5s} {'solved%/seed':>22s} {'pooled%':>12s} {'calls/task':>12s} {'$/task':>14s} {'mean ctx':>12s} {'compact/task':>12s}")
    summary: dict[str, dict] = {}
    for name, seeds in arms.items():
        per = [seed_metrics(s) for s in seeds if s]
        if not per:
            print(f"{name:10s} (no data)")
            continue
        scored = sum(p["scored"] for p in per)
        solved = sum(p["solved"] for p in per)
        pooled = 100 * solved / scored if scored else float("nan")
        se = (pooled * (100 - pooled) / scored) ** 0.5 if scored else float("nan")
        row = {k: mean_sd([p[k] for p in per]) for k in ("rate", *PER_TASK_KEYS)}
        summary[name] = {"seeds": len(per), "per_seed": per, "pooled_rate": pooled, "pooled_se": se, **{k: v[0] for k, v in row.items()}}
        print(f"{name:10s} {len(per):5d} {sum(p['tasks'] for p in per):5d} {fmt(*row['rate']):>22s} {pooled:6.1f}±{se:<4.1f} "
              f"{fmt(*row['calls']):>12s} {fmt(*row['perfect'], 3):>14s} {fmt(row['mean_prompt'][0] / 1000, row['mean_prompt'][1] / 1000):>11s}K {fmt(*row['drops']):>12s}")
    print("   solved%/seed and the per-task columns are mean±sd across seeds; pooled% is over all scored trajectories (±binomial se).")

    base_name = next(iter(arms))
    base_pairs = {(i, task): t for i, seed in enumerate(arms[base_name]) for task, t in seed.items()}
    print(f"\nPaired against {base_name} over (task, seed) pairs (mean difference ± standard error across pairs):")
    print(f"{'arm':10s} {'pairs':>5s} {'d_solved%':>12s} {'d_calls':>12s} {'d_$/task':>14s} {'d_$/call%':>10s} {'e_hat':>6s}")
    for name, seeds in arms.items():
        if name == base_name:
            continue
        pairs = [(t, base_pairs[(i, task)]) for i, seed in enumerate(seeds) for task, t in seed.items() if (i, task) in base_pairs]
        if not pairs:
            continue
        ds = [100 * (((a["reward"] or 0) >= 1) - ((b["reward"] or 0) >= 1)) for a, b in pairs]
        dc = [a["calls"] - b["calls"] for a, b in pairs]
        dd = [a["perfect"] - b["perfect"] for a, b in pairs]
        pc = 100 * (sum(a["perfect"] for a, _ in pairs) / sum(a["calls"] for a, _ in pairs)
                    / (sum(b["perfect"] for _, b in pairs) / sum(b["calls"] for _, b in pairs)) - 1)
        drops = sum(a["drops"] for a, _ in pairs)
        se = lambda xs: statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")  # noqa: E731
        print(f"{name:10s} {len(pairs):5d} {statistics.fmean(ds):+7.1f}±{se(ds):<4.1f} {statistics.fmean(dc):+7.1f}±{se(dc):<4.1f} "
              f"{statistics.fmean(dd):+8.3f}±{se(dd):<5.3f} {pc:+10.1f} {sum(dc) / drops if drops else float('nan'):6.2f}")
        summary[name].update({"pairs": len(pairs), "d_rate": statistics.fmean(ds), "d_rate_se": se(ds), "d_calls": statistics.fmean(dc),
                              "d_cost": statistics.fmean(dd), "d_cost_se": se(dd), "d_cost_per_call_pct": pc})

    print("\nPer-task solves (solved/seeds run):")
    tasks = sorted({task for seeds in arms.values() for seed in seeds for task in seed})
    print(f"{'task':28s} " + " ".join(f"{n:>8s}" for n in arms))
    for task in tasks:
        cells = []
        for seeds in arms.values():
            runs = [s[task] for s in seeds if task in s]
            cells.append(f"{sum(1 for t in runs if (t['reward'] or 0) >= 1)}/{len(runs)}" if runs else ".")
        print(f"{task:28s} " + " ".join(f"{c:>8s}" for c in cells))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
