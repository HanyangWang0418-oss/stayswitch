"""Per-arm and paired-per-task report for the Claude Code compaction experiments (docs/CLIFF_ALIGNMENT.md).

  uv run python scripts/report_arms.py <baseline_run> <arm_run> [<arm_run> ...]

Joins each run's call log (runs/<id>/calls.jsonl) with Harbor's trial results (jobs/<id>/**/result.json): a trial's
Claude Code session id is the file name under agent/sessions/projects/*/, which is also the session id the proxy
logged. For every arm it reports resolve rate, calls per task, cost per task and per call (the paper's perfect-cache
model and the provider-metered one), cache hit rate, context size, and the number of prompt drops (compactions). Against
the baseline, over tasks both finished, it reports paired differences and e_hat, the extra calls per compaction:
    e_hat = sum(calls_arm - calls_base) / sum(compactions_arm).
When a task was rerun inside the same run id (an infrastructure failure, e.g. the agent OOM-killed), only its latest
trial counts, as in summarize_runs.py. Subagent calls (``~sub-`` ids) count toward cost but not toward the main
trajectory's calls or compactions. Baseline logs
written before the subagent split lack those ids; for a run without compaction their subagent calls are separated by
message-count continuity (the main conversation's message count only grows).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from perfect_cache_cost import PRICES, perfect_cache_cost

ROOT = Path(__file__).resolve().parent.parent


def trial_index(run_id: str) -> dict[str, tuple[str, float | None, str]]:
    """session id -> (task, reward, exception) for the latest finished trial of each task in a run."""
    latest: dict[str, tuple[str, Path, float | None, str]] = {}
    for result in (ROOT / "jobs" / run_id).glob("*/*/result.json"):
        trial = result.parent
        data = json.loads(result.read_text())
        task = trial.name.rsplit("__", 1)[0]
        started = data.get("started_at") or ""
        if task in latest and latest[task][0] >= started:
            continue
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = (data.get("exception_info") or {}).get("exception_type", "")
        latest[task] = (started, trial, reward, exc)
    out: dict[str, tuple[str, float | None, str]] = {}
    for task, (_, trial, reward, exc) in latest.items():
        for f in trial.glob("agent/sessions/projects/*/*.jsonl"):
            out[f.stem] = (task, reward, exc)
    return out


def split_legacy(calls: list[dict]) -> tuple[list[dict], list[dict]]:
    """Main vs subagent calls of one session logged before subagent ids existed (no compaction in the run)."""
    main, sub, last = [], [], 0
    for r in sorted(calls, key=lambda r: r["ts"]):
        if r["n_messages"] >= last + 1 or not main:
            main.append(r)
            last = r["n_messages"]
        else:
            sub.append(r)
    return main, sub


def load(run_id: str) -> dict[str, dict]:
    path = ROOT / "runs" / run_id / "calls.jsonl"
    if not path.exists():
        return {}
    index = trial_index(run_id)
    sessions: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("ok") and not r.get("replayed") and r["model"] in PRICES:
            sessions[r["session_id"]].append(r)
    has_sub = any("~sub-" in sid for sid in sessions)
    legacy = not has_sub and not any(k in run_id for k in ("cliff", "eoq"))
    tasks: dict[str, dict] = {}
    for sid, calls in sessions.items():
        if "~sub-" in sid or sid not in index:
            continue
        sub_groups = [cs for s2, cs in sessions.items() if s2.startswith(sid + "~sub-")]
        if legacy:
            calls, extra = split_legacy(calls)
            sub_groups = [extra] if extra else []
        subs = [c for g in sub_groups for c in g]
        task, reward, exc = index[sid]
        calls.sort(key=lambda r: (r["step"], r["ts"]))
        cost, cached, prompt, drops = perfect_cache_cost(calls)
        sub_cost = sum(perfect_cache_cost(sorted(g, key=lambda r: (r["step"], r["ts"])))[0] for g in sub_groups)
        prompts = [r["usage"]["fresh_input"] + r["usage"]["cache_read"] + r["usage"]["cache_write"] for r in calls]
        tasks[task] = {
            "reward": reward, "exc": exc, "calls": len(calls), "sub_calls": len(subs), "perfect": cost + sub_cost,
            "real": sum(r["cost"] for r in calls) + sum(r["cost"] for r in subs), "cached": cached, "prompt": prompt,
            "real_cached": sum(r["usage"]["cache_read"] for r in calls), "peak": max(prompts), "mean_prompt": prompt / len(calls),
            "drops": drops,
        }
    return tasks


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def stderr(xs: list[float]) -> float:
    return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")


def main() -> None:
    runs = sys.argv[1:]
    data = {r: load(r) for r in runs}
    print(f"{'arm':22s} {'tasks':>5s} {'solved':>7s} {'calls':>6s} {'$/task':>7s} {'$/task':>7s} {'$/call':>7s} {'hit%':>5s} {'peak':>6s} {'meanctx':>7s} {'drops':>5s}")
    print(f"{'':22s} {'':>5s} {'':>7s} {'':>6s} {'perfect':>7s} {'real':>7s} {'perfect':>7s} {'perf':>5s} {'':>6s} {'':>7s} {'/task':>5s}")
    for r in runs:
        t = list(data[r].values())
        if not t:
            print(f"{r:22s} (no finished trials)")
            continue
        scored = [x for x in t if x["reward"] is not None]
        ok = sum(1 for x in scored if x["reward"] >= 1)
        calls = sum(x["calls"] for x in t)
        print(f"{r:22s} {len(t):5d} {ok:3d}/{len(scored):<3d} {mean([x['calls'] for x in t]):6.1f} {mean([x['perfect'] for x in t]):7.3f} "
              f"{mean([x['real'] for x in t]):7.3f} {sum(x['perfect'] for x in t) / calls:7.4f} "
              f"{100 * sum(x['cached'] for x in t) / sum(x['prompt'] for x in t):5.0f} {mean([x['peak'] for x in t]):6.0f} "
              f"{mean([x['mean_prompt'] for x in t]):7.0f} {mean([x['drops'] for x in t]):5.1f}")
    base_id, base = runs[0], data[runs[0]]
    print(f"\nPaired against {base_id}, over tasks both finished (mean difference, +- standard error across tasks):")
    print(f"{'arm':22s} {'n':>3s} {'d_solved':>8s} {'d_calls':>14s} {'d_$/task':>16s} {'d_$/call %':>11s} {'e_hat':>6s}")
    for r in runs[1:]:
        common = sorted(set(base) & set(data[r]))
        if not common:
            continue
        a, b = [data[r][k] for k in common], [base[k] for k in common]
        dc = [x["calls"] - y["calls"] for x, y in zip(a, b)]
        dd = [x["perfect"] - y["perfect"] for x, y in zip(a, b)]
        per_call = 100 * (sum(x["perfect"] for x in a) / sum(x["calls"] for x in a)
                          / (sum(y["perfect"] for y in b) / sum(y["calls"] for y in b)) - 1)
        ds = sum((x["reward"] or 0) >= 1 for x in a) - sum((y["reward"] or 0) >= 1 for y in b)
        drops = sum(x["drops"] for x in a)
        e_hat = f"{sum(dc) / drops:6.2f}" if drops else "   n/a"
        print(f"{r:22s} {len(common):3d} {ds:+8d} {mean(dc):+8.1f}+-{stderr(dc):4.1f} {mean(dd):+9.3f}+-{stderr(dd):5.3f} {per_call:+11.1f} {e_hat}")
    print("\nPer-task reward (1 solved, 0 failed, . missing, x infrastructure error):")
    tasks = sorted(set().union(*[set(d) for d in data.values()]))
    print(f"{'task':28s} " + " ".join(f"{r[-9:]:>9s}" for r in runs))
    for k in tasks:
        cells = []
        for r in runs:
            x = data[r].get(k)
            cells.append("." if x is None else "x" if x["reward"] is None else str(int(x["reward"] >= 1)))
        print(f"{k:28s} " + " ".join(f"{c:>9s}" for c in cells))


if __name__ == "__main__":
    main()
