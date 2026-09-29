"""Summarise baseline runs: join Harbor trial results with the proxy call log.

Usage: python scripts/summarize_runs.py RUN [RUN ...]
Reads runs/<RUN>/calls.jsonl and jobs/<RUN>/*/<trial>/, prints per-run and
per-task tables, and writes runs/summary_<RUN+...>.json.

Trials and trajectories are joined by task key (normalised hash of the first
user message), so each task must appear once per run. Terminus-2 context
summarisation calls carry derived session ids (``<sid>-summarization-...``)
and are attributed to their parent trajectory.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from stayswitch.calllog import read_calls
from stayswitch.session import task_key

ROOT = Path(__file__).resolve().parent.parent


def parent_session(sid: str) -> str:
    return sid.split("-summarization-", 1)[0]


def _epoch(ts: str | None) -> float | None:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else None


def load_trials(run: str) -> list[dict]:
    """Trials of every job under jobs/<run>; for a task run more than once (a rerun), the latest wins."""
    latest: dict[str, dict] = {}
    for result_path in sorted((ROOT / "jobs" / run).glob("*/*/result.json")):
        trial_dir = result_path.parent
        result = json.loads(result_path.read_text())
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        exc = result.get("exception_info") or {}
        key = None
        traj_path = trial_dir / "agent" / "trajectory.json"
        if traj_path.exists():
            steps = json.loads(traj_path.read_text()).get("steps") or []
            if steps and steps[0].get("message"):
                key = task_key([{"role": "user", "content": steps[0]["message"]}])
        agent_exec = result.get("agent_execution") or {}
        trial = {
            "task": result.get("task_name", trial_dir.name).split("/")[-1],
            "trial": trial_dir.name,
            "reward": rewards.get("reward"),
            "exception": exc.get("exception_type"),
            "task_key": key,
            "window": (_epoch(agent_exec.get("started_at")), _epoch(agent_exec.get("finished_at"))),
            "started": result.get("started_at") or "",
        }
        prev = latest.get(trial["task"])
        if prev is None or trial["started"] > prev["started"]:
            latest[trial["task"]] = trial
    return sorted(latest.values(), key=lambda t: t["task"])


def _in_window(first_ts: float, window: tuple[float | None, float | None], slack_s: float = 120.0) -> bool:
    start, end = window
    if start is None:
        return True
    return start - slack_s <= first_ts <= (end or float("inf")) + slack_s


def session_stats(calls: list[dict]) -> dict:
    ok = [c for c in calls if c.get("ok")]
    main = [c for c in ok if "-summarization-" not in c["session_id"]]
    prompt = [sum(c["usage"][k] for k in ("fresh_input", "cache_read", "cache_write")) for c in ok]
    cache_read = sum(c["usage"]["cache_read"] for c in ok)
    return {
        "calls": len(main),
        "summ_calls": len(ok) - len(main),
        "errors": len(calls) - len(ok),
        "max_ctx": max(prompt, default=0),
        "ctx_series": [sum(c["usage"][k] for k in ("fresh_input", "cache_read", "cache_write")) for c in main],
        "output_tokens": sum(c["usage"]["output"] for c in ok),
        "cache_hit": cache_read / sum(prompt) if sum(prompt) else 0.0,
        "cost": sum(c.get("cost") or 0.0 for c in ok),
    }


def summarise(run: str) -> dict:
    log = ROOT / "runs" / run / "calls.jsonl"
    calls = read_calls(log) if log.exists() else []
    by_parent: dict[str, list[dict]] = defaultdict(list)
    task_of: dict[str, str] = {}
    for c in calls:
        p = parent_session(c["session_id"])
        by_parent[p].append(c)
        if c["session_id"] == p:
            task_of.setdefault(p, c["task"])
    sessions_by_task: dict[str, list[str]] = defaultdict(list)
    for p, t in task_of.items():
        sessions_by_task[t].append(p)

    first_ts = {p: min(c["ts"] for c in cs) for p, cs in by_parent.items()}
    rows = []
    for trial in load_trials(run):
        sids = sessions_by_task.get(trial["task_key"], []) if trial["task_key"] else []
        # A rerun leaves several sessions per task; keep the ones that ran inside this trial's agent window.
        window = trial.pop("window")
        trial.pop("started")
        if len(sids) > 1:
            sids = [s for s in sids if _in_window(first_ts[s], window)]
        stats = session_stats([c for s in sids for c in by_parent[s]]) if sids else None
        rows.append({**trial, "sessions": len(sids), **(stats or {})})
    return {"run": run, "trials": rows}


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def print_run(s: dict) -> None:
    rows = s["trials"]
    valid = [r for r in rows if r["reward"] is not None]
    solved = [r for r in valid if r["reward"] >= 1.0]
    with_calls = [r for r in rows if r.get("calls")]
    ctx = [r["max_ctx"] for r in with_calls]
    print(f"\n## {s['run']}")
    print(f"trials {len(rows)} | scored {len(valid)} | solved {len(solved)}"
          f" ({len(solved) / len(valid):.0%} of scored)" if valid else f"trials {len(rows)} | scored 0")
    if with_calls:
        print(
            f"calls/traj mean {statistics.mean(r['calls'] for r in with_calls):.1f}"
            f" | max ctx p50 {pct(ctx, .5):,.0f} p90 {pct(ctx, .9):,.0f}"
            f" | cache hit {statistics.mean(r['cache_hit'] for r in with_calls):.0%}"
            f" | cost/traj ${statistics.mean(r['cost'] for r in with_calls):.3f}"
            f" | total ${sum(r['cost'] for r in with_calls):.2f}"
        )
    exc = defaultdict(int)
    for r in rows:
        if r["exception"]:
            exc[r["exception"]] += 1
    if exc:
        print("exceptions:", dict(exc))


def print_tasks(summaries: list[dict]) -> None:
    runs = [s["run"] for s in summaries]
    tasks = sorted({r["task"] for s in summaries for r in s["trials"]})
    idx = {(s["run"], r["task"]): r for s in summaries for r in s["trials"]}
    head = f"{'task':30s}" + "".join(f" | {run:>28s}" for run in runs)
    print("\n" + head + "\n" + "-" * len(head))
    for t in tasks:
        cells = []
        for run in runs:
            r = idx.get((run, t))
            if r is None:
                cells.append(f"{'-':>28s}")
                continue
            rew = "exc" if r["reward"] is None else f"{r['reward']:.0f}"
            cells.append(f"{rew:>3s} {r.get('calls', 0):3d}st {r.get('max_ctx', 0) / 1000:5.1f}k ${r.get('cost', 0):6.3f}")
        print(f"{t:30s}" + "".join(f" | {c:>28s}" for c in cells))


def main() -> None:
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    summaries = [summarise(r) for r in runs]
    for s in summaries:
        print_run(s)
    print_tasks(summaries)
    out = ROOT / "runs" / f"summary_{'+'.join(runs)}.json"
    out.write_text(json.dumps(summaries, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
