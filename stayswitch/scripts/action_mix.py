"""Mechanism check: what the agent spends its calls on under each compaction arm (Claude Code call logs).

  uv run python scripts/action_mix.py native=cc_mid,cc_mid_r2,cc_mid_r3 cliff40=... eoq1=...

Per arm, over the main trajectories of the latest trial per task: calls per task split by step kind (explore / edit /
test / other / no-tool), re-reads per task (a Read/Grep/Bash-cat of a path the trajectory already read), the share of
trajectories that loop (same tool call twice in a row at least 3 times), and the share ending with no tool call in the
last 3 steps (the agent wrote a final answer). Steps are classified from the recorded tool calls with stayswitch.signals.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_arms import trial_index  # noqa: E402
from stayswitch.signals import step_kind  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
READ_TOOLS = {"Read", "Grep", "Glob"}
_CAT = re.compile(r"\b(cat|head|tail|sed -n|less)\s+(\S+)")


def tool_calls(r: dict) -> list[tuple[str, dict]]:
    out = []
    for c in r.get("tool_calls") or []:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append((fn.get("name", ""), args if isinstance(args, dict) else {}))
    return out


def keys_of(calls: list[tuple[str, dict]]) -> str:
    parts = []
    for name, args in calls:
        if name == "Bash":
            parts.append(str(args.get("command", "")))
        elif name in ("Write", "Edit", "MultiEdit"):
            parts.append(f"str_replace {args.get('file_path', '')}")
        elif name in READ_TOOLS:
            parts.append(f"cat {args.get('file_path') or args.get('pattern') or ''}")
    return " ; ".join(parts)


def read_targets(calls: list[tuple[str, dict]]) -> set[str]:
    t = set()
    for name, args in calls:
        if name == "Read":
            t.add(str(args.get("file_path", "")))
        elif name == "Bash":
            t.update(m.group(2) for m in _CAT.finditer(str(args.get("command", ""))))
    return t


def trajectory_stats(calls: list[dict]) -> dict:
    kinds, seen, rereads, repeats, last_sigs = Counter(), set(), 0, 0, []
    for r in calls:
        tc = tool_calls(r)
        if not tc:
            kinds["no-tool"] += 1
            last_sigs.append("")
            continue
        kinds[step_kind(keys_of(tc))] += 1
        targets = read_targets(tc)
        rereads += len(targets & seen)
        seen |= targets
        sig = json.dumps(tc, sort_keys=True)
        if last_sigs and sig == last_sigs[-1]:
            repeats += 1
        last_sigs.append(sig)
    return {"n": len(calls), "kinds": kinds, "rereads": rereads, "loop": repeats >= 3, "final_answer": all(s == "" for s in last_sigs[-1:])}


def load(run_id: str) -> list[dict]:
    index = trial_index(run_id)
    by: dict[str, list[dict]] = defaultdict(list)
    for line in (ROOT / "runs" / run_id / "calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("ok") and "~sub-" not in r["session_id"] and r["session_id"] in index:
            by[r["session_id"]].append(r)
    return [trajectory_stats(sorted(c, key=lambda r: (r["step"], r["ts"]))) for c in by.values()]


def main() -> None:
    arms = {a.split("=")[0]: a.split("=", 1)[1].split(",") for a in sys.argv[1:]}
    print(f"{'arm':10s} {'traj':>5s} {'calls':>6s} {'explore':>8s} {'edit':>6s} {'test':>6s} {'other':>6s} {'no-tool':>8s} {'rereads':>8s} {'loop%':>6s} {'final%':>7s}")
    for name, ids in arms.items():
        t = [x for rid in ids if (ROOT / "runs" / rid / "calls.jsonl").exists() for x in load(rid)]
        if not t:
            continue
        m = lambda k: statistics.fmean(x["kinds"][k] for x in t)  # noqa: E731
        print(f"{name:10s} {len(t):5d} {statistics.fmean(x['n'] for x in t):6.1f} {m('explore'):8.1f} {m('edit'):6.1f} {m('test'):6.1f} "
              f"{m('other'):6.1f} {m('no-tool'):8.1f} {statistics.fmean(x['rereads'] for x in t):8.1f} "
              f"{100 * statistics.fmean(x['loop'] for x in t):6.0f} {100 * statistics.fmean(x['final_answer'] for x in t):7.0f}")
    print("calls per task by kind (mean over trajectories); rereads = reads of a path already read; loop% = >=3 identical consecutive tool calls.")


if __name__ == "__main__":
    main()
