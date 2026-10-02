"""Test-time scaling from the seeds: treat an arm's k seeds as k rollouts of each task (CliffCompaction paper, section 4).

  uv run python scripts/pass_at_k.py native=cc_mid,cc_mid_r2,cc_mid_r3 eoq1=...

Per arm, over tasks run in all k seeds: pass@1 (mean over seeds), oracle pass@k (solved by at least one seed), the cost
of k rollouts per task (perfect-cache), and oracle solves per dollar. Runs are joined to trials as in report_arms.py.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_arms import load  # noqa: E402


def main() -> None:
    arms = {a.split("=")[0]: [load(r) for r in a.split("=", 1)[1].split(",")] for a in sys.argv[1:]}
    k = max(len(v) for v in arms.values())
    print(f"{'arm':10s} {'tasks':>5s} {'pass@1':>7s} {f'pass@{k}':>8s} {'$/task x1':>10s} {f'$/task x{k}':>11s} {'solved/$':>9s}")
    for name, seeds in arms.items():
        common = set.intersection(*[set(s) for s in seeds if s])
        if not common:
            continue
        p1 = statistics.fmean(((s[t]["reward"] or 0) >= 1) for s in seeds for t in common) * 100
        pk = statistics.fmean(any((s[t]["reward"] or 0) >= 1 for s in seeds) for t in common) * 100
        c1 = statistics.fmean(s[t]["perfect"] for s in seeds for t in common)
        print(f"{name:10s} {len(common):5d} {p1:7.1f} {pk:8.1f} {c1:10.3f} {c1 * len(seeds):11.3f} {pk / 100 / (c1 * len(seeds)):9.2f}")
    print(f"tasks = those with all {k} seeds; pass@{k} is the oracle (any seed solved); solved/$ = oracle solves per dollar of {k} rollouts.")


if __name__ == "__main__":
    main()
