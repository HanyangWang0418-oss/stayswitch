"""EOQ v2: a per-trajectory compaction trigger with measured inputs.

Between compactions the prompt grows roughly linearly from L0 at g tokens per call and every call re-reads it
at the cached price c_r. A compaction costs C = c_w * (L0 - head) + e * (one call at L0): only the part of the
new prompt beyond the still-cached head (system prompt, tool schemas, task) is written again, plus ``e`` calls
of work the agent redoes because information was dropped. Minimising the average cost per call over a cycle
gives  L* = L0 + sqrt(2 g C / c_r).

v1 (``context.eoq_threshold``) assumed a small post-compaction size and a full cache rewrite, and set the
threshold too low. v2 differs in three ways:
  * L0 is the size the compactor actually produced for this trajectory (observed after each compaction), not a guess;
  * the cached head is excluded from the rewrite cost;
  * g is the mean growth per call over the current cycle (since the last compaction), which is what the cycle
    model assumes. A recency-weighted average was tried first: it tracks the quiet calls, misses the occasional
    large tool result, and under-estimated g by ~40% on Claude Code (seed 1), so it fired ~11K too early.
Before a trajectory's first compaction there is nothing to observe, so L0 is estimated as head + (keep + 1) * g.
"""

from __future__ import annotations

from dataclasses import dataclass

from stayswitch.pricing import PER_M, Price


def eoq_l_star(
    price: Price, *, l0: float, head: float, growth: float, extra_steps: float,
    out_tokens: int = 900, min_gap: int = 4, ceiling: int = 52000,
) -> float:
    """Optimal compaction trigger, in prompt tokens, for a cycle that restarts at ``l0`` and grows ``growth`` per call."""
    g = max(growth, 50.0)
    l0 = max(l0, 1.0)
    step_cost = (price.cache_read * l0 + price.input * g + price.output * out_tokens) / PER_M
    cost = price.cache_write * max(l0 - head, 0.0) / PER_M + extra_steps * step_cost
    star = l0 + (2 * g * cost / (price.cache_read / PER_M)) ** 0.5
    return min(max(star, l0 + min_gap * g), max(ceiling, l0 + min_gap * g))


@dataclass
class _Trajectory:
    l0: float | None = None  # prompt size right after the latest compaction
    growth: float | None = None  # mean growth per call over the current cycle
    cycle_start: int | None = None  # outgoing size at the start of the current cycle
    cycle_calls: int = 0
    compactions: int = 0


class EOQTrigger:
    def __init__(self, price: Price, *, extra_steps: float, keep_recent: int, out_tokens: int = 900,
                 min_gap: int = 4, ceiling: int = 52000) -> None:
        self.price, self.extra_steps, self.keep_recent = price, extra_steps, keep_recent
        self.out_tokens, self.min_gap, self.ceiling = out_tokens, min_gap, ceiling
        self._state: dict[str, _Trajectory] = {}

    def _get(self, key: str) -> _Trajectory:
        return self._state.setdefault(key, _Trajectory())

    def threshold(self, key: str, *, head: int, mean_growth: float) -> tuple[int, dict]:
        """Trigger for the next request of trajectory ``key``. ``mean_growth`` is the fallback growth rate
        (tokens per turn averaged over the whole history) used until a cycle has been observed."""
        s = self._get(key)
        g = s.growth if s.growth is not None else mean_growth
        l0 = s.l0 if s.l0 is not None else head + (self.keep_recent + 1) * g
        star = eoq_l_star(self.price, l0=l0, head=head, growth=g, extra_steps=self.extra_steps,
                          out_tokens=self.out_tokens, min_gap=self.min_gap, ceiling=self.ceiling)
        return int(star), {"l0": round(l0), "growth": round(g), "l0_observed": s.l0 is not None, "compactions": s.compactions}

    def observe(self, key: str, *, est_out: int, compacted: bool) -> None:
        """Record the size of the request that was just sent (after any compaction)."""
        s = self._get(key)
        if compacted:
            s.l0, s.cycle_start, s.cycle_calls, s.growth = float(est_out), est_out, 0, None
            s.compactions += 1
            return
        if s.cycle_start is None:
            s.cycle_start = est_out
            return
        s.cycle_calls += 1
        if est_out > s.cycle_start:
            s.growth = (est_out - s.cycle_start) / s.cycle_calls
