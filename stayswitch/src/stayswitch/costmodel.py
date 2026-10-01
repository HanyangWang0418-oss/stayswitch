"""Analytic cost of staying on a model or switching, under explicit cache semantics.

This is the known part of the routing objective Q(s, o) = lambda * p_succ - C
(proposal section 2.1): given the price table, the provider's cache rules and
what each model currently has cached, the cost of the next call on any model,
the premium a switch pays for a cold cache, and how many steps a cheaper model
must run to pay that premium back. Nothing here is learned.
"""

from __future__ import annotations

from stayswitch.cache import CacheLedger, CacheSemantics
from stayswitch.pricing import PER_M, Price, PriceTable, Usage, cost


class CostModel:
    def __init__(self, prices: PriceTable, semantics: CacheSemantics = CacheSemantics()) -> None:
        self.prices = prices
        self.semantics = semantics

    def price(self, model: str) -> Price:
        return self.semantics.price(self.prices[model])

    def call_cost(self, model: str, prompt_tokens: int, output_tokens: int, cached_prefix: int) -> float:
        """One call that reads ``cached_prefix`` tokens from cache and writes the rest for the next call."""
        rest = max(prompt_tokens - cached_prefix, 0)
        written = prompt_tokens >= self.semantics.min_prefix  # too short to cache: plain input, no write
        usage = Usage(
            fresh_input=0 if written else rest,
            cache_read=min(cached_prefix, prompt_tokens),
            cache_write=rest if written else 0,
            output=output_tokens,
        )
        return cost(self.price(model), usage)

    def step_cost(self, model: str, *, ctx_tokens: int, new_tokens: int, output_tokens: int) -> float:
        """Steady-state cost of a step on a model whose cache holds the whole previous prompt.

        The new tokens are charged at the cache-write price, since they are written for the
        next call to read; on providers without a write premium that equals the input price.
        """
        return self.call_cost(model, ctx_tokens + new_tokens, output_tokens, cached_prefix=ctx_tokens)

    def next_call_cost(
        self, model: str, ledger: CacheLedger, *, ctx_tokens: int, new_tokens: int, output_tokens: int, now: float | None = None
    ) -> float:
        """Cost of the next call if it goes to ``model``, given what the ledger says it has cached."""
        prompt = ctx_tokens + new_tokens
        return self.call_cost(model, prompt, output_tokens, ledger.cached_prefix(model, prompt, now, self.semantics))

    def switch_premium(
        self, model: str, ledger: CacheLedger, *, ctx_tokens: int, new_tokens: int, output_tokens: int, now: float | None = None
    ) -> float:
        """What the next call on ``model`` pays beyond a hot step, because its cache is cold or partial."""
        actual = self.next_call_cost(model, ledger, ctx_tokens=ctx_tokens, new_tokens=new_tokens, output_tokens=output_tokens, now=now)
        return actual - self.step_cost(model, ctx_tokens=ctx_tokens, new_tokens=new_tokens, output_tokens=output_tokens)

    def return_premium(self, model: str, *, ctx_tokens: int, segment_tokens: int, gap_s: float = 0.0) -> float:
        """Extra cost of coming back to ``model`` after ``segment_tokens`` were appended by another model.

        If the old prefix survived (persistent cache, or back within the TTL) only the segment is
        written; otherwise the whole prompt is. ``gap_s`` is the time spent away.
        """
        p = self.price(model)
        lost = segment_tokens if self.semantics.alive(0.0, gap_s) else ctx_tokens + segment_tokens
        return lost * (p.cache_write - p.cache_read) / PER_M

    def breakeven_steps(
        self,
        from_model: str,
        to_model: str,
        *,
        ctx_tokens: int,
        new_tokens: int,
        output_tokens: int,
        permanent: bool = False,
        gap_s: float = 0.0,
        now: float | None = None,
    ) -> float:
        """Steps ``to_model`` must run, at constant context, for its per-step saving to repay the switch.

        ``permanent=False`` also charges the return to ``from_model`` after those steps
        (``return_premium`` on the segment run away, with ``gap_s`` away). ``inf`` means
        the switch never pays for itself. The proposal's observation-A table is this
        function with ``gap_s`` beyond the TTL; section 11's correction is ``gap_s=0``
        under a persistent cache.
        """
        ledger = CacheLedger()
        ledger.record(from_model, ctx_tokens, now)
        premium = self.switch_premium(to_model, ledger, ctx_tokens=ctx_tokens, new_tokens=new_tokens, output_tokens=output_tokens, now=now)
        saving = self.step_cost(from_model, ctx_tokens=ctx_tokens, new_tokens=new_tokens, output_tokens=output_tokens) - self.step_cost(
            to_model, ctx_tokens=ctx_tokens, new_tokens=new_tokens, output_tokens=output_tokens
        )
        if not permanent:
            # Coming back writes what the other model appended (new_tokens per step away) and,
            # if the old prefix did not survive the time away, the whole context as well.
            p = self.price(from_model)
            if not self.semantics.alive(0.0, gap_s):
                premium += ctx_tokens * (p.cache_write - p.cache_read) / PER_M
            saving -= new_tokens * (p.cache_write - p.cache_read) / PER_M
        if saving <= 0:
            return float("inf")
        return premium / saving
