"""Re-price a trajectory under explicit cache semantics (experiment E0).

A trajectory is a sequence of calls, each with the model used, the prompt
size, the output size and a timestamp. ``CacheLedger`` tracks the prefix each
model holds (its last prompt, as long as it is alive under the semantics and
the history has not been rewritten); a call reads that prefix and pays a cache
write on the rest. Switching away and back therefore loses only the segment the
other model ran, and a compaction or summary makes every model cold.
Comparing the result against the cache-oblivious price shows how much of a
router's saving the switches eat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from stayswitch.cache import CacheLedger, CacheSemantics
from stayswitch.costmodel import CostModel
from stayswitch.pricing import PER_M, PriceTable


@dataclass(frozen=True)
class Call:
    model: str
    prompt_tokens: int
    output_tokens: int
    ts: float | None = None
    prefix_reset: bool = False  # the history sent was rewritten before this call (compaction, summary)

    @classmethod
    def from_record(cls, rec: Mapping[str, Any]) -> Call:
        """A call from one ``calls.jsonl`` line (a successful one: it must carry ``usage``)."""
        u = rec["usage"]
        return cls(
            model=rec["model"],
            prompt_tokens=u["fresh_input"] + u["cache_read"] + u["cache_write"],
            output_tokens=u["output"],
            ts=rec.get("ts"),
            prefix_reset=bool((rec.get("ctx") or {}).get("compacted")),
        )


@dataclass(frozen=True)
class Bill:
    cache_aware: float        # what the provider would charge with prompt caching
    cache_oblivious: float    # every prompt token at the plain input price (how most routing papers count)
    switches: int
    cold_calls: int           # calls that read nothing from cache
    read_cost: float = 0.0    # the part of cache_aware paid for cache reads
    resets: int = 0           # calls where the history had been rewritten (flagged, or the prompt shrank)


def reprice(
    calls: Iterable[Call], prices: PriceTable, semantics: CacheSemantics | None = None, *, ttl_s: float | None = 300.0
) -> Bill:
    """``semantics`` wins; ``ttl_s`` alone keeps the table's cache prices with that TTL (``None`` = persistent)."""
    sem = semantics or CacheSemantics(ttl_s=ttl_s)
    model = CostModel(prices, sem)
    ledger = CacheLedger()
    aware = oblivious = read_cost = 0.0
    switches = cold = resets = 0
    prev_model: str | None = None
    for call in calls:
        if call.prefix_reset or (ledger.entries and call.prompt_tokens < ledger.longest()):
            ledger.entries.clear()
            resets += 1
        prefix = ledger.cached_prefix(call.model, call.prompt_tokens, call.ts, sem)
        if prefix == 0:
            cold += 1
        aware += model.call_cost(call.model, call.prompt_tokens, call.output_tokens, prefix)
        read_cost += prefix * model.price(call.model).cache_read / PER_M
        price = prices[call.model]
        oblivious += (call.prompt_tokens * price.input + call.output_tokens * price.output) / PER_M
        if prev_model is not None and prev_model != call.model:
            switches += 1
        ledger.record(call.model, call.prompt_tokens, call.ts)
        prev_model = call.model
    return Bill(aware, oblivious, switches, cold, read_cost, resets)
