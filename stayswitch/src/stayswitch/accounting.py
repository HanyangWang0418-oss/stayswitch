"""Re-price a trajectory under explicit cache semantics (experiment E0).

A trajectory is a sequence of calls, each with the model used, the prompt
size, the output size and a timestamp. Agent prompts are append-only, so each
model's cache holds the prefix up to the last prompt sent to that model: a
call reads that prefix if it is still alive (within ``ttl_s``; ``None`` for
persistent caches like DeepSeek's) and pays a cache write on the rest.
Switching away and back therefore loses only the segment the other model ran.
Comparing this against the cache-oblivious price shows how much of a
router's saving the switches eat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stayswitch.pricing import PER_M, PriceTable


@dataclass(frozen=True)
class Call:
    model: str
    prompt_tokens: int
    output_tokens: int
    ts: float | None = None


@dataclass(frozen=True)
class Bill:
    cache_aware: float        # what the provider would charge with prompt caching
    cache_oblivious: float    # every prompt token at the plain input price (how most routing papers count)
    switches: int
    cold_calls: int           # calls that paid a cache write on their prefix


def reprice(calls: Iterable[Call], prices: PriceTable, *, ttl_s: float | None = 300.0) -> Bill:
    aware = oblivious = 0.0
    switches = cold = 0
    cached: dict[str, tuple[int, float | None]] = {}  # model -> (cached prefix tokens, last use ts)
    prev_model: str | None = None
    for call in calls:
        price = prices[call.model]
        prefix = 0
        if call.model in cached:
            length, last_ts = cached[call.model]
            alive = ttl_s is None or call.ts is None or last_ts is None or call.ts - last_ts <= ttl_s
            prefix = min(length, call.prompt_tokens) if alive else 0
        if prefix == 0:
            cold += 1
        # The uncached part is written to cache (cache_write price) for the next call to read.
        aware += (
            prefix * price.cache_read
            + (call.prompt_tokens - prefix) * price.cache_write
            + call.output_tokens * price.output
        ) / PER_M
        oblivious += (call.prompt_tokens * price.input + call.output_tokens * price.output) / PER_M
        if prev_model is not None and prev_model != call.model:
            switches += 1
        cached[call.model] = (call.prompt_tokens, call.ts)
        prev_model = call.model
    return Bill(aware, oblivious, switches, cold)
