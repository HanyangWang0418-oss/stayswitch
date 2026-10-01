"""Prompt-cache semantics and a per-trajectory record of what each model has cached.

Two things are kept apart on purpose:

* ``CacheSemantics`` is a provider's rulebook: how long a cached prefix lives,
  what reads and writes cost relative to plain input, the minimum cacheable
  prefix, and whether a prefix cached by one model is usable by another.
* ``CacheLedger`` records facts about one trajectory: which model last saw a
  prompt of what size, and when. It knows nothing about prices or TTLs.

Costs follow from the two together (see ``costmodel``), so the same trajectory
can be priced under any provider's rules by swapping the semantics only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from stayswitch.pricing import Price


@dataclass(frozen=True)
class CacheSemantics:
    """How a provider caches and bills prompt prefixes.

    ``ttl_s=None`` means the prefix never expires (persistent caches, as observed on
    Tinker and DeepSeek); ``0`` means nothing is ever cached. ``read_mult`` and
    ``write_mult`` override the price table's cache prices as multiples of the input
    price; ``None`` keeps whatever the table says. ``cross_model`` lets any model read
    a prefix cached by another, the hypothetical upper bound of proposal section 12.
    """

    ttl_s: float | None = None
    read_mult: float | None = None
    write_mult: float | None = None
    min_prefix: int = 0
    cross_model: bool = False
    name: str = "table"

    def alive(self, last_ts: float | None, now: float | None) -> bool:
        """Is a prefix last touched at ``last_ts`` still cached at ``now``?"""
        if self.ttl_s is None:
            return True
        if self.ttl_s <= 0:
            return False
        if last_ts is None or now is None:  # no timestamps: assume the gap was short
            return True
        return now - last_ts <= self.ttl_s

    def price(self, p: Price) -> Price:
        """The table price with this provider's cache multipliers applied."""
        return replace(
            p,
            cache_read=p.input * self.read_mult if self.read_mult is not None else p.cache_read,
            cache_write=p.input * self.write_mult if self.write_mult is not None else p.cache_write,
        )


# Presets. Multipliers are relative to the input price; check them against the provider's
# price page before quoting numbers (see configs/prices.toml for the table itself).
TABLE = CacheSemantics(name="table")                                                   # persistent, table prices
TINKER = CacheSemantics(ttl_s=None, read_mult=0.2, write_mult=1.0, name="tinker")
DEEPSEEK = CacheSemantics(ttl_s=None, read_mult=0.1, write_mult=1.0, name="deepseek")
ANTHROPIC = CacheSemantics(ttl_s=300.0, read_mult=0.1, write_mult=1.25, min_prefix=1024, name="anthropic")
OBLIVIOUS = CacheSemantics(ttl_s=0.0, read_mult=1.0, write_mult=1.0, name="oblivious")  # how most routing papers count

PRESETS: dict[str, CacheSemantics] = {s.name: s for s in (TABLE, TINKER, DEEPSEEK, ANTHROPIC, OBLIVIOUS)}


def semantics_from_config(cfg: Mapping[str, Any] | None) -> CacheSemantics:
    """``[cache]`` config: a ``preset`` name, optionally overridden field by field."""
    cfg = dict(cfg or {})
    base = PRESETS[cfg.pop("preset", "table")]
    return replace(base, **cfg) if cfg else base


@dataclass
class CacheLedger:
    """What each model has cached for one trajectory: the last prompt it saw and when.

    Agent prompts are append-only, so the prefix a model holds is its last prompt.
    A prompt shorter than any recorded one means the history was rewritten (the
    agent's summariser or the proxy's compactor ran), after which no model's prefix
    matches any more.
    """

    entries: dict[str, tuple[int, float | None]] = field(default_factory=dict)

    def cached_prefix(self, model: str, prompt_tokens: int, now: float | None, semantics: CacheSemantics) -> int:
        """Tokens of ``prompt_tokens`` that ``model`` would read from cache under ``semantics``."""
        if prompt_tokens < self.longest():
            return 0
        candidates = self.entries.items() if semantics.cross_model else [(model, self.entries.get(model, (0, None)))]
        best = 0
        for _, (length, last_ts) in candidates:
            if length >= semantics.min_prefix and semantics.alive(last_ts, now):
                best = max(best, min(length, prompt_tokens))
        return best

    def longest(self) -> int:
        return max((length for length, _ in self.entries.values()), default=0)

    def record(self, model: str, prompt_tokens: int, ts: float | None, *, prefix_reset: bool = False) -> None:
        """``model`` has now processed (and cached) a prompt of ``prompt_tokens``."""
        if prefix_reset or prompt_tokens < self.longest():
            self.entries.clear()
        self.entries[model] = (prompt_tokens, ts)
