"""Cache-aware cost accounting.

Cost is computed from token counts and a price table rather than trusted from
the provider bill, so the same trajectory can be re-priced under any price
sheet or cache assumption (observation C in proposal.md).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PER_M = 1_000_000


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> Price:
        inp = float(m["input"])
        return cls(
            input=inp,
            output=float(m["output"]),
            cache_read=float(m.get("cache_read", inp * float(m.get("cache_read_mult", 1.0)))),
            cache_write=float(m.get("cache_write", inp * float(m.get("cache_write_mult", 1.0)))),
        )


@dataclass(frozen=True)
class Usage:
    """Prompt tokens split into the three billing buckets, plus output."""

    fresh_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.fresh_input + self.cache_read + self.cache_write

    @classmethod
    def from_openai_usage(cls, usage: Mapping[str, Any]) -> Usage:
        """Split a LiteLLM-normalised usage block.

        LiteLLM reports ``prompt_tokens`` inclusive of cached tokens, cache reads
        under ``prompt_tokens_details.cached_tokens`` (or Anthropic's
        ``cache_read_input_tokens``) and cache writes as
        ``cache_creation_input_tokens``.
        """
        prompt = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        read = int(usage.get("cache_read_input_tokens") or details.get("cached_tokens") or 0)
        write = int(usage.get("cache_creation_input_tokens") or 0)
        return cls(
            fresh_input=max(prompt - read - write, 0),
            cache_read=read,
            cache_write=write,
            output=int(usage.get("completion_tokens") or 0),
        )


def cost(price: Price, usage: Usage) -> float:
    return (
        usage.fresh_input * price.input
        + usage.cache_read * price.cache_read
        + usage.cache_write * price.cache_write
        + usage.output * price.output
    ) / PER_M


def step_cost(price: Price, *, ctx_tokens: int, new_tokens: int, output_tokens: int, cache_hot: bool) -> float:
    """Analytic cost of one call: the cached prefix is read if hot, else written."""
    prefix = price.cache_read if cache_hot else price.cache_write
    return (ctx_tokens * prefix + new_tokens * price.input + output_tokens * price.output) / PER_M


class PriceTable:
    def __init__(self, prices: Mapping[str, Price]) -> None:
        self._prices = dict(prices)

    @classmethod
    def load(cls, path: str | Path) -> PriceTable:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls({name: Price.from_mapping(spec) for name, spec in raw["models"].items()})

    def __getitem__(self, model: str) -> Price:
        return self._prices[model]

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def names(self) -> list[str]:
        return list(self._prices)
