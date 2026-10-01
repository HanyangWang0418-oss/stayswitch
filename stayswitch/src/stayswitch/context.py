"""Proxy-side history eviction for agent prompts.

The agent keeps its full history; the proxy decides what is sent. The head (system
and task messages before the first assistant turn) is always kept, then the most
recent turns. Turns are evicted in blocks of ``chunk``: the cut point only moves
when ``chunk`` more turns have accumulated, so the prompt prefix stays identical
(and cached) for ``chunk`` consecutive calls. ``chunk=1`` is a plain sliding
window, which changes the prefix on every call and defeats the prompt cache.
With ``keep`` turns kept at minimum, the number of turns sent ranges over
[keep, keep + chunk - 1].
"""

from __future__ import annotations

from typing import Any

ELISION_NOTE = (
    "\n\n[Context note: some earlier steps of this session were omitted to save space. "
    "The most recent terminal output below reflects the current state.]"
)


def _with_note(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": content + ELISION_NOTE}
    if isinstance(content, list):
        return {**message, "content": [*content, {"type": "text", "text": ELISION_NOTE}]}
    return message


CHARS_PER_TOKEN = 3.5


def est_tokens(messages: list[dict[str, Any]]) -> int:
    return int(sum(len(_text(m.get("content"))) for m in messages) / CHARS_PER_TOKEN) + 4 * len(messages)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _clip(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Cut an over-long message to its head and tail. Depends only on the message, never on its
    position, so a message is sent identically on every call and the cache prefix survives."""
    content = message.get("content")
    if not isinstance(content, str) or len(content) <= max_chars:
        return message
    half = max_chars // 2
    return {**message, "content": content[:half] + f"\n[... {len(content) - max_chars} characters omitted ...]\n" + content[-half:]}


class Compactor:
    """Budget-triggered, one-shot history truncation (the CliffCompaction pattern).

    History grows untouched until the estimated prompt reaches a threshold, then the
    cut point jumps so only the task head and the last ``keep`` turns remain; it stays
    there until the threshold is reached again. ``threshold(x, view_tokens, head_tokens)``
    returns the trigger size in tokens and may read per-session state ``x`` (e.g. the
    observed context growth rate). State lives in the routing session's ``extra``.
    """

    def __init__(self, keep: int, threshold, max_msg_chars: int = 24000, min_gap: int = 4) -> None:
        self.keep, self.threshold, self.max_msg_chars, self.min_gap = keep, threshold, max_msg_chars, min_gap

    def _view(self, head, rest, starts, cut):
        tail = [_clip(m, self.max_msg_chars) for m in rest[starts[cut] :]] if cut < len(starts) else []
        head = [_clip(m, self.max_msg_chars) for m in head]
        if cut and head:
            head = [*head[:-1], _with_note(head[-1])]
        return [*head, *tail]

    def view(self, messages: list[dict[str, Any]], x: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        first_assistant = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
        head, rest = messages[:first_assistant], messages[first_assistant:]
        starts = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
        cut = x.get("cut", 0)
        if cut > len(starts):  # the harness rewrote its own history (e.g. its summariser ran): start over
            cut = x["cut"] = 0
        sent = self._view(head, rest, starts, cut)
        tokens = est_tokens(sent)
        prev = x.get("prev_tokens")
        if prev is not None and not x.get("just_compacted") and tokens > prev:
            g = tokens - prev
            x["growth"] = g if "growth" not in x else 0.7 * x["growth"] + 0.3 * g
        x["just_compacted"] = False
        head_tokens = est_tokens(self._view(head, rest, starts, max(len(starts) - self.keep, 0)))
        # Leave room for at least ``min_gap`` steps of growth after a compaction: if the kept turns
        # alone exceed the threshold, compacting every call degrades into a cache-breaking sliding window.
        floor = head_tokens + self.min_gap * x.get("growth", tokens / max(len(starts), 1))
        limit = max(self.threshold(x, tokens, head_tokens), floor)
        compacted = False
        if tokens >= limit and len(starts) - self.keep > cut:
            x["cut"] = cut = len(starts) - self.keep
            sent = self._view(head, rest, starts, cut)
            x["compactions"] = x.get("compactions", 0) + 1
            x["just_compacted"] = compacted = True
            tokens = est_tokens(sent)
        x["prev_tokens"] = tokens
        return sent, {"est_tokens": tokens, "threshold": int(limit), "cut": cut, "compacted": compacted}


def fixed_threshold(budget: int):
    return lambda x, tokens, head_tokens: budget


def eoq_threshold(price, *, extra_steps: float, out_tokens: int = 900, default_growth: int = 800, ceiling: int = 52000):
    """Economic-order-quantity rule for when to compact.

    Between compactions the prompt grows linearly from L0 at g tokens/step, and every
    step re-reads it at the cached price c_r. A compaction costs C: rewriting the kept
    L0 tokens into the cache (c_w * L0) plus ``extra_steps`` steps of work the agent
    redoes because information was dropped. Average cost per step over a cycle,
    c_r (L0 + L)/2 + C g / (L - L0), is minimised at L* = L0 + sqrt(2 g C / c_r).
    """
    per_m = 1_000_000

    def threshold(x, tokens, head_tokens):
        g = max(x.get("growth", default_growth), 50.0)
        l0 = max(head_tokens, 1)
        step_cost = (price.cache_read * l0 + price.input * g + price.output * out_tokens) / per_m
        c = price.cache_write * l0 / per_m + extra_steps * step_cost
        return min(l0 + (2 * g * c / (price.cache_read / per_m)) ** 0.5, ceiling)

    return threshold


def evict(messages: list[dict[str, Any]], keep: int, chunk: int) -> tuple[list[dict[str, Any]], int]:
    """Return (messages to send, number of assistant turns dropped)."""
    if keep < 1 or chunk < 1:
        raise ValueError("keep and chunk must be >= 1")
    first_assistant = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), None)
    if first_assistant is None:
        return messages, 0
    head, rest = messages[:first_assistant], messages[first_assistant:]
    starts = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
    excess = len(starts) - keep
    if excess < chunk:
        return messages, 0
    dropped = (excess // chunk) * chunk
    tail = rest[starts[dropped] :]
    if not head:
        return tail, dropped
    return [*head[:-1], _with_note(head[-1]), *tail], dropped
