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

import json
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
    """All prompt text of a message's content, including Anthropic thinking, tool_use and tool_result blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_block_text(p) for p in content if isinstance(p, dict))
    return ""


def _block_text(block: dict[str, Any]) -> str:
    kind = block.get("type")
    if kind == "tool_use":
        return json.dumps(block.get("input") or {}) + str(block.get("name", ""))
    if kind == "tool_result":
        return _text(block.get("content"))
    if kind == "thinking":
        return str(block.get("thinking", ""))
    return str(block.get("text", ""))


def overhead_tokens(*parts: Any) -> int:
    """Tokens of the fixed part of an Anthropic request (system prompt, tool definitions), which the
    message list does not contain but the context window pays for."""
    return int(sum(len(json.dumps(p, default=str)) for p in parts if p) / CHARS_PER_TOKEN)


def _clip_text(text: str, max_chars: int) -> str:
    half = max_chars // 2
    return text[:half] + f"\n[... {len(text) - max_chars} characters omitted ...]\n" + text[-half:]


def _clip_block(block: Any, max_chars: int) -> Any:
    if not isinstance(block, dict):
        return block
    if block.get("type") == "tool_result":
        inner = block.get("content")
        if isinstance(inner, str) and len(inner) > max_chars:
            return {**block, "content": _clip_text(inner, max_chars)}
        if isinstance(inner, list):
            return {**block, "content": [_clip_block(b, max_chars) for b in inner]}
        return block
    text = block.get("text")
    if isinstance(text, str) and len(text) > max_chars:
        return {**block, "text": _clip_text(text, max_chars)}
    return block


def _clip(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Cut an over-long message (or over-long text / tool_result block in it) to its head and tail.
    Depends only on the message, never on its position, so a message is sent identically on every
    call and the cache prefix survives."""
    content = message.get("content")
    if isinstance(content, list):
        clipped = [_clip_block(b, max_chars) for b in content]
        return message if all(a is b for a, b in zip(clipped, content)) else {**message, "content": clipped}
    if not isinstance(content, str) or len(content) <= max_chars:
        return message
    return {**message, "content": _clip_text(content, max_chars)}


def _note_target(head: list[dict[str, Any]]) -> int:
    """Index of the head message that carries the elision note: the last user turn. Claude Code's head can end
    with a role="system" note, and a note there would not be seen as part of the task."""
    return next((i for i in range(len(head) - 1, -1, -1) if head[i].get("role") == "user"), len(head) - 1)


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
            i = _note_target(head)
            head = [*head[:i], _with_note(head[i]), *head[i + 1 :]]
        return [*head, *tail]

    def view(
        self, messages: list[dict[str, Any]], x: dict[str, Any], overhead: int = 0
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """``overhead``: tokens sent with every call outside ``messages`` (system prompt, tool schemas)."""
        first_assistant = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
        head, rest = messages[:first_assistant], messages[first_assistant:]
        starts = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
        cut = x.get("cut", 0)
        if cut > len(starts):  # the harness rewrote its own history (e.g. its summariser ran): start over
            cut = x["cut"] = 0
        sent = self._view(head, rest, starts, cut)
        tokens = est_tokens(sent) + overhead
        prev = x.get("prev_tokens")
        if prev is not None and not x.get("just_compacted") and tokens > prev:
            g = tokens - prev
            x["growth"] = g if "growth" not in x else 0.7 * x["growth"] + 0.3 * g
        x["just_compacted"] = False
        head_tokens = est_tokens(self._view(head, rest, starts, max(len(starts) - self.keep, 0))) + overhead
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
            tokens = est_tokens(sent) + overhead
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
    i = _note_target(head)
    return [*head[:i], _with_note(head[i]), *head[i + 1 :], *tail], dropped
