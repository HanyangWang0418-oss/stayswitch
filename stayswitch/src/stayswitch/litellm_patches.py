"""Workarounds for LiteLLM behaviour that breaks Anthropic-format agents (Claude Code) on OpenAI-style pools."""

from __future__ import annotations

import copy
from typing import Any


def split_parallel_tool_calls() -> bool:
    """Make the /v1/messages stream translator emit one tool_use block per parallel tool call.

    Tinker sends all of a response's tool-call deltas in one chunk (call 0's name and arguments, then call
    1's). The translator reads only the chunk as a whole, opens one tool_use block and concatenates every
    call's arguments into it (``{"a":1}{"b":2}``), which Claude Code rejects as unparsable. Splitting such
    chunks into one chunk per delta makes the translator start a block at each new call's name. Returns
    False if this LiteLLM has no such hook.
    """
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import _CombinedChunkSplitter
    except ImportError:
        return False
    original = _CombinedChunkSplitter._split_by_payload_kind
    if getattr(original, "_stayswitch_patched", False):
        return True

    def patched(chunk: Any) -> tuple[Any, ...]:
        pieces: list[Any] = []
        for piece in original(chunk):
            choices = getattr(piece, "choices", None) or []
            calls = (getattr(choices[0].delta, "tool_calls", None) or []) if len(choices) == 1 else []
            if len(calls) > 1:
                for call in calls:
                    single = copy.deepcopy(piece)
                    single.choices[0].delta.tool_calls = [copy.deepcopy(call)]
                    pieces.append(single)
            else:
                pieces.append(piece)
        return tuple(pieces)

    patched._stayswitch_patched = True  # type: ignore[attr-defined]
    _CombinedChunkSplitter._split_by_payload_kind = staticmethod(patched)  # type: ignore[method-assign]
    return True
