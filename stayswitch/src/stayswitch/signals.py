"""Cheap, per-call features of an agent trajectory that routing policies can read.

Everything here works on the chat messages the proxy sees. For terminus-2 the
assistant turns are JSON with ``commands[].keystrokes`` and each user turn after
the first carries the terminal output of the previous commands.
"""

from __future__ import annotations

import json
import re
from typing import Any

_EDIT = re.compile(r"sed -i|cat\s*>|cat <<|\btee\b|python3? - <<|>\s*\S+\.(py|txt|cfg|toml|rst|md)\b|\bpatch\b|git apply|str_replace")
_TEST = re.compile(r"pytest|runtests|\btox\b|unittest|python3? \S*test\S*\.py|manage\.py test")
_EXPLORE = re.compile(r"\b(grep|rg|find|ls|cat|head|tail|sed -n|wc|less|tree)\b|git (log|diff|show|status|grep)")
_ERROR = re.compile(
    r"Traceback \(most recent call last\)|\bError:|\berror:|command not found|No such file or directory"
    r"|SyntaxError|\bFAILED\b|\b\d+ failed\b|AssertionError|ModuleNotFoundError"
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def keystrokes(message: dict[str, Any]) -> str:
    """The shell input of one assistant turn ('' if it is not terminus-style JSON)."""
    text = _text(message.get("content"))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return ""
    try:
        action = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ""
    return " ; ".join(str(c.get("keystrokes", "")) for c in action.get("commands", []) if isinstance(c, dict))


def step_kind(keys: str) -> str:
    if _EDIT.search(keys):
        return "edit"
    if _TEST.search(keys):
        return "test"
    if _EXPLORE.search(keys):
        return "explore"
    return "other" if keys.strip() else "empty"


def assistant_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m.get("role") == "assistant"]


def last_observation(messages: list[dict[str, Any]]) -> str:
    """Terminal output the agent is about to react to (the final user turn, if it follows an assistant turn)."""
    if len(messages) >= 2 and messages[-1].get("role") == "user" and messages[-2].get("role") == "assistant":
        return _text(messages[-1].get("content"))
    return ""


def observation_failed(messages: list[dict[str, Any]]) -> bool:
    return bool(_ERROR.search(last_observation(messages)[-4000:]))


def repeated_action(messages: list[dict[str, Any]]) -> bool:
    """The last two assistant turns sent the same non-empty keystrokes: a sign of looping."""
    turns = assistant_turns(messages)
    if len(turns) < 2:
        return False
    a, b = keystrokes(turns[-1]).strip(), keystrokes(turns[-2]).strip()
    return bool(a) and a == b


def kinds(messages: list[dict[str, Any]]) -> list[str]:
    return [step_kind(keystrokes(m)) for m in assistant_turns(messages)]
