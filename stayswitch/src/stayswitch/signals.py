"""Cheap, per-call features of an agent trajectory that routing policies can read.

Everything here works on the chat messages the proxy sees. For terminus-2 the
assistant turns are JSON with ``commands[].keystrokes`` and each user turn after
the first carries the terminal output of the previous commands. For Claude Code
(Anthropic ``/v1/messages``) the assistant turns hold ``tool_use`` blocks and the
following user turn holds the matching ``tool_result`` blocks; ``keystrokes`` maps
those tool calls onto the same shell-like text so ``step_kind`` and ``repeated_action``
work unchanged.
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
    r"|SyntaxError|\bFAILED\b|\b\d+ failed\b|AssertionError|ModuleNotFoundError|\bExit code [1-9]\d*"
)

# Claude Code tools -> the shell text a terminus-style agent would have typed for the same step.
_EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_READ_TOOLS = {"Read": ("cat", "file_path"), "Grep": ("grep", "pattern"), "Glob": ("find", "pattern"), "LS": ("ls", "path")}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _tool_use_keys(block: dict[str, Any]) -> str:
    name, args = block.get("name", ""), block.get("input") or {}
    if not isinstance(args, dict):
        return ""
    if name == "Bash":
        return str(args.get("command", ""))
    if name in _EDIT_TOOLS:
        return f"str_replace {args.get('file_path') or args.get('notebook_path') or ''}"
    if name in _READ_TOOLS:
        command, key = _READ_TOOLS[name]
        return f"{command} {args.get(key, '')}"
    return ""  # Task, TodoWrite, WebFetch, ...: no shell-level meaning


def _tool_text(content: Any) -> str:
    """Text of a tool_result's content (a string or a list of text parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def keystrokes(message: dict[str, Any]) -> str:
    """The shell input of one assistant turn ('' if it is neither terminus-style JSON nor tool calls)."""
    content = message.get("content")
    if isinstance(content, list):
        keys = [_tool_use_keys(b) for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        if keys:
            return " ; ".join(k for k in keys if k)
    text = _text(content)
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


def _observation(message: dict[str, Any]) -> tuple[str, bool]:
    """(text, tool reported an error) of a user turn: plain text, plus every tool_result block."""
    content = message.get("content")
    if not isinstance(content, list):
        return _text(content), False
    parts, errored = [], False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            parts.append(_tool_text(block.get("content")))
            errored = errored or bool(block.get("is_error"))
        else:
            parts.append(block.get("text", ""))
    return " ".join(parts), errored


def _last_observation(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    # Claude Code interleaves role="system" notes (a running token counter); they are not turns.
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) >= 2 and turns[-1].get("role") == "user" and turns[-2].get("role") == "assistant":
        return _observation(turns[-1])
    return "", False


def last_observation(messages: list[dict[str, Any]]) -> str:
    """Terminal output the agent is about to react to (the final user turn, if it follows an assistant turn)."""
    return _last_observation(messages)[0]


def observation_failed(messages: list[dict[str, Any]]) -> bool:
    text, errored = _last_observation(messages)
    return errored or bool(_ERROR.search(text[-4000:]))


def repeated_action(messages: list[dict[str, Any]]) -> bool:
    """The last two assistant turns sent the same non-empty keystrokes: a sign of looping."""
    turns = assistant_turns(messages)
    if len(turns) < 2:
        return False
    a, b = keystrokes(turns[-1]).strip(), keystrokes(turns[-2]).strip()
    return bool(a) and a == b


def kinds(messages: list[dict[str, Any]]) -> list[str]:
    return [step_kind(keystrokes(m)) for m in assistant_turns(messages)]
