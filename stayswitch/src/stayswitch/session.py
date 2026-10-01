"""Per-trajectory state the router sees at each call."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any


# Container shell prompts (``root@d598cdcdc569:/app#``) differ per trial; forks must still match their source.
_CONTAINER_HOST = re.compile(r"\b([\w.-]+)@[0-9a-f]{12}\b")


def _digest(obj: Any, *, normalize: bool = True) -> str:
    text = json.dumps(obj, sort_keys=True, default=str)
    if normalize:
        text = _CONTAINER_HOST.sub(r"\1@<host>", text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def messages_hash(messages: list[dict[str, Any]]) -> str:
    """Trial-independent hash of a prompt, used to check that a replayed prefix matches its source."""
    return _digest(messages)


def task_key(messages: list[dict[str, Any]]) -> str:
    """Identify the task by its first user message, so a fork finds its source trace."""
    for m in messages:
        if m.get("role") == "user":
            return _digest(m.get("content"))
    return messages_hash(messages[:1])


def fallback_session_id(messages: list[dict[str, Any]]) -> str:
    """Trajectory id when the agent sends no session header.

    Agent prompts are append-only, so the first message is fixed for a whole
    trajectory; for Harbor tasks it ends with the container's shell prompt,
    which makes it unique per trial. Hashed without normalisation for that reason.
    """
    return "anon-" + _digest(messages[:1], normalize=False)


@dataclass
class SessionState:
    session_id: str
    task: str
    step: int = 0
    model: str | None = None
    ctx_tokens: int = 0
    last_call_ts: float | None = None
    switches: int = 0
    # Replay bookkeeping for forks: once the live prefix diverges from the source trace, replay stops.
    diverged_at: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def seconds_since_last_call(self, now: float | None = None) -> float | None:
        if self.last_call_ts is None:
            return None
        return (now or time.time()) - self.last_call_ts


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str, messages: list[dict[str, Any]]) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id, task=task_key(messages))
                self._sessions[session_id] = state
            return state

    def peek(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def advance(self, state: SessionState, model: str) -> None:
        """Record the routing decision; called before the model call so steps stay ordered."""
        with self._lock:
            if state.model is not None and model != state.model:
                state.switches += 1
            state.model = model
            state.last_call_ts = time.time()
            state.step += 1

    def observe(self, session_id: str, prompt_tokens: int) -> None:
        """Record the context size the finished call actually had."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                state.ctx_tokens = prompt_tokens
