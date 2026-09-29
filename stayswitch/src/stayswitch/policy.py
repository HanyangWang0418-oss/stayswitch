"""Routing policies: given a trajectory's state, pick the model for the next call.

``ForkPolicy`` implements replay-based forking: calls before the fork step
return the source trajectory's recorded response (the agent re-executes the
same actions, rebuilding the environment), then an option runs for a horizon,
then the base model takes over. No container snapshot is needed.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from stayswitch.session import SessionState, messages_hash


@dataclass(frozen=True)
class Decision:
    model: str
    replay: str | None = None  # recorded response text to return instead of calling a model
    note: str = ""


class Policy(Protocol):
    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision: ...


class FixedPolicy:
    def __init__(self, model: str) -> None:
        self.model = model

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        return Decision(self.model)


class RandomSegmentPolicy:
    """Baseline: switch to a random model with probability p, then stay for `min_stay` calls."""

    def __init__(self, models: list[str], p_switch: float, min_stay: int = 1, seed: int = 0) -> None:
        self.models = models
        self.p_switch = p_switch
        self.min_stay = min_stay
        self._rng = random.Random(seed)

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        stayed = state.extra.get("stayed", 0)
        if state.model is None:
            model = self.models[0]
        elif stayed >= self.min_stay and self._rng.random() < self.p_switch:
            model = self._rng.choice([m for m in self.models if m != state.model] or self.models)
        else:
            model = state.model
        state.extra["stayed"] = stayed + 1 if model == state.model else 1
        return Decision(model)


@dataclass(frozen=True)
class RecordedCall:
    step: int
    input_hash: str
    response_text: str
    model: str


class TraceBank:
    """Source trajectories for forking, one per task, read from the proxy's call log."""

    def __init__(self, traces: Mapping[str, list[RecordedCall]]) -> None:
        self._traces = dict(traces)

    @classmethod
    def from_call_log(cls, path: str | Path, *, session_ids: set[str] | None = None) -> TraceBank:
        by_session: dict[str, list[RecordedCall]] = defaultdict(list)
        task_of: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                sid = rec["session_id"]
                if session_ids is not None and sid not in session_ids:
                    continue
                task_of[sid] = rec["task"]
                by_session[sid].append(
                    RecordedCall(rec["step"], rec["input_hash"], rec["response_text"], rec["model"])
                )
        traces: dict[str, list[RecordedCall]] = {}
        for sid, calls in by_session.items():
            # First complete session per task wins; pass session_ids to choose explicitly.
            traces.setdefault(task_of[sid], sorted(calls, key=lambda c: c.step))
        return cls(traces)

    def get(self, task: str) -> list[RecordedCall] | None:
        return self._traces.get(task)


class ForkPolicy:
    """Replay the source trace up to ``fork_step``, run ``option_model`` for ``horizon`` calls, then ``base_model``.

    ``horizon=None`` keeps the option model to the end (a permanent handoff).
    If the live prefix stops matching the source trace before the fork step,
    replay ends there and the fork starts at the divergence point instead.
    """

    def __init__(
        self,
        bank: TraceBank,
        *,
        fork_step: int,
        option_model: str,
        base_model: str,
        horizon: int | None = None,
        check_divergence: bool = True,
    ) -> None:
        self.bank = bank
        self.fork_step = fork_step
        self.option_model = option_model
        self.base_model = base_model
        self.horizon = horizon
        self.check_divergence = check_divergence

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        trace = self.bank.get(state.task)
        if trace is None:
            return Decision(self.base_model, note="no-source-trace")
        fork_at = state.extra.setdefault("fork_at", self.fork_step)
        if state.step < fork_at and state.step < len(trace):
            recorded = trace[state.step]
            if self.check_divergence and recorded.input_hash != messages_hash(messages):
                state.diverged_at = state.step
                state.extra["fork_at"] = fork_at = state.step
            else:
                return Decision(recorded.model, replay=recorded.response_text, note="replay")
        if state.step < fork_at:  # source trace ended before the fork step
            state.extra["fork_at"] = fork_at = state.step
        in_option = self.horizon is None or state.step < fork_at + self.horizon
        return Decision(self.option_model if in_option else self.base_model, note="option" if in_option else "base")


def build_policy(cfg: Mapping[str, Any]) -> Policy:
    kind = cfg["kind"]
    if kind == "fixed":
        return FixedPolicy(cfg["model"])
    if kind == "random_segment":
        return RandomSegmentPolicy(cfg["models"], cfg["p_switch"], cfg.get("min_stay", 1), cfg.get("seed", 0))
    if kind == "fork":
        sessions = cfg.get("source_sessions")
        bank = TraceBank.from_call_log(cfg["source_log"], session_ids=set(sessions) if sessions else None)
        horizon = cfg.get("horizon", 0)
        return ForkPolicy(
            bank,
            fork_step=cfg["fork_step"],
            option_model=cfg["option_model"],
            base_model=cfg["base_model"],
            horizon=horizon or None,
            check_divergence=cfg.get("check_divergence", True),
        )
    raise ValueError(f"unknown policy kind {kind!r}")
