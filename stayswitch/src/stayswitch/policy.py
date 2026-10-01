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

from stayswitch.context import est_tokens
from stayswitch.costmodel import CostModel
from stayswitch.session import SessionState, messages_hash
from stayswitch.signals import assistant_turns, keystrokes, observation_failed, repeated_action, step_kind


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


class RandomStepPolicy:
    """Baseline: each call independently goes to the weak model with probability ``p_weak``."""

    def __init__(self, weak: str, strong: str, p_weak: float, seed: int = 0) -> None:
        self.weak, self.strong, self.p_weak = weak, strong, p_weak
        self._rng = random.Random(seed)

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        return Decision(self.weak if self._rng.random() < self.p_weak else self.strong)


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


def _fail_streak(state: SessionState, messages: list[dict[str, Any]]) -> int:
    """Consecutive calls whose incoming terminal output shows an error; kept across summaries."""
    streak = state.extra.get("fail_streak", 0) + 1 if observation_failed(messages) else 0
    state.extra["fail_streak"] = streak
    return streak


class WeakFirstPolicy:
    """Explore with the cheap model for ``k`` calls, then hand off to the strong one for good.

    The fixed-prefix form of SWE-Router (Son et al., 2026): their learned value
    decides whether to escalate after the prefix; this baseline always does.
    """

    def __init__(self, weak: str, strong: str, k: int) -> None:
        self.weak, self.strong, self.k = weak, strong, k

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        return Decision(self.weak if state.step < self.k else self.strong)


class TriggerEscalatePolicy:
    """Cheap model by default; escalate permanently on a failure signal.

    Heuristic stand-in for TACIT-Switch / ReDAct-style deferral: escalate after
    ``fail_streak`` consecutive failing observations, a repeated action, or
    ``max_weak_steps`` calls on the cheap model.
    """

    def __init__(self, weak: str, strong: str, fail_streak: int = 3, max_weak_steps: int | None = 40) -> None:
        self.weak, self.strong = weak, strong
        self.fail_streak, self.max_weak_steps = fail_streak, max_weak_steps

    def trigger(self, state: SessionState, messages: list[dict[str, Any]]) -> str | None:
        """Why the cheap model should hand over now, or None."""
        streak = _fail_streak(state, messages)
        if streak >= self.fail_streak:
            return "fail_streak"
        if repeated_action(messages):
            return "repeat"
        if self.max_weak_steps is not None and state.step >= self.max_weak_steps:
            return "max_weak"
        return None

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        reason = self.trigger(state, messages)
        if reason and not state.extra.get("escalated"):
            state.extra["escalated"] = reason
        return Decision(self.strong if state.extra.get("escalated") else self.weak, note=state.extra.get("escalated") or "")


class CostAwareEscalatePolicy(TriggerEscalatePolicy):
    """``trigger_escalate`` with a cost gate: the same signals ask for the strong model,
    but the handoff only happens when it is worth paying for.

    The extra cost of escalating now is analytic (``CostModel``): the cold-cache premium
    of the first strong call, given what this trajectory's ledger says is cached, plus
    ``remaining_steps`` times the per-step price difference at the current context size.
    It is paid iff it does not exceed ``value_usd``, the dollar value assigned to the
    escalation (lambda times the assumed success uplift in Q = lambda * p - C; sweep it
    for a cost/success curve). A trigger that is too expensive is re-examined on every
    later call: the context only grows, so it gets dearer, unless a compaction or summary
    empties the cache, which makes the switch free and lets the escalation go through.
    Escalation stays permanent, so the only difference from the baseline is the gate.
    """

    def __init__(
        self,
        weak: str,
        strong: str,
        cost_model: CostModel,
        *,
        value_usd: float,
        remaining_steps: int = 20,
        out_tokens: int = 900,
        fail_streak: int = 3,
        max_weak_steps: int | None = 40,
    ) -> None:
        super().__init__(weak, strong, fail_streak, max_weak_steps)
        self.cost_model, self.value_usd, self.remaining_steps, self.out_tokens = cost_model, value_usd, remaining_steps, out_tokens

    def escalation_cost(self, state: SessionState, messages: list[dict[str, Any]]) -> float:
        """Dollars the rest of the trajectory costs extra if the strong model takes over now."""
        ctx = state.ctx_tokens
        new = est_tokens(messages[-2:]) if ctx else est_tokens(messages)
        kw = dict(ctx_tokens=ctx, new_tokens=new, output_tokens=self.out_tokens)
        premium = self.cost_model.switch_premium(self.strong, state.cache, now=state.last_call_ts, **kw)
        per_step = self.cost_model.step_cost(self.strong, **kw) - self.cost_model.step_cost(self.weak, **kw)
        return premium + self.remaining_steps * per_step

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        x = state.extra
        reason = self.trigger(state, messages)
        if x.get("escalated"):
            return Decision(self.strong, note=x["escalated"])
        if reason is None:
            return Decision(self.weak)
        extra = self.escalation_cost(state, messages)
        if extra <= self.value_usd:
            x["escalated"] = f"{reason} cost={extra:.3f}"
            return Decision(self.strong, note=x["escalated"])
        x["deferred"] = x.get("deferred", 0) + 1
        return Decision(self.weak, note=f"{reason} too_expensive={extra:.3f}")


class StrongLeadPolicy:
    """StaySwitch v0 heuristic: the strong model localises and makes the first fix,
    the cheap model verifies and iterates, and trouble brings the strong model back
    for a committed segment.

    Phases: ``lead`` (strong) until an edit has been made and a test run has
    followed it, or ``lead_max`` calls; then ``follow`` (weak). In ``follow``, a
    failure signal (``fail_streak`` failing observations or a repeated action)
    switches to strong for ``commit`` calls before returning to weak.
    """

    def __init__(self, weak: str, strong: str, lead_max: int = 30, fail_streak: int = 3, commit: int = 8) -> None:
        self.weak, self.strong = weak, strong
        self.lead_max, self.fail_streak, self.commit = lead_max, fail_streak, commit

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        x = state.extra
        streak = _fail_streak(state, messages)
        turns = assistant_turns(messages)
        last_kind = step_kind(keystrokes(turns[-1])) if turns else "empty"
        if last_kind == "edit":
            x["edited"] = True
        elif last_kind == "test" and x.get("edited"):
            x["tested_after_edit"] = True

        if x.get("phase", "lead") == "lead":
            if x.get("tested_after_edit") or state.step >= self.lead_max:
                x["phase"] = "follow"
            else:
                return Decision(self.strong, note="lead")

        until = x.get("strong_until", -1)
        if state.step < until:
            return Decision(self.strong, note="rescue")
        if streak >= self.fail_streak or repeated_action(messages):
            x["strong_until"] = state.step + self.commit
            x["fail_streak"] = 0
            x["rescues"] = x.get("rescues", 0) + 1
            return Decision(self.strong, note="rescue")
        return Decision(self.weak, note="follow")


def build_policy(cfg: Mapping[str, Any], cost_model: CostModel | None = None) -> Policy:
    kind = cfg["kind"]
    if kind == "fixed":
        return FixedPolicy(cfg["model"])
    if kind == "random_segment":
        return RandomSegmentPolicy(cfg["models"], cfg["p_switch"], cfg.get("min_stay", 1), cfg.get("seed", 0))
    if kind == "random_step":
        return RandomStepPolicy(cfg["weak"], cfg["strong"], cfg["p_weak"], cfg.get("seed", 0))
    if kind == "weak_first":
        return WeakFirstPolicy(cfg["weak"], cfg["strong"], cfg["k"])
    if kind == "trigger_escalate":
        return TriggerEscalatePolicy(cfg["weak"], cfg["strong"], cfg.get("fail_streak", 3), cfg.get("max_weak_steps", 40))
    if kind == "cost_escalate":
        if cost_model is None:
            raise ValueError("cost_escalate needs a CostModel (prices + [cache] semantics)")
        return CostAwareEscalatePolicy(
            cfg["weak"], cfg["strong"], cost_model,
            value_usd=cfg["value_usd"], remaining_steps=cfg.get("remaining_steps", 20), out_tokens=cfg.get("out_tokens", 900),
            fail_streak=cfg.get("fail_streak", 3), max_weak_steps=cfg.get("max_weak_steps", 40),
        )
    if kind == "strong_lead":
        return StrongLeadPolicy(cfg["weak"], cfg["strong"], cfg.get("lead_max", 30), cfg.get("fail_streak", 3), cfg.get("commit", 8))
    if kind == "routellm":
        from stayswitch.routellm_router import RouteLLMPolicy  # heavy deps (torch); only the proxy env needs them

        return RouteLLMPolicy(cfg["weak"], cfg["strong"], cfg["threshold"], cfg.get("checkpoint", "routellm/bert_gpt4_augmented"))
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
