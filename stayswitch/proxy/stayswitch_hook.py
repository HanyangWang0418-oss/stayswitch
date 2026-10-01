"""LiteLLM proxy callback: route each agent call and log it.

Agents call the virtual model ``stayswitch``. The pre-call hook identifies the
trajectory by its ``X-Session-ID`` header (Harbor's terminus-2 and
mini-swe-agent send one per trial), asks the policy for a model, and rewrites
``data["model"]``; a replayed step returns the recorded text through
LiteLLM's ``mock_response`` without calling a provider. Every call, replayed
or live, is appended to the call log.

Config comes from the TOML file named by ``STAYSWITCH_CONFIG``.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from stayswitch.budget import Budget
from stayswitch.calllog import CallLog
from stayswitch.context import Compactor, eoq_threshold, evict, fixed_threshold, overhead_tokens
from stayswitch.policy import Decision, ForkPolicy, build_policy
from stayswitch.litellm_patches import split_parallel_tool_calls
from stayswitch.pricing import PriceTable, Usage, cost
from stayswitch.replay_server import REPLAY_HEADER, encode_recording, ensure_started
from stayswitch.session import SessionStore, agent_key, fallback_session_id, messages_hash, root_session_id

# Trajectory id headers: Harbor's LiteLLM-based agents send X-Session-ID, Claude Code sends x-claude-code-session-id.
split_parallel_tool_calls()  # Claude Code on OpenAI-style pools: see litellm_patches
SESSION_HEADERS = ("x-session-id", "x-claude-code-session-id")
VIRTUAL_MODEL = "stayswitch"
REPLAY_MODEL = "replay"  # litellm_config.yaml entry that points at stayswitch.replay_server


def _load_config() -> dict[str, Any]:
    path = os.environ.get("STAYSWITCH_CONFIG")
    if not path:
        raise RuntimeError("STAYSWITCH_CONFIG must name the router TOML config")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for section, key in (("log", "path"), ("prices", "path"), ("policy", "source_log")):
        value = cfg.get(section, {}).get(key)
        if value and not os.path.isabs(value):
            cfg[section][key] = os.path.join(base, value)
    cfg["_dir"] = base
    return cfg


def _session_id(data: dict[str, Any]) -> str | None:
    headers = {k.lower(): v for k, v in ((data.get("proxy_server_request") or {}).get("headers") or {}).items()}
    for name in SESSION_HEADERS:
        if headers.get(name):
            return str(headers[name])
    return data.get("session_id") or (data.get("metadata") or {}).get("session_id")


def _response_text(response_obj: Any) -> str:
    try:
        return response_obj.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _tool_calls(response_obj: Any) -> list[dict[str, Any]]:
    try:
        calls = response_obj.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        return []
    return [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in calls]


def _replay_extra(response_obj: Any, usage: Usage) -> dict[str, Any]:
    """What a replay needs beyond text and tool calls so the agent sees the same prompt next step."""
    try:
        thinking = getattr(response_obj.choices[0].message, "reasoning_content", None) or ""
    except (AttributeError, IndexError):
        thinking = ""
    return {
        "thinking": thinking,
        "input_tokens": usage.fresh_input,
        "cache_read_input_tokens": usage.cache_read,
        "output_tokens": usage.output,
    }


def _usage(response_obj: Any) -> dict[str, Any]:
    usage = getattr(response_obj, "usage", None)
    if usage is None:
        return {}
    return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


class StaySwitchRouter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        cfg = _load_config()
        self.policy = build_policy(cfg["policy"])
        self.prices = PriceTable.load(cfg["prices"]["path"])
        self.log = CallLog(cfg["log"]["path"])
        self.run_id = cfg.get("run", {}).get("id", "")
        self.sessions = SessionStore()
        self.budget = Budget.from_config(cfg.get("budget", {}), config_dir=cfg["_dir"])
        self.context = cfg.get("context")  # {"keep": N, "chunk": K} enables proxy-side history eviction

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        if data.get("model") != VIRTUAL_MODEL:
            return data
        messages = data.get("messages") or []
        sid = _session_id(data)
        if sid is None:
            # Terminus-2 does not actually send X-Session-ID (it never passes its session id to its LLM client).
            sid = fallback_session_id(messages)
        # Harbor also puts the session id in extra_body; providers reject unknown fields.
        data.pop("session_id", None)
        if str(call_type) == "anthropic_messages":
            # Claude Code's effort/thinking/context-management fields become reasoning_effort etc. on
            # OpenAI-compatible pools, which reject them; the model pool decides its own reasoning.
            for key in ("output_config", "thinking", "context_management", "reasoning_effort"):
                data.pop(key, None)

        if str(call_type) == "anthropic_messages":
            # A Claude Code subagent shares the session header with its parent; give it its own trajectory.
            key = agent_key(data.get("system"), data.get("tools"))
            main = self.sessions.peek(sid)
            if main is not None and main.extra.setdefault("agent_key", key) != key:
                sid = f"{sid}~sub-{key[:8]}"
        state = self.sessions.get(sid, messages)
        state.extra.setdefault("agent_key", agent_key(data.get("system"), data.get("tools")) if str(call_type) == "anthropic_messages" else None)
        input_hash = messages_hash(messages)
        # Anthropic clients retry a call whose response they could not use (replayed mock responses are not
        # streamed). An identical prompt in the same session is that retry, not a new step.
        retry = str(call_type) == "anthropic_messages" and state.extra.get("last_hash") == input_hash
        if retry:
            decision, prev_model = state.extra["last_decision"], state.extra["last_prev_model"]
            step, stop_reason, parent = state.step - 1, state.extra["last_stop"], None
        else:
            step = state.step
            # Summarisation calls carry derived ids; they draw on their trajectory's budget.
            root = root_session_id(sid)
            stop_reason = self.budget.exhausted(root)
            parent = self.sessions.peek(root) if root != sid else None
            if parent is not None and parent.model is not None and not isinstance(self.policy, ForkPolicy):
                # Terminus-2's context summaries run under a derived session id; they belong to the
                # trajectory, so they use whatever model the trajectory is on instead of a fresh decision.
                decision = Decision(parent.model, note="summary" if "-summarization-" in sid else "subagent")
            else:
                decision = self.policy.decide(state, messages)
            if stop_reason is not None:
                decision = Decision(state.model or decision.model, replay=self.budget.stop_response, note="budget_stop")
            prev_model = state.model
            self.sessions.advance(state, decision.model)
            state.extra.update(last_hash=input_hash, last_decision=decision, last_prev_model=prev_model, last_stop=stop_reason)

        data["model"] = decision.model
        if decision.replay is not None and str(call_type) == "anthropic_messages":
            # /v1/messages cannot mock tool calls or streams: serve the recording from the replay backend.
            ensure_started()
            data["model"] = REPLAY_MODEL
            data["extra_headers"] = {**(data.get("extra_headers") or {}), REPLAY_HEADER: encode_recording(decision.replay, decision.replay_tool_calls, decision.replay_extra)}
        elif decision.replay is not None:
            data["mock_response"] = decision.replay
            if decision.replay_tool_calls:
                data["mock_tool_calls"] = decision.replay_tool_calls
        ctx_dropped, ctx_info = 0, {}
        if self.context and decision.replay is None and parent is None and root_session_id(sid) == sid:
            mode = self.context.get("mode", "evict")
            if mode == "evict":
                data["messages"], ctx_dropped = evict(messages, self.context["keep"], self.context.get("chunk", 1))
            else:
                if mode == "budget":
                    threshold = fixed_threshold(self.context["budget"])
                else:  # "eoq"
                    threshold = eoq_threshold(
                        self.prices[decision.model],
                        extra_steps=self.context.get("extra_steps", 2.0),
                        out_tokens=self.context.get("out_tokens", 900),
                        ceiling=self.context.get("ceiling", 52000),
                    )
                compactor = Compactor(self.context.get("keep", 4), threshold, self.context.get("max_msg_chars", 24000))
                # Anthropic requests keep the system prompt and tool schemas outside "messages"; they count.
                overhead = overhead_tokens(data.get("system"), data.get("tools"))
                data["messages"], ctx_info = compactor.view(messages, state.extra.setdefault("ctx", {}), overhead)
                ctx_dropped = ctx_info["cut"]
        # Anthropic-format routes (/v1/messages) carry LiteLLM metadata under "litellm_metadata", because
        # "metadata" is a real Anthropic API field there.
        path = str((data.get("proxy_server_request") or {}).get("url", ""))
        meta_key = "litellm_metadata" if "/messages" in path else "metadata"
        data.setdefault(meta_key, {})["stayswitch"] = {
            "session_id": sid,
            "task": state.task,
            "step": step,
            "model": decision.model,
            "prev_model": prev_model,
            "switched": prev_model is not None and prev_model != decision.model,
            "replayed": decision.replay is not None,
            "note": decision.note,
            "input_hash": input_hash,
            "n_messages": len(messages),
            "diverged_at": state.diverged_at,
            "stop_reason": stop_reason,
            "ctx_dropped_turns": ctx_dropped,
            "ctx": ctx_info,
            "n_messages_sent": len(data["messages"]),
        }
        return data

    def _meta(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        params = kwargs.get("litellm_params") or {}
        for key in ("metadata", "litellm_metadata"):
            found = (params.get(key) or {}).get("stayswitch")
            if found:
                return found
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        meta = self._meta(kwargs)
        if meta is None:
            return
        usage_raw = _usage(response_obj)
        usage = Usage.from_openai_usage(usage_raw)
        self.sessions.observe(meta["session_id"], usage.prompt_tokens)
        model = meta["model"]
        # Replayed and budget-stop responses are mocks: their usage is fake and nothing was billed.
        usd = 0.0 if meta["replayed"] or model not in self.prices else cost(self.prices[model], usage)
        self.budget.charge(root_session_id(meta["session_id"]), usd)
        self.log.append(
            {
                "run_id": self.run_id,
                **meta,
                "ok": True,
                "response_text": _response_text(response_obj),
                "tool_calls": _tool_calls(response_obj),
                "replay_extra": _replay_extra(response_obj, usage),
                "usage": {
                    "fresh_input": usage.fresh_input,
                    "cache_read": usage.cache_read,
                    "cache_write": usage.cache_write,
                    "output": usage.output,
                },
                "usage_raw": usage_raw,
                "cost": usd,
                "provider_cost": kwargs.get("response_cost"),
                "latency_s": (end_time - start_time).total_seconds(),
                "ts": time.time(),
            }
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        meta = self._meta(kwargs)
        if meta is None:
            return
        self.log.append(
            {
                "run_id": self.run_id,
                **meta,
                "ok": False,
                "error": str(kwargs.get("exception") or response_obj)[:2000],
                "ts": time.time(),
            }
        )


proxy_handler_instance = StaySwitchRouter()
