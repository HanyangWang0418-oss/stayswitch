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

import os
import time
import tomllib
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from stayswitch.budget import Budget
from stayswitch.calllog import CallLog
from stayswitch.policy import Decision, build_policy
from stayswitch.pricing import PriceTable, Usage, cost
from stayswitch.session import SessionStore, fallback_session_id, messages_hash

SESSION_HEADER = "x-session-id"
VIRTUAL_MODEL = "stayswitch"


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
    headers = (data.get("proxy_server_request") or {}).get("headers") or {}
    for name, value in headers.items():
        if name.lower() == SESSION_HEADER:
            return str(value)
    return data.get("session_id") or (data.get("metadata") or {}).get("session_id")


def _response_text(response_obj: Any) -> str:
    try:
        return response_obj.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


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

        state = self.sessions.get(sid, messages)
        step = state.step
        # Summarisation calls carry derived ids; they draw on their trajectory's budget.
        stop_reason = self.budget.exhausted(sid.split("-summarization-", 1)[0])
        decision = self.policy.decide(state, messages)
        if stop_reason is not None:
            decision = Decision(state.model or decision.model, replay=self.budget.stop_response, note="budget_stop")
        prev_model = state.model
        self.sessions.advance(state, decision.model)

        data["model"] = decision.model
        if decision.replay is not None:
            data["mock_response"] = decision.replay
        data.setdefault("metadata", {})["stayswitch"] = {
            "session_id": sid,
            "task": state.task,
            "step": step,
            "model": decision.model,
            "prev_model": prev_model,
            "switched": prev_model is not None and prev_model != decision.model,
            "replayed": decision.replay is not None,
            "note": decision.note,
            "input_hash": messages_hash(messages),
            "n_messages": len(messages),
            "diverged_at": state.diverged_at,
            "stop_reason": stop_reason,
        }
        return data

    def _meta(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        return metadata.get("stayswitch")

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
        self.budget.charge(meta["session_id"].split("-summarization-", 1)[0], usd)
        self.log.append(
            {
                "run_id": self.run_id,
                **meta,
                "ok": True,
                "response_text": _response_text(response_obj),
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
