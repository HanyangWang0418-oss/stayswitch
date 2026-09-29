import json

import pytest

from stayswitch.policy import ForkPolicy, RandomSegmentPolicy, TraceBank, build_policy
from stayswitch.pricing import Price, Usage, cost, step_cost
from stayswitch.session import SessionStore, messages_hash

HAIKU = Price.from_mapping({"input": 1.0, "output": 5.0, "cache_write_mult": 1.25, "cache_read_mult": 0.1})


def test_usage_split_anthropic_style():
    u = Usage.from_openai_usage(
        {"prompt_tokens": 1000, "completion_tokens": 50, "cache_read_input_tokens": 700, "cache_creation_input_tokens": 200}
    )
    assert (u.fresh_input, u.cache_read, u.cache_write, u.output) == (100, 700, 200, 50)


def test_usage_split_openai_style():
    u = Usage.from_openai_usage({"prompt_tokens": 1000, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 900}})
    assert (u.fresh_input, u.cache_read, u.cache_write) == (100, 900, 0)


def test_cost_and_step_cost_agree():
    usage = Usage(fresh_input=2000, cache_read=60000, output=800)
    assert cost(HAIKU, usage) == pytest.approx(step_cost(HAIKU, ctx_tokens=60000, new_tokens=2000, output_tokens=800, cache_hot=True))
    cold = step_cost(HAIKU, ctx_tokens=60000, new_tokens=2000, output_tokens=800, cache_hot=False)
    assert cold == pytest.approx((60000 * 1.25 + 2000 + 800 * 5) / 1e6)


def _conv(task: str, n_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": task}]
    for i in range(n_turns):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"obs{i}"}]
    return msgs


def _write_trace(path, task: str, n: int, session="src"):
    store = SessionStore()
    with open(path, "w") as f:
        for step in range(n):
            msgs = _conv(task, step)
            state = store.get(session, msgs)
            f.write(json.dumps({"session_id": session, "task": state.task, "step": step,
                                "input_hash": messages_hash(msgs), "response_text": f"a{step}", "model": "strong"}) + "\n")


def _drive(policy, task: str, n: int, perturb_at: int | None = None):
    store = SessionStore()
    out = []
    for step in range(n):
        msgs = _conv(task, step)
        if perturb_at is not None and step >= perturb_at:
            msgs[-1] = {"role": "user", "content": f"different{step}"}
        state = store.get("fork", msgs)
        d = policy.decide(state, msgs)
        store.advance(state, d.model)
        out.append(d)
    return out, state


def test_fork_replays_then_option_then_base(tmp_path):
    log = tmp_path / "calls.jsonl"
    _write_trace(log, "fix the bug", 12)
    policy = ForkPolicy(TraceBank.from_call_log(log), fork_step=4, option_model="weak", base_model="strong", horizon=3)
    decisions, state = _drive(policy, "fix the bug", 10)
    assert [d.replay for d in decisions[:4]] == ["a0", "a1", "a2", "a3"]
    assert [d.model for d in decisions[4:7]] == ["weak"] * 3
    assert [d.model for d in decisions[7:]] == ["strong"] * 3
    assert state.switches == 2 and state.diverged_at is None


def test_fork_stops_replay_on_divergence(tmp_path):
    log = tmp_path / "calls.jsonl"
    _write_trace(log, "fix the bug", 12)
    policy = ForkPolicy(TraceBank.from_call_log(log), fork_step=6, option_model="weak", base_model="strong", horizon=None)
    decisions, state = _drive(policy, "fix the bug", 9, perturb_at=2)
    assert [d.note for d in decisions[:2]] == ["replay", "replay"]
    assert state.diverged_at == 2
    assert all(d.model == "weak" and d.replay is None for d in decisions[2:])


def test_fork_without_source_trace_uses_base(tmp_path):
    log = tmp_path / "calls.jsonl"
    _write_trace(log, "task A", 3)
    policy = ForkPolicy(TraceBank.from_call_log(log), fork_step=2, option_model="weak", base_model="strong")
    decisions, _ = _drive(policy, "task B", 2)
    assert [d.model for d in decisions] == ["strong", "strong"] and decisions[0].note == "no-source-trace"


def test_random_segment_respects_min_stay():
    p = RandomSegmentPolicy(["strong", "weak"], p_switch=1.0, min_stay=3, seed=1)
    decisions, _ = _drive(p, "t", 9)
    assert [d.model for d in decisions] == ["strong"] * 3 + ["weak"] * 3 + ["strong"] * 3


def test_build_policy_fixed():
    assert build_policy({"kind": "fixed", "model": "weak"}).decide(None, []).model == "weak"


def test_reprice_switch_costs_a_cache_write():
    from stayswitch.accounting import Call, reprice
    from stayswitch.pricing import PriceTable

    strong = Price.from_mapping({"input": 5.0, "output": 25.0, "cache_write_mult": 1.25, "cache_read_mult": 0.1})
    prices = PriceTable({"strong": strong, "weak": HAIKU})
    ctx = [60_000 + 2_000 * i for i in range(20)]
    stay = [Call("strong", c, 800, ts=float(i)) for i, c in enumerate(ctx)]
    dip = [Call("weak" if i == 10 else "strong", c, 800, ts=float(i)) for i, c in enumerate(ctx)]
    b_stay, b_dip = reprice(stay, prices), reprice(dip, prices)
    # Only the first call and the dip are cold: returning to strong reads its prefix up to the dip.
    assert b_dip.switches == 2 and b_dip.cold_calls == 2
    # Cache-oblivious pricing says a one-step dip saves money; with caching it costs more.
    assert b_dip.cache_oblivious < b_stay.cache_oblivious
    assert b_dip.cache_aware > b_stay.cache_aware
    # TTL expiry makes the next call cold even without a switch.
    gap = [Call("strong", 60_000, 800, ts=0.0), Call("strong", 62_000, 800, ts=400.0)]
    assert reprice(gap, prices).cold_calls == 2


def test_reprice_persistent_cache_vs_ttl():
    from stayswitch.accounting import Call, reprice
    from stayswitch.pricing import PriceTable

    prices = PriceTable({"weak": HAIKU})
    gap = [Call("weak", 60_000, 800, ts=0.0), Call("weak", 62_000, 800, ts=4000.0)]
    assert reprice(gap, prices, ttl_s=None).cold_calls == 1
    assert reprice(gap, prices, ttl_s=300).cold_calls == 2


def test_container_host_normalisation():
    from stayswitch.session import fallback_session_id, task_key

    a = [{"role": "user", "content": "Task: x\nCurrent Terminal Screen:\nroot@d598cdcdc569:/app#"}]
    b = [{"role": "user", "content": "Task: x\nCurrent Terminal Screen:\nroot@0123456789ab:/app#"}]
    assert task_key(a) == task_key(b) and messages_hash(a) == messages_hash(b)
    assert fallback_session_id(a) != fallback_session_id(b)
    # Later calls of the same trial keep its session id.
    assert fallback_session_id(a + [{"role": "assistant", "content": "{}"}]) == fallback_session_id(a)


def test_budget_per_session_and_total(tmp_path):
    from stayswitch.budget import Budget, spent_in_logs

    log = tmp_path / "calls.jsonl"
    log.write_text(json.dumps({"cost": 1.5}) + "\n" + json.dumps({"cost": 0.5}) + "\n")
    assert spent_in_logs(str(tmp_path / "*.jsonl")) == pytest.approx(2.0)

    b = Budget(per_session_usd=1.0, total_usd=3.5, spent_before=2.0)
    assert b.exhausted("a") is None
    b.charge("a", 0.6)
    assert b.exhausted("a") is None
    b.charge("a", 0.5)  # a at 1.1, total 3.1
    assert "trajectory" in b.exhausted("a") and b.exhausted("b") is None
    b.charge("b", 0.5)  # total 3.6
    assert "total" in b.exhausted("b")
