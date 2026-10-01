import json

import pytest

from stayswitch.policy import ForkPolicy, RandomSegmentPolicy, TraceBank, build_policy
from stayswitch.pricing import Price, Usage, cost, step_cost
from stayswitch.session import SessionStore, messages_hash, task_key

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


def _terminus_turn(keys: str) -> dict:
    return {"role": "assistant", "content": json.dumps({"analysis": "", "plan": "", "commands": [{"keystrokes": keys}]})}


def test_signals_step_kind_and_failures():
    from stayswitch.signals import observation_failed, repeated_action, step_kind

    assert step_kind("grep -rn foo src/\n") == "explore"
    assert step_kind("sed -i 's/a/b/' x.py\n") == "edit"
    assert step_kind("python -m pytest tests/test_x.py\n") == "test"
    msgs = [{"role": "user", "content": "task"}, _terminus_turn("ls\n"), {"role": "user", "content": "Traceback (most recent call last):"}]
    assert observation_failed(msgs)
    msgs2 = msgs + [_terminus_turn("ls\n"), {"role": "user", "content": "ok"}]
    assert repeated_action(msgs2) and not observation_failed(msgs2)


def _drive_msgs(policy, turns):
    """turns: list of (keystrokes of the previous assistant turn, observation)."""
    store = SessionStore()
    msgs = [{"role": "user", "content": "fix the bug"}]
    out = []
    for keys, obs in [(None, None)] + turns:
        if keys is not None:
            msgs = msgs + [_terminus_turn(keys), {"role": "user", "content": obs}]
        state = store.get("s", msgs)
        d = policy.decide(state, msgs)
        store.advance(state, d.model)
        out.append(d.model)
    return out


def test_weak_first_and_trigger_escalate():
    from stayswitch.policy import TriggerEscalatePolicy, WeakFirstPolicy

    assert _drive_msgs(WeakFirstPolicy("w", "s", 2), [("ls\n", "ok")] * 3) == ["w", "w", "s", "s"]
    p = TriggerEscalatePolicy("w", "s", fail_streak=2, max_weak_steps=None)
    got = _drive_msgs(p, [("ls\n", "ok"), ("cat a\n", "Error: x"), ("cat b\n", "Error: y"), ("ls -l\n", "ok")])
    assert got == ["w", "w", "w", "s", "s"]


def test_strong_lead_phases_and_rescue():
    from stayswitch.policy import StrongLeadPolicy

    p = StrongLeadPolicy("w", "s", lead_max=50, fail_streak=2, commit=2)
    turns = [
        ("grep -rn bug src\n", "src/a.py:3"),
        ("sed -i 's/x/y/' src/a.py\n", ""),
        ("python -m pytest -q\n", "1 passed"),  # edit then test: lead ends after this observation
        ("git diff\n", "diff"),
        ("python -m pytest -q\n", "1 failed"),
        ("python -m pytest -q -x\n", "2 failed"),  # two failing observations: rescue for 2 calls
        ("cat src/a.py\n", "ok"),
        ("ls\n", "ok"),
    ]
    assert _drive_msgs(p, turns) == ["s", "s", "s", "w", "w", "w", "s", "s", "w"]


def _conv_turns(n):
    msgs = [{"role": "user", "content": "task"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"o{i}"}]
    return msgs


def test_evict_chunked_keeps_prefix_stable_within_a_block():
    from stayswitch.context import ELISION_NOTE, evict

    assert evict(_conv_turns(5), keep=4, chunk=3) == (_conv_turns(5), 0)  # excess 1 < chunk
    sent, dropped = evict(_conv_turns(7), keep=4, chunk=3)  # excess 3 -> drop 3
    assert dropped == 3 and sent[0]["content"] == "task" + ELISION_NOTE and sent[1]["content"] == "a3"
    # Two more turns: still the same cut, so the sent prefix is unchanged (cache-friendly).
    sent9, d9 = evict(_conv_turns(9), keep=4, chunk=3)
    assert d9 == 3 and sent9[: len(sent)] == sent
    assert evict(_conv_turns(10), keep=4, chunk=3)[1] == 6


def test_evict_sliding_window_moves_every_turn():
    from stayswitch.context import evict

    assert [evict(_conv_turns(n), keep=4, chunk=1)[1] for n in (4, 5, 6)] == [0, 1, 2]
    sent, _ = evict(_conv_turns(6), keep=4, chunk=1)
    assert [m["content"] for m in sent if m["role"] == "assistant"] == ["a2", "a3", "a4", "a5"]


def _long_conv(n, obs_chars=3500):
    msgs = [{"role": "user", "content": "task " * 200}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"a{i} " * 50}, {"role": "user", "content": f"o{i}" + "x" * obs_chars}]
    return msgs


def test_compactor_truncates_once_at_budget_then_holds_the_cut():
    from stayswitch.context import Compactor, fixed_threshold

    c, x = Compactor(keep=3, threshold=fixed_threshold(9000)), {}
    sizes, cuts = [], []
    for n in range(1, 16):
        sent, info = c.view(_long_conv(n), x)
        sizes.append(info["est_tokens"]); cuts.append(info["cut"])
    assert max(sizes) < 9000 + 1200          # never far above the budget
    assert cuts == sorted(cuts)               # the cut only moves forward
    assert x["compactions"] >= 2 and len(set(cuts)) == x["compactions"] + 1
    # Between compactions the sent prefix is unchanged, so the cache prefix survives.
    first = cuts.index(cuts[-1])
    a, _ = c.view(_long_conv(first + 1), dict(x, cut=cuts[-1], prev_tokens=None))
    b, _ = c.view(_long_conv(first + 2), dict(x, cut=cuts[-1], prev_tokens=None))
    assert b[: len(a) - 1] == a[:-1]


def test_compactor_resets_when_harness_rewrites_history():
    from stayswitch.context import Compactor, fixed_threshold

    c, x = Compactor(keep=2, threshold=fixed_threshold(3000)), {}
    for n in range(1, 10):
        c.view(_long_conv(n), x)
    assert x["cut"] > 2
    sent, info = c.view(_long_conv(1), x)  # e.g. the agent's own summariser restarted the chat
    assert info["cut"] == 0 and len(sent) == 3


def test_eoq_threshold_scales_with_sqrt_of_growth_and_read_price():
    from stayswitch.context import eoq_threshold

    strong = Price.from_mapping({"input": 3.0, "output": 7.5, "cache_read": 0.6, "cache_write": 3.0})
    cheap_read = Price.from_mapping({"input": 3.0, "output": 7.5, "cache_read": 0.3, "cache_write": 3.0})
    t = eoq_threshold(strong, extra_steps=2, ceiling=10**9)
    l0 = 6000
    base = t({"growth": 1000}, 0, l0) - l0
    assert 9000 < base + l0 < 30000
    # sqrt(g) law, a little steeper because C itself includes the new input of a redone step
    assert 2.0 <= (t({"growth": 4000}, 0, l0) - l0) / base <= 2.6
    assert eoq_threshold(cheap_read, extra_steps=2, ceiling=10**9)({"growth": 1000}, 0, l0) > base + l0


def test_compactor_does_not_thrash_when_kept_turns_exceed_threshold():
    from stayswitch.context import Compactor, fixed_threshold

    c, x = Compactor(keep=2, threshold=fixed_threshold(100)), {}
    for n in range(1, 12):
        c.view(_long_conv(n), x)
    assert x["compactions"] <= 5  # not one per call


def test_token_counter_is_ignored_by_prompt_hash():
    a = [{"role": "system", "content": "<total_tokens>14982101 tokens left</total_tokens>"}]
    b = [{"role": "system", "content": "<total_tokens>15000000 tokens left</total_tokens>"}]
    assert messages_hash(a) == messages_hash(b)


def test_replay_server_rebuilds_tool_use_message():
    from stayswitch.replay_server import anthropic_message, sse_events

    recording = {
        "text": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}],
        "extra": {"thinking": "plan", "input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 5},
    }
    message = anthropic_message(recording, "m")
    assert [b["type"] for b in message["content"]] == ["thinking", "tool_use"]
    assert message["content"][1]["input"] == {"command": "ls"} and message["stop_reason"] == "tool_use"
    assert message["usage"]["cache_read_input_tokens"] == 5
    names = [name for name, _ in sse_events(message)]
    assert names[0] == "message_start" and names[-1] == "message_stop" and names.count("content_block_start") == 2


def test_fork_replays_recorded_tool_calls(tmp_path):
    log = tmp_path / "calls.jsonl"
    msgs = [{"role": "user", "content": "task"}]
    log.write_text(json.dumps({"session_id": "s", "task": task_key(msgs), "step": 0, "input_hash": messages_hash(msgs),
                               "response_text": "", "model": "strong", "tool_calls": [{"id": "c1"}],
                               "replay_extra": {"thinking": "t"}}) + "\n")
    policy = ForkPolicy(TraceBank.from_call_log(log), fork_step=1, option_model="weak", base_model="strong")
    decision = policy.decide(SessionStore().get("x", msgs), msgs)
    assert decision.replay_tool_calls == [{"id": "c1"}] and decision.replay_extra == {"thinking": "t"}


def _cc_conv(n, out_chars=3500, tool="Bash", args=None):
    """A Claude Code (Anthropic /v1/messages) conversation: task, system note, then n tool_use / tool_result turns."""
    args = args or {"command": "pytest -x"}
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>"}, {"type": "text", "text": "fix the bug"}]},
        {"role": "system", "content": "Available agent types"},
    ]
    for i in range(n):
        msgs += [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": ""},
                                              {"type": "tool_use", "id": f"call_{i}", "name": tool, "input": {**args, "n": i}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call_{i}", "content": f"o{i}" + "x" * out_chars}]},
            {"role": "system", "content": [{"type": "text", "text": f"<total_tokens>{1000 - i} tokens left</total_tokens>"}]},
        ]
    return msgs


def test_signals_read_claude_code_tool_calls_and_results():
    from stayswitch.signals import keystrokes, kinds, observation_failed, repeated_action

    msgs = _cc_conv(2)
    assert keystrokes(msgs[2]) == "pytest -x" and kinds(msgs) == ["test", "test"]
    assert kinds(_cc_conv(1, tool="Edit", args={"file_path": "a.py"})) == ["edit"]
    assert kinds(_cc_conv(1, tool="Grep", args={"pattern": "foo"})) == ["explore"]
    assert kinds(_cc_conv(1, tool="TodoWrite", args={})) == ["empty"]
    assert not observation_failed(msgs)  # trailing role="system" token counter must not hide the tool_result
    msgs[-2]["content"][0]["content"] = "Exit code 1\nFAILED test_x.py"
    assert observation_failed(msgs)
    msgs[-2]["content"][0].update(content="boom", is_error=True)
    assert observation_failed(msgs)
    assert repeated_action(_cc_conv(2))  # same command twice in a row
    other = _cc_conv(2)
    other[5]["content"][1]["input"]["command"] = "git diff"
    assert not repeated_action(other)


def test_compactor_on_anthropic_messages_keeps_tool_pairs_and_counts_tool_output():
    from stayswitch.context import Compactor, ELISION_NOTE, est_tokens, fixed_threshold, overhead_tokens

    assert est_tokens(_cc_conv(3)) > 3 * 3500 / 3.5  # tool_result text is counted
    c, x = Compactor(keep=3, threshold=fixed_threshold(12000)), {}
    for n in range(1, 16):
        sent, info = c.view(_cc_conv(n), x)
        uses = {b["id"] for m in sent if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"}
        results = {b["tool_use_id"] for m in sent if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"}
        assert results <= uses and uses == results or n == 0  # no orphaned tool_result, no unanswered tool_use
    assert x["compactions"] >= 1 and info["cut"] > 0
    assert sent[0]["role"] == "user" and sent[0]["content"][-1]["text"] == ELISION_NOTE  # note on the task turn, not the system note
    assert sent[1]["role"] == "system" and sent[2]["role"] == "assistant"
    # System prompt + tool schemas count toward the trigger.
    big = overhead_tokens([{"name": "t", "description": "d" * 35000}])
    _, info2 = Compactor(keep=3, threshold=fixed_threshold(12000)).view(_cc_conv(1), {}, big)
    assert info2["est_tokens"] >= big


def test_compactor_clips_long_tool_results():
    from stayswitch.context import Compactor, fixed_threshold

    sent, _ = Compactor(keep=3, threshold=fixed_threshold(10**9), max_msg_chars=1000).view(_cc_conv(2, out_chars=5000), {})
    result = sent[3]["content"][0]["content"]
    assert len(result) < 1200 and "characters omitted" in result


def test_agent_key_separates_subagents_and_ignores_billing_header():
    from stayswitch.session import agent_key, root_session_id

    tools = [{"name": "Bash"}, {"name": "Read"}]
    main_a = [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.260.8c5; cc_entrypoint=cli;"}, {"type": "text", "text": "You are an agent"}]
    main_b = [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.260.ffe; cc_entrypoint=cli;"}, {"type": "text", "text": "You are an agent"}]
    sub = [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.260.8c5; cc_entrypoint=cli;"}, {"type": "text", "text": "You are a search subagent"}]
    assert agent_key(main_a, tools) == agent_key(main_b, tools)
    assert agent_key(main_a, tools) != agent_key(sub, tools)
    assert agent_key(main_a, tools) != agent_key(main_a, tools[:1])
    assert root_session_id("abc~sub-1234") == "abc" and root_session_id("abc-summarization-x") == "abc" and root_session_id("abc") == "abc"


def test_eoq_v2_threshold_properties():
    from stayswitch.eoq_trigger import EOQTrigger, eoq_l_star

    price = Price.from_mapping({"input": 0.54, "output": 1.335, "cache_read": 0.108, "cache_write": 0.54})
    kw = dict(extra_steps=2.3, ceiling=10**9)
    base = eoq_l_star(price, l0=20000, head=18000, growth=800, **kw)
    assert base > 20000 + 4 * 800                                   # at least min_gap steps of room
    assert eoq_l_star(price, l0=20000, head=18000, growth=3200, **kw) > base          # sqrt(g) law
    assert eoq_l_star(price, l0=20000, head=0, growth=800, **kw) > base                # a smaller cached head means a dearer rewrite
    cheap_read = Price.from_mapping({"input": 0.54, "output": 1.335, "cache_read": 0.054, "cache_write": 0.54})
    assert eoq_l_star(cheap_read, l0=20000, head=18000, growth=800, **kw) > base        # cheaper reads -> compact later
    assert eoq_l_star(price, l0=20000, head=18000, growth=800, extra_steps=2.3, ceiling=21000) == 20000 + 4 * 800  # ceiling never beats the min gap

    t = EOQTrigger(price, extra_steps=2.3, keep_recent=3, ceiling=10**9)
    first, info = t.threshold("k", head=18000, mean_growth=800)
    assert not info["l0_observed"]
    for out in (21000, 21700, 22600):
        t.observe("k", est_out=out, compacted=False)
    assert abs(t.threshold("k", head=18000, mean_growth=800)[1]["growth"] - 800) < 150   # EMA of observed deltas
    t.observe("k", est_out=26000, compacted=True)
    after, info = t.threshold("k", head=18000, mean_growth=800)
    assert info["l0_observed"] and info["l0"] == 26000 and info["compactions"] == 1
    assert after > 26000                                                                  # next cycle starts from the measured size
