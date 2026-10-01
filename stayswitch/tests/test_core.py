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


# ---- cache semantics / ledger / cost model ----

OPUS = Price.from_mapping({"input": 5.0, "output": 25.0, "cache_write_mult": 1.25, "cache_read_mult": 0.1})


def test_ledger_prefix_ttl_shrink_and_cross_model():
    from stayswitch.cache import ANTHROPIC, CacheLedger, CacheSemantics

    persistent, ttl = CacheSemantics(ttl_s=None), CacheSemantics(ttl_s=300)
    led = CacheLedger()
    led.record("strong", 10_000, ts=0.0)
    assert led.cached_prefix("strong", 12_000, 100.0, persistent) == 10_000
    assert led.cached_prefix("weak", 12_000, 100.0, persistent) == 0            # another model's cache
    assert led.cached_prefix("weak", 12_000, 100.0, CacheSemantics(ttl_s=None, cross_model=True)) == 10_000
    assert led.cached_prefix("strong", 12_000, 400.0, ttl) == 0                # expired
    assert led.cached_prefix("strong", 8_000, 100.0, persistent) == 0          # prompt shrank: history rewritten
    led.record("weak", 12_000, ts=1.0)
    assert led.cached_prefix("strong", 14_000, 2.0, persistent) == 10_000      # strong's prefix survived the switch
    led.record("strong", 6_000, ts=3.0)                                       # a summary shortened the prompt
    assert led.entries == {"strong": (6_000, 3.0)}
    assert CacheLedger({"m": (500, 0.0)}).cached_prefix("m", 2_000, 1.0, ANTHROPIC) == 0  # below min_prefix


def test_semantics_price_override_and_config():
    from stayswitch.cache import TINKER, CacheSemantics, semantics_from_config

    p = TINKER.price(HAIKU)
    assert (p.cache_read, p.cache_write) == pytest.approx((0.2, 1.0))
    assert CacheSemantics().price(HAIKU) == HAIKU
    s = semantics_from_config({"preset": "anthropic", "ttl_s": 3600})
    assert s.ttl_s == 3600 and s.read_mult == 0.1 and s.min_prefix == 1024
    assert semantics_from_config(None).name == "table"


def test_reprice_goes_cold_after_compaction_or_summary():
    from stayswitch.accounting import Call, reprice
    from stayswitch.cache import OBLIVIOUS
    from stayswitch.pricing import PriceTable

    prices = PriceTable({"s": OPUS})
    steady = [Call("s", 60_000 + 2_000 * i, 800, ts=float(i)) for i in range(6)]
    flagged = [*steady[:3], Call("s", 66_000, 800, ts=3.0, prefix_reset=True), *steady[4:]]
    shrunk = [*steady[:3], Call("s", 20_000, 800, ts=3.0), *steady[4:]]
    assert reprice(steady, prices).cold_calls == 1
    assert reprice(flagged, prices).cold_calls == 2
    assert reprice(shrunk, prices).cold_calls == 2
    assert reprice(flagged, prices).cache_aware > reprice(steady, prices).cache_aware
    b = reprice(steady, prices, OBLIVIOUS)
    assert b.cache_aware == pytest.approx(b.cache_oblivious)


def test_call_from_record():
    from stayswitch.accounting import Call

    rec = {"model": "s", "usage": {"fresh_input": 100, "cache_read": 900, "cache_write": 0, "output": 50},
           "ts": 1.5, "ctx": {"compacted": True}}
    assert Call.from_record(rec) == Call("s", 1000, 50, 1.5, True)


def test_breakeven_reproduces_observation_a_and_its_correction():
    from stayswitch.cache import ANTHROPIC, OBLIVIOUS, CacheSemantics
    from stayswitch.costmodel import CostModel
    from stayswitch.pricing import PriceTable

    prices = PriceTable({"opus": OPUS, "haiku": HAIKU})
    kw = dict(ctx_tokens=60_000, new_tokens=2_000, output_tokens=800)
    cm = CostModel(prices, CacheSemantics(ttl_s=300))
    # New tokens are charged at the cache-write price (they are written for the next call), so a step
    # costs a little more than the proposal's table, which priced them as plain input.
    assert cm.step_cost("opus", **kw) == pytest.approx(0.0625)
    assert cm.step_cost("haiku", **kw) == pytest.approx(0.0125)
    # Permanent downgrade: the cold Haiku call repays itself in about 1.4 Haiku steps (proposal table, 60k row).
    assert cm.breakeven_steps("opus", "haiku", permanent=True, **kw) == pytest.approx(1.4, abs=0.1)
    # Dip and return inside the TTL: only the segment is rewritten (section 11's correction).
    within = cm.breakeven_steps("opus", "haiku", gap_s=10, **kw)
    assert 1.5 < within < 2.5
    # Return after the TTL: the whole prefix is rewritten, the original observation-A regime.
    expired = cm.breakeven_steps("opus", "haiku", gap_s=1000, **kw)
    assert expired > 8 and expired > within
    # Under cache-oblivious accounting a switch costs nothing, which is the accounting most papers use.
    assert CostModel(prices, OBLIVIOUS).breakeven_steps("opus", "haiku", **kw) == 0
    # Anthropic rules via the preset ignore the table's multipliers and apply their own.
    assert CostModel(prices, ANTHROPIC).price("haiku").cache_write == pytest.approx(1.25)
    # Switching to a model that is not cheaper never pays back.
    assert cm.breakeven_steps("haiku", "opus", **kw) == float("inf")


def test_session_store_observe_feeds_the_ledger():
    from stayswitch.cache import CacheSemantics

    store = SessionStore()
    msgs = [{"role": "user", "content": "t"}]
    st = store.get("s", msgs)
    store.advance(st, "strong"); store.observe("s", "strong", 10_000)
    store.advance(st, "weak"); store.observe("s", "weak", 12_000)
    sem = CacheSemantics(ttl_s=None)
    assert st.cache.cached_prefix("strong", 14_000, None, sem) == 10_000
    assert st.cache.cached_prefix("weak", 14_000, None, sem) == 12_000
    store.advance(st, "weak"); store.observe("s", "weak", 5_000, prefix_reset=True)
    assert st.cache.entries == {"weak": (5_000, st.last_call_ts)}


def _drive_cost(policy, turns, ctx_tokens, *, reset_at=None):
    """Like _drive_msgs, but feeds the session ledger: every call is observed at ``ctx_tokens`` on its model."""
    store = SessionStore()
    msgs = [{"role": "user", "content": "fix the bug"}]
    out = []
    for i, (keys, obs) in enumerate([(None, None)] + turns):
        if keys is not None:
            msgs = msgs + [_terminus_turn(keys), {"role": "user", "content": obs}]
        state = store.get("s", msgs)
        d = policy.decide(state, msgs)
        store.advance(state, d.model)
        store.observe("s", d.model, ctx_tokens, prefix_reset=(i == reset_at))
        out.append(d)
    return out, state


def test_cost_escalate_gates_the_trigger_by_price():
    from stayswitch.cache import TINKER
    from stayswitch.costmodel import CostModel
    from stayswitch.policy import CostAwareEscalatePolicy, build_policy
    from stayswitch.pricing import PriceTable

    prices = PriceTable({"w": HAIKU, "s": OPUS})
    cm = CostModel(prices, TINKER)
    failing = [("ls\n", "ok"), ("cat a\n", "Error: x"), ("cat b\n", "Error: y"), ("ls -l\n", "ok"), ("pwd\n", "ok")]

    def run(value_usd, ctx, **kw):
        p = CostAwareEscalatePolicy("w", "s", cm, value_usd=value_usd, remaining_steps=10, fail_streak=2, max_weak_steps=None, **kw)
        return _drive_cost(p, failing, ctx, **{k: v for k, v in kw.items() if k == "reset_at"})[0]

    # A generous value escalates exactly where trigger_escalate would.
    rich = run(100.0, 20_000)
    assert [d.model for d in rich] == ["w", "w", "w", "s", "s", "s"] and rich[3].note.startswith("fail_streak cost=")
    # A stingy value never escalates, but records that the trigger fired and was deferred.
    poor = run(0.0, 20_000)
    assert [d.model for d in poor] == ["w"] * 6 and poor[3].note.startswith("fail_streak too_expensive=")
    # The gate is monotone in context size: the same value escalates at 5k tokens but not at 60k.
    p_small = CostAwareEscalatePolicy("w", "s", cm, value_usd=0.5, remaining_steps=10, fail_streak=2, max_weak_steps=None)
    p_large = CostAwareEscalatePolicy("w", "s", cm, value_usd=0.5, remaining_steps=10, fail_streak=2, max_weak_steps=None)
    assert _drive_cost(p_small, failing, 5_000)[0][3].model == "s"
    assert _drive_cost(p_large, failing, 60_000)[0][3].model == "w"


def test_cost_escalate_switch_is_cheaper_right_after_a_compaction():
    from stayswitch.cache import TINKER
    from stayswitch.costmodel import CostModel
    from stayswitch.pricing import PriceTable
    from stayswitch.policy import CostAwareEscalatePolicy

    cm = CostModel(PriceTable({"w": HAIKU, "s": OPUS}), TINKER)
    p = CostAwareEscalatePolicy("w", "s", cm, value_usd=10.0, remaining_steps=10, fail_streak=2, max_weak_steps=None)
    store = SessionStore()
    msgs = [{"role": "user", "content": "t"}, _terminus_turn("ls\n"), {"role": "user", "content": "Error: a"},
            _terminus_turn("ls\n"), {"role": "user", "content": "Error: b"}]
    st = store.get("s", msgs)
    store.advance(st, "w"); store.observe("s", "w", 40_000)
    warm_strong_absent = p.escalation_cost(st, msgs)          # strong never ran: its whole prefix is cold
    store.advance(st, "s"); store.observe("s", "s", 40_000)   # strong now holds the prefix
    assert p.escalation_cost(st, msgs) < warm_strong_absent
    # The per-step part is the same either way; the difference is exactly the cold-cache premium.
    from stayswitch.cache import CacheLedger
    from stayswitch.context import est_tokens
    cold = cm.switch_premium("s", CacheLedger(), ctx_tokens=40_000, new_tokens=est_tokens(msgs[-2:]), output_tokens=900)
    assert warm_strong_absent - p.escalation_cost(st, msgs) == pytest.approx(cold)


def test_build_policy_cost_escalate_requires_cost_model():
    from stayswitch.policy import build_policy
    from stayswitch.costmodel import CostModel
    from stayswitch.pricing import PriceTable

    cfg = {"kind": "cost_escalate", "weak": "w", "strong": "s", "value_usd": 1.0}
    with pytest.raises(ValueError):
        build_policy(cfg)
    assert build_policy(cfg, CostModel(PriceTable({"w": HAIKU, "s": OPUS}))).value_usd == 1.0


LEAD_TURNS = [
    ("grep -rn bug src\n", "src/a.py:3"),
    ("sed -i 's/x/y/' src/a.py\n", ""),
    ("python -m pytest -q\n", "1 passed"),  # edit then test: lead ends after this observation
    ("git diff\n", "diff"),
    ("python -m pytest -q\n", "1 failed"),
    ("python -m pytest -q -x\n", "2 failed"),  # two failing observations: rescue for 2 calls
    ("cat src/a.py\n", "ok"),
    ("ls\n", "ok"),
]


def _cost_lead(value_usd, remaining_steps, ctx=40_000):
    from stayswitch.cache import TINKER
    from stayswitch.costmodel import CostModel
    from stayswitch.policy import CostAwareStrongLeadPolicy
    from stayswitch.pricing import PriceTable

    cm = CostModel(PriceTable({"w": HAIKU, "s": OPUS}), TINKER)
    p = CostAwareStrongLeadPolicy("w", "s", cm, value_usd=value_usd, remaining_steps=remaining_steps, lead_max=50, fail_streak=2, commit=2)
    out, state = _drive_cost(p, LEAD_TURNS, ctx)
    return p, cm, out, state


def test_cost_lead_matches_strong_lead_when_money_is_no_object():
    _, _, out, _ = _cost_lead(value_usd=100.0, remaining_steps=1000)
    assert [d.model for d in out] == ["s", "s", "s", "w", "w", "w", "s", "s", "w"]
    assert [d.note for d in out][3:] == ["follow", "follow", "follow", "rescue", "rescue", "follow"]


def test_cost_lead_defers_a_rescue_it_cannot_afford():
    _, _, out, state = _cost_lead(value_usd=0.0, remaining_steps=1000)
    assert [d.model for d in out] == ["s", "s", "s"] + ["w"] * 6
    assert out[6].note == "rescue_deferred" and state.extra.get("rescues", 0) == 0


def test_cost_lead_holds_the_strong_model_when_a_handoff_cannot_pay_back():
    _, _, out, state = _cost_lead(value_usd=100.0, remaining_steps=0)
    assert [d.model for d in out] == ["s"] * 9
    assert out[3].note == "hold" and state.extra["holds"] == 4  # steps 3-5 and 8; 6-7 are the rescue
    # Trouble while holding needs no rescue: the strong model is already there.
    assert state.extra.get("rescues", 0) == 1 and out[6].note == "rescue"


def test_cost_lead_rescue_is_cheap_because_the_lead_prefix_is_still_cached():
    from stayswitch.cache import CacheLedger

    p, cm, out, state = _cost_lead(value_usd=100.0, remaining_steps=1000)
    from stayswitch.context import est_tokens

    keys, obs = LEAD_TURNS[5]  # the last turn before the rescue decision
    kw = dict(ctx_tokens=40_000, new_tokens=est_tokens([_terminus_turn(keys), {"role": "user", "content": obs}]), output_tokens=900)
    per_step = cm.step_cost("s", **kw) - cm.step_cost("w", **kw)
    # The strong model's prefix survived the weak segment (persistent cache): no premium, only the committed steps.
    assert state.extra["rescue_cost"] == pytest.approx(p.commit * per_step)
    cold = cm.switch_premium("s", CacheLedger(), **kw) + p.commit * per_step
    assert cold > state.extra["rescue_cost"] * 2
    # And the handoff break-even on Tinker prices is about a step, which is why the base heuristic works there.
    assert state.extra["handoff_breakeven"] < 2


def test_build_policy_cost_lead():
    from stayswitch.costmodel import CostModel
    from stayswitch.policy import CostAwareStrongLeadPolicy, build_policy
    from stayswitch.pricing import PriceTable

    cfg = {"kind": "cost_lead", "weak": "w", "strong": "s", "value_usd": 0.5, "commit": 4}
    with pytest.raises(ValueError):
        build_policy(cfg)
    p = build_policy(cfg, CostModel(PriceTable({"w": HAIKU, "s": OPUS})))
    assert isinstance(p, CostAwareStrongLeadPolicy) and p.commit == 4 and p.value_usd == 0.5
