"""Compaction telemetry (Phase 1 §3): ``CompactionFacts`` + the tracer hook.

``compact()`` now returns the message list paired with a ``CompactionFacts``
describing what happened, and ``Agent._maybe_compact`` feeds that, plus the
pre/post token counts it already computes, to ``Tracer.record_compaction`` — a
flat Event (``kind="compaction"``) and a span (``operation="compact"``) with
the eight ``carbon.compaction.*`` attributes. None of this changes the actual
compaction OUTPUT (the messages, the summary text) — it only observes it.
"""

from __future__ import annotations

from unittest.mock import patch

import harness.agent as agent_mod
from harness import compaction, events
from harness.observability import Tracer
from model import LLMResponse


def _tool_call(name: str, path: str, cid: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": cid, "function": {"name": name, "arguments": f'{{"path":"{path}"}}'}}
        ],
    }


def _filler(n: int) -> list[dict]:
    return [
        m
        for i in range(n)
        for m in (
            {"role": "user", "content": f"filler turn {i}"},
            {"role": "assistant", "content": f"acknowledged {i}"},
        )
    ]


# --- compact() returns facts matching what it actually cut -------------------


def test_compact_returns_facts_matching_the_middle_slice():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    with patch.object(compaction, "chat", return_value=LLMResponse(content="SUMMARY")):
        out, facts = compaction.compact(msgs, keep_head=2, keep_tail=2, strategy="summarize_middle")
    assert len(out) < len(msgs)
    assert facts.strategy == "summarize_middle"
    # middle = messages[head_end:tail_start] — everything but the kept head/tail.
    assert facts.middle_count == len(msgs) - 2 - 2
    # Production computes this as len(summary); pin it to the scripted content's
    # actual length rather than a bound that can never fail (was `>= 0`).
    assert facts.summary_chars == len("SUMMARY")
    # No tool calls in this fixture, and `summarize_middle` never truncates.
    assert facts.files_read == 0
    assert facts.files_modified == 0
    assert facts.truncated is False
    # Codex finding 1: a real summarizer call ran here, so its usage must be
    # carried (never a phantom None reserved for the no-call paths).
    assert facts.summarizer_usage is not None


def test_compact_facts_count_touched_files_and_flag_truncation():
    """The stateful strategy's file counts and truncation flag both come through."""
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        _tool_call("edit_file", "a.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "edited"},
        _tool_call("read_file", "b.py", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "..."},
        *_filler(8),
    ]
    with patch.object(compaction, "chat", return_value=LLMResponse(content="B" * 40_000)):
        _out, facts = compaction.compact(
            messages,
            keep_head=2,
            strategy="token_budget_checkpoint",
            summary_max_tokens=100,
            recent_token_reserve=5,
        )
    assert facts.files_modified == 1
    assert facts.files_read == 1
    assert facts.truncated is True
    assert facts.summarizer_usage is not None


def test_compact_no_op_returns_zeroed_facts():
    """A history too small to compact returns the same list, with honest zeros."""
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out, facts = compaction.compact(msgs, keep_head=2, keep_tail=2, strategy="summarize_middle")
    assert out is msgs
    assert facts.middle_count == 0
    assert facts.summary_chars == 0
    assert facts.truncated is False
    # No model call happened at all — never a phantom summarizer usage.
    assert facts.summarizer_usage is None


# --- Agent._maybe_compact wires facts through Tracer.record_compaction -------


def test_agent_compaction_emits_event_and_span_with_shrinking_tokens():
    def fake_chat(messages, **kwargs):
        first = messages[0] if messages else {}
        if first.get("role") == "system" and "summar" in first.get("content", "").lower():
            return LLMResponse(content="SUMMARY")
        return LLMResponse(content="ok")

    tracer = Tracer()
    with (
        patch.object(agent_mod, "chat", side_effect=fake_chat),
        patch.object(compaction, "chat", side_effect=fake_chat),
    ):
        a = agent_mod.Agent(context_limit=20, tracer=tracer)
        for i in range(8):
            a.send(f"a reasonably long message number {i} with some filler text")

    compaction_events = [e for e in tracer.events if e.kind == "compaction"]
    assert compaction_events, "no compaction Event was recorded"

    spans = [s for s in tracer.spans if s.operation == "compact"]
    assert spans, "no compaction span was recorded"
    # Early on, a checkpoint note can cost more than the sliver of history it replaces
    # (there just isn't much middle yet) — the genuinely shrinking case this asserts on
    # is the one that shows up once real history has piled up, not necessarily the
    # first compaction of the run.
    shrinking = [
        s
        for s in spans
        if s.attributes[events.COMPACTION_PRE_TOKENS] > s.attributes[events.COMPACTION_POST_TOKENS]
    ]
    assert shrinking, f"no compaction shrank the window: {[s.attributes for s in spans]}"
    attrs = shrinking[-1].attributes
    idx = spans.index(shrinking[-1])
    # Amendment (2026-08-19, audit finding 1): the flat Event books tokens=0 for
    # every compaction — the pre/post delta lives only in the span attributes
    # above, so totals()["tokens"] never absorbs it.
    assert compaction_events[idx].tokens == 0
    assert attrs[events.COMPACTION_STRATEGY] == compaction_events[idx].label


def test_totals_tokens_stays_pure_llm_spend_through_a_compacting_run():
    """Audit finding 1: totals()["tokens"] must equal exactly the sum of
    provider-reported LLM spend, even in a run that triggers compaction — a
    compaction Event must never inflate (or deflate) the flat token total.

    Codex finding 1 (amended here): the summarizer's own call is ITSELF traced
    LLM spend now (scripted with its own distinct usage below) — the exact-sum
    invariant must hold with it included, never doubled, never dropped."""

    def fake_chat(messages, **kwargs):
        first = messages[0] if messages else {}
        if first.get("role") == "system" and "summar" in first.get("content", "").lower():
            return LLMResponse(
                content="SUMMARY",
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            )
        return LLMResponse(
            content="ok",
            usage={"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
        )

    tracer = Tracer()
    with (
        patch.object(agent_mod, "chat", side_effect=fake_chat),
        patch.object(compaction, "chat", side_effect=fake_chat),
    ):
        a = agent_mod.Agent(context_limit=20, tracer=tracer)
        for i in range(8):
            a.send(f"a reasonably long message number {i} with some filler text")

    compaction_events = [e for e in tracer.events if e.kind == "compaction"]
    assert compaction_events, "no compaction Event was recorded"
    llm_events = [e for e in tracer.events if e.kind == "llm"]
    # Every llm Event reports either exactly 50 (a turn call) or exactly 25 (a
    # summarizer call) total_tokens — the sum is a known, exact number derived
    # from the two scripted usages, unaffected by how many compactions ran.
    main_events = [e for e in llm_events if e.tokens == 50]
    summarizer_events = [e for e in llm_events if e.tokens == 25]
    assert len(main_events) + len(summarizer_events) == len(llm_events)
    assert summarizer_events, "the summarizer's call was never traced"
    expected = 50 * len(main_events) + 25 * len(summarizer_events)
    assert sum(e.tokens for e in llm_events) == expected
    assert tracer.totals()["tokens"] == expected
    assert tracer.totals()["llm_calls"] == len(llm_events)


def test_maybe_compact_pre_tokens_is_the_message_estimate_not_provider_usage():
    """Audit finding 3: pre (and post) must both be estimate_tokens(self.messages)
    computed immediately before/after compaction — never `window`, which can be
    `_last_tokens` (a provider-reported total over the full payload, including
    the system prompt). Scripts a provider total wildly larger than the real
    message-list estimate so the two are unmistakably different quantities: if
    `pre` ever reads `_last_tokens` instead, this test catches it.
    """
    from harness.compaction import estimate_tokens

    orig_compact = agent_mod.compact
    captured: list[tuple[int, bool]] = []

    def spying_compact(messages, **kwargs):
        pre_estimate = estimate_tokens(messages)
        out, facts = orig_compact(messages, **kwargs)
        captured.append((pre_estimate, out is not messages))
        return out, facts

    def fake_chat(messages, **kwargs):
        first = messages[0] if messages else {}
        if first.get("role") == "system" and "summar" in first.get("content", "").lower():
            return LLMResponse(content="SUMMARY")
        # A provider-reported total deliberately far larger than the true
        # message-list estimate for this tiny fixture.
        return LLMResponse(content="ok", usage={"total_tokens": 999_999})

    tracer = Tracer()
    with (
        patch.object(agent_mod, "chat", side_effect=fake_chat),
        patch.object(agent_mod, "compact", side_effect=spying_compact),
        patch.object(compaction, "chat", side_effect=fake_chat),
    ):
        a = agent_mod.Agent(context_limit=20, tracer=tracer)
        for i in range(8):
            a.send(f"a reasonably long message number {i} with some filler text")

    spans = [s for s in tracer.spans if s.operation == "compact"]
    assert spans, "no compaction span was recorded"
    real_pres = [pre for pre, real in captured if real]
    assert len(real_pres) == len(spans)
    for span, expected_pre in zip(spans, real_pres, strict=True):
        assert span.attributes[events.COMPACTION_PRE_TOKENS] == expected_pre
        # The bug this guards against: pre pinned to the scripted provider total.
        assert span.attributes[events.COMPACTION_PRE_TOKENS] != 999_999


def test_agent_without_tracer_still_compacts_the_same_way():
    """No tracer, no crash, no change in behavior — the hook is additive."""

    def fake_chat(messages, **kwargs):
        first = messages[0] if messages else {}
        if first.get("role") == "system" and "summar" in first.get("content", "").lower():
            return LLMResponse(content="SUMMARY")
        return LLMResponse(content="ok")

    with (
        patch.object(agent_mod, "chat", side_effect=fake_chat),
        patch.object(compaction, "chat", side_effect=fake_chat),
    ):
        a = agent_mod.Agent(context_limit=20)
        for i in range(8):
            a.send(f"a reasonably long message number {i} with some filler text")

    assert any(str(m.get("content", "")).startswith("[summary") for m in a.messages)


# --- Codex finding 1: the summarizer's own call is traced, never phantom -----


def test_record_compaction_emits_the_summarizer_as_an_llm_event():
    """When CompactionFacts carries a real summarizer_usage, Agent._record_compaction
    (the shared seam both compact() call sites route through) feeds it to
    Tracer.record_llm exactly once, alongside the usual compaction Event/span."""
    tracer = Tracer()
    a = agent_mod.Agent(context_limit=20, tracer=tracer)
    facts = compaction.CompactionFacts(
        strategy="structured_checkpoint",
        middle_count=4,
        summary_chars=7,
        files_read=0,
        files_modified=0,
        truncated=False,
        summarizer_usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        summarizer_seconds=0.02,
    )
    a._record_compaction(facts, pre=200, post=100, seconds=0.05)

    llm_events = [e for e in tracer.events if e.kind == "llm"]
    assert len(llm_events) == 1
    assert llm_events[0].tokens == 14
    compaction_events = [e for e in tracer.events if e.kind == "compaction"]
    assert len(compaction_events) == 1
    totals = tracer.totals()
    assert totals["llm_calls"] == 1
    assert totals["tokens"] == 14


def test_record_compaction_does_not_fabricate_an_llm_event_when_summarizer_was_skipped():
    """Strategies/paths that never call the summarizer (identity/deterministic
    no-op compactions, and an incremental strategy's 'nothing new to fold in'
    carry-forward) carry `summarizer_usage=None` — this must record ONLY the
    compaction Event/span, never a phantom llm one."""
    tracer = Tracer()
    a = agent_mod.Agent(context_limit=20, tracer=tracer)
    facts = compaction.CompactionFacts(
        strategy="token_budget_checkpoint",
        middle_count=0,
        summary_chars=3,
        files_read=0,
        files_modified=0,
        truncated=False,
        summarizer_usage=None,
    )
    a._record_compaction(facts, pre=100, post=100, seconds=0.01)

    assert [e.kind for e in tracer.events] == ["compaction"]
    assert tracer.totals()["llm_calls"] == 0
    assert tracer.totals()["tokens"] == 0
