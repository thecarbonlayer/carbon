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
    assert facts.summary_chars >= 0
    # No tool calls in this fixture, and `summarize_middle` never truncates.
    assert facts.files_read == 0
    assert facts.files_modified == 0
    assert facts.truncated is False


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


def test_compact_no_op_returns_zeroed_facts():
    """A history too small to compact returns the same list, with honest zeros."""
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out, facts = compaction.compact(msgs, keep_head=2, keep_tail=2, strategy="summarize_middle")
    assert out is msgs
    assert facts.middle_count == 0
    assert facts.summary_chars == 0
    assert facts.truncated is False


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
    # The flat Event's tokens field is the same pre-post delta, not clamped to >= 0.
    assert compaction_events[idx].tokens == (
        attrs[events.COMPACTION_PRE_TOKENS] - attrs[events.COMPACTION_POST_TOKENS]
    )
    assert attrs[events.COMPACTION_STRATEGY] == compaction_events[idx].label


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
