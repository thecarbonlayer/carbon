"""Compaction v4 — token-budgeted, turn-aware, incremental, deterministically stateful.

Each test here pins one of the five mechanisms that separate ``token_budget_checkpoint``
from the message-count strategies, and each is written so that it fails if that
mechanism is removed rather than merely if the output changes shape.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from harness import checkpoint, compaction
from harness.harness_config import CONFIG
from model import LLMResponse, Provider


def _reply(text: str = "CHECKPOINT"):
    def summarize(payload, **kwargs):
        _reply.captured = {"payload": payload, "kwargs": kwargs}  # type: ignore[attr-defined]
        return LLMResponse(content=text)

    return summarize


def _tool_call(name: str, path: str, cid: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": cid, "function": {"name": name, "arguments": f'{{"path":"{path}"}}'}}
        ],
    }


def _filler(n: int, start: int = 0) -> list[dict]:
    """``n`` complete user/assistant turns, to give the budget a middle to cut into."""
    return [
        m
        for i in range(start, start + n)
        for m in (
            {"role": "user", "content": f"filler turn {i}"},
            {"role": "assistant", "content": f"acknowledged {i}"},
        )
    ]


# A reserve small enough that only the last turn or two is "recent", so the middle is
# unambiguously non-empty. The tests are about WHERE the cut lands, not how big.
TINY_RESERVE = 5


# --- checkpoint.py: the state that must never be paraphrased ----------------------


def test_file_ops_reads_tool_calls_not_prose():
    """A path is tracked because a tool was called AND succeeded, not because it
    was mentioned in prose."""
    messages = [
        {"role": "user", "content": "please edit services/ghost.py"},  # prose only
        _tool_call("read_file", "a.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "..."},
        _tool_call("write_file", "b.py", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "error: disk full"},
    ]
    ops = compaction.checkpoint.file_ops(messages)
    assert ops.read == ("a.py",)
    # NOT tracked: the write FAILED, so nothing on disk actually changed.
    assert ops.modified == ()
    assert "services/ghost.py" not in ops.read + ops.modified
    assert "b.py" not in ops.read + ops.modified


def test_file_ops_survives_malformed_arguments():
    """One broken tool call must not cost the whole checkpoint."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "read_file", "arguments": "{not json"}},
                {"id": "2", "function": {"name": "read_file", "arguments": '{"path":"ok.py"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "..."},
        {"role": "tool", "tool_call_id": "2", "content": "..."},
    ]
    assert checkpoint.file_ops(messages).read == ("ok.py",)


def test_file_ops_excludes_a_failed_write_from_ops_and_the_rendered_note():
    """A denied or failed write must not appear as a modified file, in FileOps or
    in the rendered checkpoint note — nothing on disk actually changed, so the
    carried state must not claim otherwise."""
    messages = [
        _tool_call("read_file", "kept.py", "0"),
        {"role": "tool", "tool_call_id": "0", "content": "..."},
        _tool_call("write_file", "denied.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "error: permission denied"},
    ]
    ops = checkpoint.file_ops(messages)
    assert ops.modified == ()
    note = checkpoint.render(ops)
    assert "denied.py" not in note
    assert "kept.py" in note


def test_file_ops_includes_a_successful_write():
    """The positive case: a write whose result did not error IS tracked, in both
    FileOps and the rendered note."""
    messages = [
        _tool_call("write_file", "ok.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "written"},
    ]
    ops = checkpoint.file_ops(messages)
    assert ops.modified == ("ok.py",)
    assert "ok.py" in checkpoint.render(ops)


def test_file_ops_counts_an_error_then_retry_success_once():
    """A write that fails and is then retried at the same path, successfully,
    must appear exactly once — this is state ("this file is now modified"), not
    a log of every attempt."""
    messages = [
        _tool_call("write_file", "retry.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "error: locked"},
        _tool_call("write_file", "retry.py", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "written"},
    ]
    ops = checkpoint.file_ops(messages)
    assert ops.modified == ("retry.py",)


def test_state_block_round_trips():
    ops = checkpoint.FileOps(read=("a.py", "b.py"), modified=("c.py",))
    assert checkpoint.parse(checkpoint.render(ops)) == ops


def test_empty_state_renders_nothing():
    assert checkpoint.render(checkpoint.FileOps()) == ""


def test_merge_preserves_first_seen_order_and_dedups():
    a = checkpoint.FileOps(read=("a.py", "b.py"))
    b = checkpoint.FileOps(read=("b.py", "c.py"), modified=("d.py",))
    merged = checkpoint.merge(a, b)
    assert merged.read == ("a.py", "b.py", "c.py")
    assert merged.modified == ("d.py",)


# --- the five v4 mechanisms -------------------------------------------------------


def test_cut_point_is_token_budgeted_not_message_counted():
    """The same message COUNT with different sizes must cut in different places.

    This is the whole claim of a token budget. Two histories identical in shape, one
    with a large message in the tail: the budgeted cut keeps fewer messages there,
    while a keep_tail count would keep exactly the same number in both.
    """

    def history(big: bool) -> list[dict]:
        # Filler sized so several turns fit inside the reserve, then one final turn
        # that is either negligible or larger than the whole reserve on its own.
        return [
            {"role": "user", "content": "head"},
            {"role": "assistant", "content": "ack"},
            *_filler(12),
            {"role": "user", "content": "final"},
            {"role": "assistant", "content": "X" * 3000 if big else "small"},
        ]

    kept = {}
    for big in (False, True):
        with patch.object(compaction, "chat", side_effect=_reply()):
            out, _facts = compaction.compact(
                history(big),
                keep_head=2,
                strategy="token_budget_checkpoint",
                recent_token_reserve=60,
            )
        kept[big] = len(out)
    assert kept[True] < kept[False], (
        "a large tail message did not shrink the kept tail — the cut is not token-budgeted"
    )


def test_cut_snaps_back_to_a_turn_boundary():
    """A cut must not land mid-turn, splitting an exchange across the summary."""
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "the turn that must stay whole"},
        {"role": "assistant", "content": "A" * 4000},
        {"role": "user", "content": "last"},
        {"role": "assistant", "content": "done"},
    ]
    with patch.object(compaction, "chat", side_effect=_reply()):
        out, _facts = compaction.compact(
            messages,
            keep_head=1,
            strategy="token_budget_checkpoint",
            recent_token_reserve=200,
        )
    # Whatever the budget chose, the first kept message after the note starts a turn.
    assert out[2]["role"] == "user", f"kept region begins mid-turn: {out[2]['role']}"


def test_previous_checkpoint_is_passed_as_its_own_message():
    """Incremental update: the prior checkpoint is context to carry forward, not
    transcript to re-compress. Concatenating the two is what erodes old facts."""
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        {"role": "system", "content": "[summary of earlier conversation]\nPRIOR-FACT-1"},
        *_filler(8),
    ]
    # The update instruction asserted below is the strategy's DEFAULT suffix; pin
    # that state rather than trusting the checked-in file, so a config that
    # legitimately sets compaction.prompt_suffix cannot fail this test.
    default_suffix = replace(CONFIG, compaction=replace(CONFIG.compaction, prompt_suffix=None))
    with (
        patch.object(compaction, "CONFIG", default_suffix),
        patch.object(compaction, "chat", side_effect=_reply()),
    ):
        compaction.compact(
            messages,
            keep_head=2,
            strategy="token_budget_checkpoint",
            recent_token_reserve=TINY_RESERVE,
        )
    payload = _reply.captured["payload"]  # type: ignore[attr-defined]
    prior = [m for m in payload if "PRIOR-FACT-1" in str(m.get("content", ""))]
    assert len(prior) == 1, "previous checkpoint was not passed to the summarizer"
    assert "Existing checkpoint to update" in prior[0]["content"], (
        "previous checkpoint was folded into the transcript instead of being kept separate"
    )
    assert "UPDATING an existing checkpoint" in payload[0]["content"]


def test_tracked_files_survive_a_summarizer_that_drops_everything():
    """The deterministic half of the checkpoint does not depend on the model.

    The summarizer here returns a single useless word. The file state must still be
    in the note, because it was extracted from tool calls and re-attached verbatim.
    """
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        _tool_call("edit_file", "services/ledger/reconcile.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "edited"},
        _tool_call("read_file", "schema.sql", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "..."},
        *_filler(8),
    ]
    with patch.object(compaction, "chat", side_effect=_reply("nothing")):
        out, _facts = compaction.compact(
            messages,
            keep_head=2,
            strategy="token_budget_checkpoint",
            recent_token_reserve=TINY_RESERVE,
        )
    note = next(m["content"] for m in out if str(m.get("content", "")).startswith("[summary"))
    assert "services/ledger/reconcile.py" in note
    assert "schema.sql" in note
    assert checkpoint.parse(note).modified == ("services/ledger/reconcile.py",)


def test_file_state_accumulates_across_repeated_compactions():
    """A file touched before the FIRST compaction is still listed after the second."""
    first = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        _tool_call("edit_file", "early.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "edited"},
        *_filler(8),
    ]
    with patch.object(compaction, "chat", side_effect=_reply("first")):
        once, _facts = compaction.compact(
            first, keep_head=2, strategy="token_budget_checkpoint", recent_token_reserve=50
        )
    # A second round whose own middle touches a DIFFERENT file.
    second = once + [
        _tool_call("edit_file", "late.py", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "edited"},
        *_filler(8, start=100),
    ]
    with patch.object(compaction, "chat", side_effect=_reply("second")):
        twice, _facts = compaction.compact(
            second, keep_head=2, strategy="token_budget_checkpoint", recent_token_reserve=50
        )
    note = next(m["content"] for m in twice if str(m.get("content", "")).startswith("[summary"))
    ops = checkpoint.parse(note)
    assert "early.py" in ops.modified, "state from before the first compaction was lost"
    assert "late.py" in ops.modified


def test_oversized_checkpoint_is_bounded_but_keeps_its_state_block():
    """The fallback clamps the prose and must not clamp the state off the end."""
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        _tool_call("write_file", "kept.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "written"},
        *_filler(8),
    ]
    with patch.object(compaction, "chat", side_effect=_reply("B" * 40_000)):
        out, _facts = compaction.compact(
            messages,
            keep_head=2,
            strategy="token_budget_checkpoint",
            summary_max_tokens=100,
            recent_token_reserve=TINY_RESERVE,
        )
    note = next(m["content"] for m in out if str(m.get("content", "")).startswith("[summary"))
    assert len(note) < 40_000, "an oversized checkpoint was not bounded"
    assert checkpoint.parse(note).modified == ("kept.py",), (
        "the fallback truncated the deterministic state block away"
    )


def test_empty_middle_carries_the_checkpoint_forward_without_calling_the_model():
    """Regression: a middle holding only the previous checkpoint must not be summarized.

    Found live, not in review. A repeat compaction that advances by less than one turn
    leaves nothing to fold in, and the resulting empty transcript made the local
    provider return a real 400 that killed the run mid-suite. Carrying the prior text
    forward is also the correct answer on the merits — re-summarizing a checkpoint
    against no new material can only lose from it.
    """
    prior = "[summary of earlier conversation; strategy=token_budget_checkpoint]\nKEEP-ME-7"
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ack"},
        {"role": "system", "content": prior},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "ok"},
    ]
    calls = []

    def explode(payload, **kwargs):  # pragma: no cover - must never run
        calls.append(payload)
        raise AssertionError("summarizer was called with an empty transcript")

    with patch.object(compaction, "chat", side_effect=explode):
        out, facts = compaction.compact(
            messages,
            keep_head=2,
            strategy="token_budget_checkpoint",
            recent_token_reserve=TINY_RESERVE,
        )
    assert not calls
    note = next(m["content"] for m in out if str(m.get("content", "")).startswith("[summary"))
    assert "KEEP-ME-7" in note, "the carried-forward checkpoint lost its content"
    assert note.count("[summary of earlier conversation") == 1, "note header was nested"
    # Codex finding 1: no model call happened here — the facts must say so, so a
    # caller feeding this to Tracer.record_llm never fabricates a phantom event.
    assert facts.summarizer_usage is None


def test_overflow_recovery_still_compacts_a_short_prefix_under_token_budgeting():
    """Regression: found live, not in review, in a full-suite delta run (H2 went
    1.0 -> 0.0 — a catastrophic regression, and the acceptance rule correctly
    refused the candidate over it).

    ``_compact_active_history()`` (the overflow-RECOVERY call, distinct from the
    steady-state pre-turn door) used to call ``compact()`` with no override, so it
    inherited whatever ``recent_token_reserve`` was configured. A prefix that is
    many messages but few TOKENS — exactly this fixture, and exactly what a real
    session can produce after a burst of short tool-call acknowledgements — fits
    entirely inside a real reserve, so the token-budgeted cut correctly finds no
    middle to summarize. Correct as a general policy; wrong for this one caller,
    whose job is to guarantee SOME reduction happens or the identical overflow
    recurs on retry with no path forward.

    The fix passes ``recent_token_reserve=0`` at this ONE call site, which by
    ``_token_budget_cut``'s own design falls back to the guaranteed-progress
    message-count cut — restoring the pre-v4 robustness property for this path
    while leaving steady-state compaction fully token-budgeted everywhere else.
    """
    state = {"main_calls": 0, "summary_calls": 0}

    def responder(messages, **kwargs):
        is_summary = bool(
            messages
            and messages[0].get("role") == "system"
            and "checkpoint" in str(messages[0].get("content", "")).lower()
        )
        if is_summary:
            state["summary_calls"] += 1
            return LLMResponse(content="CHECKPOINT")
        state["main_calls"] += 1
        if state["main_calls"] == 1:
            raise RuntimeError("maximum context length exceeded")
        return LLMResponse(content="OVERFLOW-RECOVERED")

    provider = Provider("fake://overflow", "fake", responder=responder)
    budgeted = replace(
        CONFIG,
        compaction=replace(
            CONFIG.compaction, strategy="token_budget_checkpoint", recent_token_reserve=200
        ),
    )
    from harness.agent import Agent

    # Patched in BOTH modules' namespaces: `from harness.harness_config import CONFIG`
    # binds a name in each importing module, so patching the origin alone would not
    # reach either module's already-bound reference.
    with patch("harness.compaction.CONFIG", budgeted), patch("harness.agent.CONFIG", budgeted):
        agent = Agent(provider=provider)
        # Many small messages, few tokens — the exact shape that made compaction a
        # silent no-op: comfortably under a 200-token reserve regardless of count.
        agent.messages = [{"role": "user", "content": f"old-{i}"} for i in range(10)]
        result = agent.run("continue")

    assert result.text == "OVERFLOW-RECOVERED", (
        f"overflow was not recovered under token_budget_checkpoint; "
        f"calls={state} reply={result.text!r}"
    )
    assert state == {"main_calls": 2, "summary_calls": 1}
    assert agent.compaction_count == 1


def test_unknown_strategy_is_refused():
    try:
        compaction.compact([{"role": "user", "content": "x"}] * 10, strategy="invent_something")
    except ValueError as exc:
        assert "invent_something" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unregistered strategy was accepted")
