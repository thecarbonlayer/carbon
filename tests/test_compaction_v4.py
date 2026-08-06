"""Compaction v4 — token-budgeted, turn-aware, incremental, deterministically stateful.

Each test here pins one of the five mechanisms that separate ``token_budget_checkpoint``
from the message-count strategies, and each is written so that it fails if that
mechanism is removed rather than merely if the output changes shape.
"""

from __future__ import annotations

from unittest.mock import patch

from harness import checkpoint, compaction
from model import LLMResponse


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
    """A path is tracked because a tool was CALLED with it, not because it was mentioned."""
    messages = [
        {"role": "user", "content": "please edit services/ghost.py"},  # prose only
        _tool_call("read_file", "a.py", "1"),
        {"role": "tool", "tool_call_id": "1", "content": "..."},
        _tool_call("write_file", "b.py", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "error: disk full"},
    ]
    ops = compaction.checkpoint.file_ops(messages)
    assert ops.read == ("a.py",)
    # Tracked despite the tool FAILING: what the agent was working on is still state.
    assert ops.modified == ("b.py",)
    assert "services/ghost.py" not in ops.read + ops.modified


def test_file_ops_survives_malformed_arguments():
    """One broken tool call must not cost the whole checkpoint."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "read_file", "arguments": "{not json"}},
                {"id": "2", "function": {"name": "read_file", "arguments": '{"path":"ok.py"}'}},
            ],
        }
    ]
    assert checkpoint.file_ops(messages).read == ("ok.py",)


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
            out = compaction.compact(
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
        out = compaction.compact(
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
    with patch.object(compaction, "chat", side_effect=_reply()):
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
        out = compaction.compact(
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
        once = compaction.compact(
            first, keep_head=2, strategy="token_budget_checkpoint", recent_token_reserve=50
        )
    # A second round whose own middle touches a DIFFERENT file.
    second = once + [
        _tool_call("edit_file", "late.py", "2"),
        {"role": "tool", "tool_call_id": "2", "content": "edited"},
        *_filler(8, start=100),
    ]
    with patch.object(compaction, "chat", side_effect=_reply("second")):
        twice = compaction.compact(
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
        out = compaction.compact(
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
        out = compaction.compact(
            messages,
            keep_head=2,
            strategy="token_budget_checkpoint",
            recent_token_reserve=TINY_RESERVE,
        )
    assert not calls
    note = next(m["content"] for m in out if str(m.get("content", "")).startswith("[summary"))
    assert "KEEP-ME-7" in note, "the carried-forward checkpoint lost its content"
    assert note.count("[summary of earlier conversation") == 1, "note header was nested"


def test_unknown_strategy_is_refused():
    try:
        compaction.compact([{"role": "user", "content": "x"}] * 10, strategy="invent_something")
    except ValueError as exc:
        assert "invent_something" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unregistered strategy was accepted")
