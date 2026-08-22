"""ch-06 — Context management.

Capability: when history exceeds the budget, the harness compacts it (summarize
the middle, keep head + tail) so the agent stays coherent past the limit. Folded
in: door control — per-item size caps clamp oversized @file blocks and tool
results before they enter the prompt.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness import compaction
from harness.compaction import compact, estimate_tokens
from harness.context import deliver
from harness.harness_config import CONFIG
from harness.limits import MAX_ITEM_CHARS, clamp
from harness.tools import Tool, ToolRegistry
from model import LLMResponse
from tasks.checks import _ceiling


def test_estimate_tokens():
    assert estimate_tokens([{"role": "user", "content": "x" * 40}]) == 10


def test_compact_keeps_head_and_tail_and_summarizes_middle():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    with patch.object(compaction, "chat", return_value=LLMResponse(content="SUMMARY")):
        out, _facts = compact(msgs, keep_head=2, keep_tail=2)
    assert out[0] == msgs[0] and out[1] == msgs[1]
    assert out[-1] == msgs[-1] and out[-2] == msgs[-2]
    assert any("SUMMARY" in m["content"] for m in out)
    assert len(out) < len(msgs)


def test_agent_compacts_when_over_limit():
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


# --- door control: per-item size caps ----------------------------------------
def test_clamp_truncates_with_marker():
    out = clamp("A" * 10_000, max_chars=100)
    assert len(out) < 10_000
    assert "truncated" in out


def test_clamp_leaves_small_text():
    assert clamp("short") == "short"


def test_delivered_file_is_clamped(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("X" * (MAX_ITEM_CHARS * 3))
    (block,) = deliver(f"@{big} look")
    assert len(block) <= MAX_ITEM_CHARS + 100  # cap + marker headroom
    assert "truncated" in block


def test_tool_result_is_clamped():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="dump",
            description="returns a huge blob",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "Z" * (MAX_ITEM_CHARS * 3),
        )
    )
    replies = iter(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "function": {"name": "dump", "arguments": "{}"}}],
            ),
            LLMResponse(content="done"),
        ]
    )
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        a = agent_mod.Agent(tools=reg)
        a.send("dump it")
    tool_msg = next(m for m in a.messages if m.get("role") == "tool")
    # The allowance follows the SELECTED strategy — `offload_to_file` adds a footer
    # naming the complete copy, which is the point of it, not door-control slack. A
    # fixed `+ 100` read one legal menu choice as a regression. `tasks.checks` already
    # derives this for the live accept check; the offline test hardcoded it.
    assert len(tool_msg["content"]) <= _ceiling(CONFIG.tool_output)
    assert "truncated" in tool_msg["content"]
