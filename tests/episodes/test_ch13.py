"""ch-13 — Observability.

Capability: a tracer records model calls (with tokens/latency) and tool calls.
Going deeper, the trace captures tool arguments and results, so a run is
replayable — the interesting bug is usually a few tool calls before the failure.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness.observability import Tracer
from harness.tools import ToolRegistry, read_file_tool
from model import LLMResponse


def test_tracer_totals_and_timeline():
    tr = Tracer()
    tr.record_llm({"total_tokens": 10}, 0.5)
    tr.record_tool("calculator", 0.001)
    totals = tr.totals()
    assert totals["llm_calls"] == 1 and totals["tool_calls"] == 1 and totals["tokens"] == 10
    assert "calculator" in tr.timeline()


def test_agent_records_a_trace():
    tr = Tracer()
    with patch.object(
        agent_mod, "chat", return_value=LLMResponse(content="ok", usage={"total_tokens": 7})
    ):
        agent_mod.Agent(tracer=tr).send("hi")
    totals = tr.totals()
    assert totals["llm_calls"] == 1 and totals["tokens"] == 7


# --- observability depth: tool args + results -------------------------------
def test_event_captures_args_and_result():
    tr = Tracer()
    tr.record_tool("calculator", 0.001, args='{"expression": "2+2"}', result="4")
    e = tr.events[0]
    assert e.args and e.result == "4"
    assert "expression" in tr.timeline() and "-> 4" in tr.timeline()


def test_agent_trace_records_tool_io(tmp_path):
    (tmp_path / "note.txt").write_text("42")
    replies = iter(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "function": {"name": "read_file", "arguments": '{"path": "note.txt"}'},
                    }
                ],
            ),
            LLMResponse(content="done"),
        ]
    )
    tools = ToolRegistry()
    tools.register(read_file_tool(root=tmp_path))
    tr = Tracer()
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        a = agent_mod.Agent(tools=tools, tracer=tr)
        a.send("compute it")

    tool_events = [e for e in tr.events if e.kind == "tool"]
    assert tool_events
    assert "note.txt" in tool_events[0].args
    assert tool_events[0].result == "42"
