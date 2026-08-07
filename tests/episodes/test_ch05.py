"""ch-05 — Tools.

Capability: the agent runs a tool the model requests, feeds the result back,
and the model continues to a final answer. Folded in: the approval gate for
boundary-crossing tools, and file editing over a scoped workspace.
"""

from unittest.mock import patch

import pytest

import harness.agent as agent_mod
from harness.sandbox import Sandbox, bash_tool
from harness.tools import Tool, ToolRegistry, calculator, calculator_tool, default_tools
from harness.workspace import Workspace, edit_file_tool, write_file_tool
from model import LLMResponse


def test_calculator_tool():
    assert calculator("47 * 89") == "4183"
    assert calculator("2 ** 10") == "1024"


def test_tool_call_loop_executes_and_returns():
    replies = iter(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "function": {"name": "calculator", "arguments": '{"expression": "6 * 7"}'},
                    }
                ],
            ),
            LLMResponse(content="The answer is 42."),
        ]
    )

    def fake_chat(messages, **kwargs):
        return next(replies)

    tools = default_tools()
    tools.register(calculator_tool())
    with patch.object(agent_mod, "chat", side_effect=fake_chat):
        a = agent_mod.Agent(tools=tools)
        out = a.send("what is 6 * 7?")

    assert "42" in out
    # the tool result was recorded back into the conversation
    tool_msgs = [m for m in a.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "42"


# --- approval gate -----------------------------------------------------------
def _danger_registry(ran: list):
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="danger",
            description="a boundary-crossing action",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: (ran.append(1), "executed")[1],
        )
    )
    return reg


def _calls_danger_then_done():
    return iter(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "function": {"name": "danger", "arguments": "{}"}}],
            ),
            LLMResponse(content="done"),
        ]
    )


def _run_with(approve, approval_required):
    ran: list[int] = []
    replies = _calls_danger_then_done()
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        a = agent_mod.Agent(
            tools=_danger_registry(ran), approve=approve, approval_required=approval_required
        )
        a.send("do the danger")
    tool_msgs = [m for m in a.messages if m.get("role") == "tool"]
    return ran, tool_msgs


def test_denied_tool_does_not_execute():
    ran, tool_msgs = _run_with(approve=lambda n, a: False, approval_required={"danger"})
    assert ran == []
    assert any("[denied" in m["content"] for m in tool_msgs)


def test_approved_tool_executes():
    ran, tool_msgs = _run_with(approve=lambda n, a: True, approval_required={"danger"})
    assert ran == [1]
    assert any("executed" in m["content"] for m in tool_msgs)


def test_no_approver_fails_closed():
    ran, tool_msgs = _run_with(approve=None, approval_required={"danger"})
    assert ran == []
    assert any("[denied" in m["content"] for m in tool_msgs)


def test_ungated_tool_runs_freely():
    ran, tool_msgs = _run_with(approve=lambda n, a: False, approval_required=set())
    assert ran == [1]  # not in approval_required → not gated


# --- file editing / workspace ------------------------------------------------
def test_write_read_edit(tmp_path):
    ws = Workspace(root=tmp_path)
    assert "wrote" in ws.write("calc.py", "def add(a, b):\n    return a+b\n")
    assert "def add" in ws.read("calc.py")
    assert "edited" in ws.edit("calc.py", "a+b", "a + b")
    assert "a + b" in ws.read("calc.py")


def test_path_escape_blocked(tmp_path):
    ws = Workspace(root=tmp_path)
    with pytest.raises(ValueError):
        ws.write("../escape.py", "nope")


def test_tools_round_trip(tmp_path):
    ws = Workspace(root=tmp_path)
    write_file_tool(ws).func(path="a.txt", content="hello")
    assert ws.read("a.txt") == "hello"
    edit_file_tool(ws).func(path="a.txt", old="hello", new="world")
    assert ws.read("a.txt") == "world"


def test_bash_runs_in_workspace(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.write("hi.txt", "HELLO-WS")
    bash = bash_tool(Sandbox(prefer_docker=False), workdir=str(ws.root))
    assert "HELLO-WS" in bash.func(command="cat hi.txt")  # bash sees the written file
