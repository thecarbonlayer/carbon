"""ch-05 — Tools.

Capability: the agent runs a tool the model requests, feeds the result back,
and the model continues to a final answer. Folded in: the approval gate for
boundary-crossing tools, and file editing over a scoped workspace.
"""

import json
from unittest.mock import patch

import pytest

import harness.agent as agent_mod
from harness.sandbox import Sandbox, bash_tool
from harness.tools import Tool, ToolRegistry, read_file_tool
from harness.workspace import Workspace, edit_file_tool, write_file_tool
from model import LLMResponse
from tasks.checks import _build_workspace_agent


def test_tool_call_loop_executes_and_returns(tmp_path):
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
            LLMResponse(content="The answer is 42."),
        ]
    )

    def fake_chat(messages, **kwargs):
        return next(replies)

    tools = ToolRegistry()
    tools.register(read_file_tool(root=tmp_path))
    with patch.object(agent_mod, "chat", side_effect=fake_chat):
        a = agent_mod.Agent(tools=tools)
        out = a.send("what does note.txt say?")

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


def test_workspace_agents_bash_tool_reaches_its_own_scratch():
    """``_build_workspace_agent`` (tasks/checks.py) is shared by the ch-05 accept
    check AND its demo — proven here directly against that real production
    helper, not a hand-built stand-in, so a regression in the helper itself would
    be caught. No live model needed: only ``.send()`` (the model loop) is
    skipped — the bash tool the helper actually wires runs for real, through the
    registry, exactly as the model would call it."""
    a, ws = _build_workspace_agent()
    try:
        out = a.tools.call(
            "bash", json.dumps({"command": 'echo PROOF > "$CARBON_SCRATCH_DIR/probe.txt"'})
        )
        assert out.startswith("[exit 0"), out
        assert (a.session_env.scratch_root / "probe.txt").read_text().strip() == "PROOF"
    finally:
        a.close()


def test_workspace_agents_read_file_tool_also_reaches_its_own_scratch():
    """The matching half of the same footer the test above proves bash can
    resolve: ``_build_workspace_agent``'s registry must ALSO pass
    ``scratch_root=`` into ``default_tools()``, or a footer naming both routes
    (``scratch://...`` for read_file, ``$CARBON_SCRATCH_DIR/...`` for bash)
    would have only the bash half actually work. Stands in for a real spill by
    writing directly under the session's own scratch, then resolves it through
    the REGISTRY's read_file tool — never the bare function."""
    from harness.limits import spill_ref

    a, ws = _build_workspace_agent()
    try:
        (a.session_env.scratch_root / "offload").mkdir(parents=True)
        (a.session_env.scratch_root / "offload" / "probe.txt").write_text("SPILL-PROOF")
        result = a.tools.call("read_file", json.dumps({"path": spill_ref("probe.txt")}))
        assert result == "SPILL-PROOF"
    finally:
        a.close()


def test_build_workspace_agent_closes_the_agent_it_just_built_when_setup_raises(monkeypatch):
    """``_build_workspace_agent`` constructs the Agent (allocating its scratch in
    ``Agent.__init__``) BEFORE registering tools — the canonical order this task
    put it in, so the bash Sandbox can read ``a.session_env.scratch_root``. That
    reorder re-opened a leak window commit 0468747 closed elsewhere (``ui/tui.py``'s
    ``_build_agent``, ``harness/agent.py``'s ``run_once``/``_run_repl``): if tool
    registration raises, the Agent already built must still be closed, not leaked
    until the next process's ``scavenge()``. Same spy-on-close technique as
    ``test_build_agent_closes_the_agent_it_just_built_when_setup_raises``
    (tests/test_tui_streaming.py)."""
    import harness.agent as agent_mod
    import harness.tools as tools_mod

    closed: list[bool] = []
    real_close = agent_mod.Agent.close

    def spy_close(self):
        closed.append(True)
        return real_close(self)

    monkeypatch.setattr(agent_mod.Agent, "close", spy_close)
    monkeypatch.setattr(
        tools_mod,
        "default_tools",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tool-building blew up")),
    )

    try:
        _build_workspace_agent()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the tool-building failure to propagate")

    assert closed, "a raise while building tools must close the Agent already built"
