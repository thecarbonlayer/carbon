"""ch-12 — Verification.

Capability: when a turn changes code, the harness refuses "done" until it OBSERVES
a real passing run of the project's declared test command (AGENTS.md ## Testing) in
the tool transcript. No code change, or no declared command → no gate. Capped at
verify_attempts so a model that won't run the tests can't hang the loop.
"""

import json
from unittest.mock import patch

import harness.agent as agent_mod
from harness.sandbox import Sandbox, bash_tool
from harness.tools import default_tools
from harness.workspace import Workspace, write_file_tool
from model import LLMResponse
from tasks.checks import _build_ch12_agent

AGENTS = "## Testing\n```\npython3 test_thing.py\n```\n"
PASS = "print('ok')\n"
FAIL = "import sys\n\nsys.exit(1)\n"


def _agent(test_body: str, declare: bool = True):
    ws = Workspace()
    if declare:
        ws.write("AGENTS.md", AGENTS)
    ws.write("test_thing.py", test_body)
    tools = default_tools()
    tools.register(write_file_tool(ws))
    tools.register(bash_tool(Sandbox(trusted=True), workdir=str(ws.root)))
    a = agent_mod.Agent(tools=tools, agents_dir=str(ws.root), verify_attempts=2)
    return a, ws


def _call(cid: str, name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    )


def _pushbacks(a):
    return [m for m in a.messages if "passing run of the" in str(m.get("content", ""))]


def test_no_code_change_no_gate():
    """A pure Q&A turn writes nothing → no gate, even though a test command exists."""
    a, _ = _agent(PASS)
    with patch.object(agent_mod, "chat", return_value=LLMResponse(content="here you go")):
        out = a.send("what does is_prime do?")
    assert out == "here you go"
    assert not _pushbacks(a)


def test_accepts_after_a_passing_run():
    """Model changes code and runs the declared command; it exits 0 → accepted."""
    a, _ = _agent(PASS)
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            _call("c2", "bash", {"command": "python3 test_thing.py"}),
            LLMResponse(content="done"),
        ]
    )
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        out = a.send("write foo.py")
    assert out == "done"
    assert a._observed_pass("python3 test_thing.py", 0)
    assert not _pushbacks(a)


def test_pushes_back_when_code_changed_but_not_run():
    """Code changed, model narrates done without running → pushback, then it runs."""
    a, _ = _agent(PASS)
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            LLMResponse(content="done, looks right"),
            _call("c2", "bash", {"command": "python3 test_thing.py"}),
            LLMResponse(content="now it really passes"),
        ]
    )
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        a.send("write foo.py")
    assert len(_pushbacks(a)) == 1
    assert a._observed_pass("python3 test_thing.py", 0)


def test_caps_when_tests_keep_failing():
    """Tests always exit 1 → no [exit 0] ever observed → capped at verify_attempts."""
    a, _ = _agent(FAIL)

    def scripted():
        yield _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"})
        while True:
            yield _call("cX", "bash", {"command": "python3 test_thing.py"})
            yield LLMResponse(content="I think it passes")

    g = scripted()
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(g)):
        a.send("write foo.py")
    assert len(_pushbacks(a)) == 2
    assert not a._observed_pass("python3 test_thing.py", 0)


def test_no_declared_command_no_gate():
    """No ## Testing block in AGENTS.md → nothing to enforce, even on a code change."""
    a, _ = _agent(PASS, declare=False)
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            LLMResponse(content="done"),
        ]
    )
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        out = a.send("write foo.py")
    assert out == "done"
    assert not _pushbacks(a)


def test_ch12_agents_bash_tool_reaches_its_own_scratch():
    """``_build_ch12_agent`` (tasks/checks.py) is shared by the ch-12 accept check
    AND its demo — proven here directly against that real production helper, not
    the independent stand-in ``_agent()`` above builds for this file's own
    verification-gate tests. No live model needed: only ``.send()`` (the model
    loop) is skipped — the bash tool the helper actually wires runs for real."""
    a, ws = _build_ch12_agent()
    try:
        out = a.tools.call(
            "bash", json.dumps({"command": 'echo PROOF > "$CARBON_SCRATCH_DIR/probe.txt"'})
        )
        assert out.startswith("[exit 0"), out
        assert (a.session_env.scratch_root / "probe.txt").read_text().strip() == "PROOF"
    finally:
        a.close()


def test_ch12_agents_read_file_tool_also_reaches_its_own_scratch():
    """The matching half of the same footer the test above proves bash can
    resolve: ``_build_ch12_agent``'s registry must ALSO pass ``scratch_root=``
    into ``default_tools()``, or a footer naming both routes would have only
    the bash half actually work. Stands in for a real spill by writing directly
    under the session's own scratch, then resolves it through the REGISTRY's
    read_file tool — never the bare function."""
    from harness.limits import spill_ref

    a, ws = _build_ch12_agent()
    try:
        (a.session_env.scratch_root / "offload").mkdir(parents=True)
        (a.session_env.scratch_root / "offload" / "probe.txt").write_text("SPILL-PROOF")
        result = a.tools.call("read_file", json.dumps({"path": spill_ref("probe.txt")}))
        assert result == "SPILL-PROOF"
    finally:
        a.close()


def test_build_ch12_agent_closes_the_agent_it_just_built_when_setup_raises(monkeypatch):
    """``_build_ch12_agent`` constructs the Agent (allocating its scratch in
    ``Agent.__init__``) BEFORE registering tools — the canonical order this task
    put it in. That reorder re-opened a leak window commit 0468747 closed
    elsewhere (``ui/tui.py``'s ``_build_agent``, ``harness/agent.py``'s
    ``run_once``/``_run_repl``): if tool registration raises, the Agent already
    built must still be closed, not leaked. Same spy-on-close technique as
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
        _build_ch12_agent()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the tool-building failure to propagate")

    assert closed, "a raise while building tools must close the Agent already built"
