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
