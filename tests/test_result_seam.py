"""RunResult is the public seam: verdict, usage split, compactions.

External consumers (eval suites) must never need agent._observed_pass or
agent._last_tokens; these tests pin the public alternative."""

from __future__ import annotations

import json

import pytest

import harness.agent as agent_mod
from harness.agent import Agent
from harness.observability import Tracer
from harness.result import RunResult
from harness.sandbox import Sandbox, bash_tool
from harness.tools import default_tools
from harness.workspace import Workspace, write_file_tool
from model import LLMResponse, Provider

# Reused from tests/episodes/test_ch12.py's verification-gate pattern.
AGENTS = "## Testing\n```\npython3 test_thing.py\n```\n"
PASS = "print('ok')\n"
FAIL = "import sys\n\nsys.exit(1)\n"


def _scripted(responses: list[LLMResponse]) -> Provider:
    """A provider that returns each scripted ``LLMResponse`` in order (no network) —
    mirrors tests/test_embedding_seam.py's ``_scripted`` helper."""
    it = iter(responses)

    def responder(messages, **kwargs) -> LLMResponse:
        return next(it)

    return Provider(base_url="fake://x", model="fake", api_key="x", responder=responder)


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


def _verify_agent(test_body: str, *, verify_attempts: int = 2) -> Agent:
    """An Agent wired for the real ``_enforce_run`` verification-gate path —
    mirrors tests/episodes/test_ch12.py's ``_agent`` helper."""
    ws = Workspace()
    ws.write("AGENTS.md", AGENTS)
    ws.write("test_thing.py", test_body)
    tools = default_tools()
    tools.register(write_file_tool(ws))
    tools.register(bash_tool(Sandbox(trusted=True), workdir=str(ws.root)))
    return Agent(tools=tools, agents_dir=str(ws.root), verify_attempts=verify_attempts)


@pytest.fixture
def scripted_agent() -> Agent:
    """One scripted turn with a tracer attached, no tool calls (so no code change,
    no verification gate) — isolates the usage-split / verified=None seam."""
    provider = _scripted(
        [
            LLMResponse(
                content="PONG",
                finish_reason="stop",
                usage={"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            )
        ]
    )
    return Agent(provider=provider, tracer=Tracer(model="fake"))


# --- RunResult defaults --------------------------------------------------------
def test_runresult_defaults_are_backward_compatible():
    r = RunResult(text="x")
    assert r.verified is None
    assert r.usage == {}
    assert r.compactions == 0


# --- usage split -----------------------------------------------------------------
def test_totals_carry_token_split(scripted_agent):
    result = scripted_agent.run("hello")
    totals = scripted_agent.tracer.totals()
    assert set(totals) >= {"input_tokens", "output_tokens", "tokens"}
    assert totals["input_tokens"] + totals["output_tokens"] <= totals["tokens"] or (
        totals["tokens"] == totals["input_tokens"] + totals["output_tokens"]
    )
    assert result.usage == {
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["tokens"],
    }


def test_verified_none_without_verification(scripted_agent):
    assert scripted_agent.run("hello").verified is None


# --- verification verdict, driven through the real _enforce_run path -----------
def test_verified_true_after_a_passing_run(monkeypatch):
    a = _verify_agent(PASS)
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            _call("c2", "bash", {"command": "python3 test_thing.py"}),
            LLMResponse(content="done"),
        ]
    )
    monkeypatch.setattr(agent_mod, "chat", lambda *a, **k: next(replies))
    result = a.run("write foo.py")
    assert result.verified is True


def test_verified_false_when_tests_keep_failing(monkeypatch):
    a = _verify_agent(FAIL)

    def scripted():
        yield _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"})
        while True:
            yield _call("cX", "bash", {"command": "python3 test_thing.py"})
            yield LLMResponse(content="I think it passes")

    g = scripted()
    monkeypatch.setattr(agent_mod, "chat", lambda *a, **k: next(g))
    result = a.run("write foo.py")
    assert result.verified is False
    assert "[unverified:" in result.text
