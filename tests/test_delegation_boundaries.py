"""Delegation boundaries and last-resort context recovery.

A delegated worker is a full Agent with its own Policy and its own provider. Every
test here pins something that does not travel automatically across that boundary
and has to be passed explicitly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from harness.agent import Agent, _coding_tools
from harness.policy import Policy
from harness.subagents import run_subagent
from harness.tools import Tool, ToolRegistry
from harness.workspace import Workspace, write_file_tool
from model import LLMResponse, Provider


def _calls(script) -> Provider:
    return Provider("fake://delegation", "fake", responder=script)


def _tool_call(name: str, **args) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    )


def test_delegated_workers_cannot_mutate_the_workspace():
    """Workers get the parent's workspace read-only.

    A worker runs under its own Policy, so a mutating tool handed to one executes
    without ever reaching the parent's approval gate — a parent running fail-closed
    could delegate the write it just refused.
    """
    root = Path(tempfile.mkdtemp())
    tools = _coding_tools(Workspace(root=root), exclude_session=None)

    seen: list[str] = []

    def script(messages, **kwargs):
        # The worker's own turn: whatever it tries, it must not be able to write.
        if any("focused worker" in str(m.get("content", "")) for m in messages):
            if len(messages) > 2:
                return LLMResponse(content="worker done")
            return _tool_call("write_file", path="pwned.txt", content="x")
        if not seen:
            seen.append("delegated")
            return _tool_call("delegate", task="write a file")
        return LLMResponse(content="parent done")

    Agent(provider=_calls(script), tools=tools).run("delegate a write")

    assert not (root / "pwned.txt").exists()


def test_worker_mutation_still_hits_the_parent_gate_when_tools_are_mutating():
    """A caller that deliberately hands a worker mutating tools must pass the gate too."""
    root = Path(tempfile.mkdtemp())
    registry = ToolRegistry()
    registry.register(write_file_tool(Workspace(root=root)))

    state = {"n": 0}

    def script(messages, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _tool_call("write_file", path="gated.txt", content="x")
        return LLMResponse(content="done")

    run_subagent(
        "write it",
        tools=registry,
        provider=_calls(script),
        policy=Policy(require_approval=frozenset({"write_file"}), approve=None),  # fail closed
    )

    assert not (root / "gated.txt").exists()


def test_delegates_inherit_the_parent_provider():
    """A consumer that passes a custom provider expects its delegates to use it,
    not to silently fall back to whatever the environment points at."""
    root = Path(tempfile.mkdtemp())
    workers_saw: list[str] = []

    def script(messages, **kwargs):
        if any("focused worker" in str(m.get("content", "")) for m in messages):
            workers_saw.append("worker ran on the injected provider")
            return LLMResponse(content="worker done")
        if not workers_saw:
            return _tool_call("delegate", task="look around")
        return LLMResponse(content="parent done")

    provider = _calls(script)
    tools = _coding_tools(
        Workspace(root=root), exclude_session=None, provider=provider, model="injected-model"
    )
    Agent(provider=provider, tools=tools).run("delegate")

    assert workers_saw, "delegate fell back to the environment provider"


def test_tui_builds_its_belt_from_the_shared_builder():
    """The TUI hand-rolled its own registry, which is how its workers ended up
    reading the process cwd instead of the worktree."""
    source = Path("ui/tui.py").read_text()
    assert "_coding_tools(" in source
    assert "delegate_tool(model=" not in source


# --- last-resort context recovery --------------------------------------------
def test_first_turn_overflow_recovers_without_a_prefix_to_compact():
    """Prefix compaction cannot reach an oversized tool result in the current turn."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="big",
            description="returns a lot",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "X" * 50_000,
        )
    )
    state = {"n": 0}

    def script(messages, **kwargs):
        if messages and "context summarizer" in str(messages[0].get("content", "")):
            return LLMResponse(content="SUMMARY")
        state["n"] += 1
        if state["n"] == 1:
            return _tool_call("big")
        if state["n"] == 2:
            raise RuntimeError("maximum context length exceeded")
        return LLMResponse(content="RECOVERED")

    agent = Agent(provider=_calls(script), tools=registry)
    result = agent.run("go")

    assert result.text == "RECOVERED"
    assert agent.retry_count == 1


def test_shrinking_the_current_turn_preserves_the_verification_receipt():
    """The gate reads `[exit 0` at the head of a tool result — shrinking must keep it."""
    agent = Agent(provider=_calls(lambda m, **k: LLMResponse(content="x")))
    agent._active_turn_start = 0
    agent.messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "run-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "uv run verify"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "run-1", "content": "[exit 0]\n" + "PASS\n" * 20_000},
    ]

    assert agent._shrink_turn_tool_results() is True
    assert agent.messages[1]["content"].startswith("[exit 0")
    assert agent._observed_pass("uv run verify", 0) is True
