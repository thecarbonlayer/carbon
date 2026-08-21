"""Delegation boundaries and last-resort context recovery.

A delegated worker is a full Agent with its own Policy and its own provider. Every
test here pins something that does not travel automatically across that boundary
and has to be passed explicitly.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from harness.agent import Agent, _coding_tools
from harness.harness_config import CONFIG
from harness.policy import Policy
from harness.subagents import run_subagent
from harness.tools import Tool, ToolRegistry
from harness.workspace import Workspace, write_file_tool
from model import LLMResponse, Provider


def _calls(script) -> Provider:
    return Provider("fake://delegation", "fake", responder=script)


def _tool_call(tool: str, **args) -> LLMResponse:  # `tool`, not `name` — a tool
    # argument may itself be called "name" (see save_report below).
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "1",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
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

    # Agent before tools: _coding_tools' scratch_root has to come from a
    # SessionEnvironment that already exists (Task 4), and the Agent creates one
    # in __init__ when none is supplied.
    agent = Agent(provider=_calls(script))
    agent.tools = _coding_tools(
        Workspace(root=root), exclude_session=None, session_env=agent.session_env
    )
    agent.run("delegate a write")

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
    agent = Agent(provider=provider)
    agent.tools = _coding_tools(
        Workspace(root=root),
        exclude_session=None,
        provider=provider,
        model="injected-model",
        session_env=agent.session_env,
    )
    agent.run("delegate")

    assert workers_saw, "delegate fell back to the environment provider"


def _needle_workspace() -> tuple[Path, str]:
    """A workspace with enough NEEDLE hits that ``search_text`` overflows the
    small offload budget the two tests below use — shared setup, not a shared
    assertion: each test still drives its OWN model call and its OWN registry."""
    root = Path(tempfile.mkdtemp())
    lines = [f"needle-line-{i:04d}: NEEDLE marker text here" for i in range(150)]
    (root / "log.txt").write_text("\n".join(lines))
    return root, "NEEDLE"


def test_coding_tools_threads_session_env_into_delegate():
    """``delegate_tool`` accepting ``session_env`` (Task 4) is necessary but not
    sufficient — something has to actually PASS it, and that something is
    ``_coding_tools`` (harness/agent.py), the one production caller the tool goes
    through. Driven through a real ``Agent`` + its REAL registry, not the bare
    factory called directly by hand (the way test_offload_strategy.py's
    ``test_delegate_tool_forwards_session_env_...`` proves the factory itself) —
    a regression that drops ``session_env=session_env`` from JUST the
    ``delegate_tool(...)`` registration call, leaving ``fan_out_tool``'s and both
    factories' own plumbing intact, would pass every other test in this suite.

    Proven the same way the factory-level test proves it: the WORKER offloads its
    own oversized result, then reads it back through its own registered
    ``read_file``. Reviewer-caught gap: an earlier version of this test drove
    "delegate" only but was named (and, more importantly, reasoned about in the
    self-review) as if it covered "fan_out" too — it never called that tool, so a
    regression dropping ONLY the ``fan_out_tool`` registration's ``session_env=``
    left the whole suite, this test included, green. See the sibling test below
    for that half, kept deliberately separate so each registration is
    independently provable by mutating just its own line.
    """
    import re

    from harness.harness_config import TruncationPolicy
    from harness.limits import SCRATCH_SCHEME

    root, query = _needle_workspace()

    worker_calls = {"n": 0}
    captured: dict = {}

    def script(messages, **kwargs):
        if any("focused worker" in str(m.get("content", "")) for m in messages):
            worker_calls["n"] += 1
            if worker_calls["n"] == 1:
                return _tool_call("search_text", query=query)
            if worker_calls["n"] == 2:
                footer = str(messages[-1]["content"])
                refs = set(
                    re.findall(re.escape(SCRATCH_SCHEME) + r"offload/[0-9a-f]{16}\.txt", footer)
                )
                assert len(refs) == 1, f"footer should name exactly one file: {refs}"
                return _tool_call("read_file", path=refs.pop())
            # worker_calls["n"] == 3: the read_file readback is now the last message.
            captured["readback"] = str(messages[-1]["content"])
            return LLMResponse(content="worker done")
        if not captured.get("delegated"):
            captured["delegated"] = True
            return _tool_call("delegate", task="search then read back")
        return LLMResponse(content="parent done")

    provider = _calls(script)
    agent = Agent(provider=provider)
    agent.tools = _coding_tools(
        Workspace(root=root),
        exclude_session=None,
        provider=provider,  # the worker must run on THIS script, not the env default
        session_env=agent.session_env,
        tool_output=TruncationPolicy("offload_to_file", 300, 0.5),
    )
    agent.run("delegate: search then read back")

    assert "readback" in captured, (
        "setup: the scripted worker turn must have reached its final call"
    )
    readback = captured["readback"]
    assert not readback.startswith("error:"), readback
    assert "needle-line-0000: NEEDLE marker text here" in readback  # the file's head
    assert "more than 100 hits; narrow it" in readback  # the file's tail


def test_coding_tools_threads_session_env_into_fan_out():
    """The ``fan_out_tool`` half of the test above — kept as a SEPARATE test
    rather than folded into one parametrized case, specifically so that mutating
    only the ``fan_out_tool(...)`` registration's ``session_env=`` in
    ``_coding_tools`` (agent.py) fails ONLY this test, and mutating only
    ``delegate_tool(...)``'s fails only the sibling above — a reviewer's own
    reproduction showed the combined test could not tell the two apart, since it
    only ever drove "delegate."

    A single task in the list keeps ``ThreadPoolExecutor(max_workers=min(4, 1))``
    effectively sequential — fan_out's OWN concurrency and order-preservation are
    a different, already-covered concern (tests/episodes/test_ch11.py's
    ``test_fan_out_preserves_order``); this is about the wiring, not the fan-out.
    """
    import re

    from harness.harness_config import TruncationPolicy
    from harness.limits import SCRATCH_SCHEME

    root, query = _needle_workspace()

    worker_calls = {"n": 0}
    captured: dict = {}

    def script(messages, **kwargs):
        if any("focused worker" in str(m.get("content", "")) for m in messages):
            worker_calls["n"] += 1
            if worker_calls["n"] == 1:
                return _tool_call("search_text", query=query)
            if worker_calls["n"] == 2:
                footer = str(messages[-1]["content"])
                refs = set(
                    re.findall(re.escape(SCRATCH_SCHEME) + r"offload/[0-9a-f]{16}\.txt", footer)
                )
                assert len(refs) == 1, f"footer should name exactly one file: {refs}"
                return _tool_call("read_file", path=refs.pop())
            captured["readback"] = str(messages[-1]["content"])
            return LLMResponse(content="worker done")
        if not captured.get("fanned_out"):
            captured["fanned_out"] = True
            return _tool_call("fan_out", tasks=["search then read back"])
        return LLMResponse(content="parent done")

    provider = _calls(script)
    agent = Agent(provider=provider)
    agent.tools = _coding_tools(
        Workspace(root=root),
        exclude_session=None,
        provider=provider,
        session_env=agent.session_env,
        tool_output=TruncationPolicy("offload_to_file", 300, 0.5),
    )
    agent.run("fan out: search then read back")

    assert "readback" in captured, (
        "setup: the scripted worker turn must have reached its final call"
    )
    readback = captured["readback"]
    assert not readback.startswith("error:"), readback
    assert "needle-line-0000: NEEDLE marker text here" in readback
    assert "more than 100 hits; narrow it" in readback


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


def test_subagent_api_is_read_only_when_no_policy_is_given():
    """Omitting `policy` must not mean "no gate" — the default is read-only.

    A worker is a full Agent with its own Policy, so `policy=None` defaulting to an
    empty Policy would allow every tool in `tools` with no approval. Mutation is
    opt-in; a caller that wants a worker to write must say so.
    """
    root = Path(tempfile.mkdtemp())
    registry = ToolRegistry()
    registry.register(write_file_tool(Workspace(root=root)))

    state = {"n": 0}

    def script(messages, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _tool_call("write_file", path="ungated.txt", content="x")
        return LLMResponse(content="done")

    run_subagent("write it", tools=registry, provider=_calls(script))  # no policy

    assert not (root / "ungated.txt").exists()


def test_mixed_history_overflow_compacts_and_shrinks_in_one_pass():
    """Compactable prefix *and* an oversized current tool result.

    There is one recovery attempt. Doing only the prefix leaves the tool result
    whole and spends the attempt, so the next call overflows again and escapes.
    """
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="big",
            description="returns a lot",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "X" * 80_000,
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
    agent.messages = [{"role": "user", "content": f"old-{i}"} for i in range(20)]

    # An exact compaction count is the assertion, so the PRE-TURN door is pinned shut:
    # at a legal `trigger_fraction` of 0.001 it fires too and the count reads 2, which
    # is correct behaviour presenting as a broken harness. What this test is about is
    # the OVERFLOW door doing both jobs in its one recovery attempt.
    with patch(
        "harness.agent.CONFIG",
        replace(CONFIG, compaction=replace(CONFIG.compaction, trigger_fraction=0.8)),
    ):
        result = agent.run("go")

    assert result.text == "RECOVERED"
    assert agent.compaction_count == 1  # prefix summarized
    tool_results = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_results and all(len(m["content"]) < 80_000 for m in tool_results)  # and shrunk


def test_read_only_workers_can_still_explore_the_repository():
    """Read-only must not mean blind. A worker with only read_file needs exact
    paths, which is not enough to explore a tree it has never seen."""
    root = Path(tempfile.mkdtemp())
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("def target():\n    return 1\n")
    (root / ".env").write_text("LLM_API_KEY=secret")

    worker_names: list[str] = []

    def script(messages, **kwargs):
        if any("focused worker" in str(m.get("content", "")) for m in messages):
            return LLMResponse(content="worker done")
        if not worker_names:
            worker_names.append("x")
            return _tool_call("delegate", task="explore")
        return LLMResponse(content="done")

    agent = Agent(provider=_calls(script))
    agent.tools = _coding_tools(
        Workspace(root=root), exclude_session=None, session_env=agent.session_env
    )
    agent.run("go")

    from harness.tools import list_files, search_text

    listing = list_files("**/*.py", root=root)
    assert "pkg/mod.py" in listing
    assert ".env" not in list_files("**/*", root=root)  # secrets stay refused
    assert "pkg/mod.py:1:" in search_text("def target", root=root)
    assert "outside" in list_files("../**", root=root) or "must stay inside" in list_files(
        "/etc/**", root=root
    )


def test_read_only_refuses_a_custom_tool_that_never_declared_its_effect():
    """`Policy.mutators` only knows carbon's own built-in names.

    A consumer tool called `save_report` is not `write_file`, so a name-only check
    waves it through. carbon cannot inspect a callable to find out what it does, so
    `Tool.mutates` defaults to True and an undeclared tool is refused.
    """
    root = Path(tempfile.mkdtemp())

    def save_report(name: str) -> str:
        (root / name).write_text("written")
        return "saved"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="save_report",  # deliberately not in DEFAULT_MUTATORS
            description="save a report",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            func=save_report,
        )
    )
    state = {"n": 0}

    def script(messages, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _tool_call("save_report", name="report.txt")
        return LLMResponse(content="done")

    run_subagent("save it", tools=registry, provider=_calls(script))  # no policy

    assert not (root / "report.txt").exists()


def test_a_tool_declaring_itself_read_only_still_runs_under_a_read_only_policy():
    """Fail-closed must not mean useless — a declared read-only tool is allowed."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="describe",
            description="describe something",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "DESCRIBED",
            mutates=False,
        )
    )
    state = {"n": 0}

    def script(messages, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _tool_call("describe")
        return LLMResponse(content="done")

    result = Agent(provider=_calls(script), tools=registry, policy=Policy(read_only=True)).run(
        "describe it"
    )

    assert result.tool_calls[0].result == "DESCRIBED"


def test_exploration_tools_refuse_symlinks_that_leave_the_workspace():
    """Confinement is judged on the resolved target, not the link's own name.

    `read_file` already resolves before its containment check; list_files and
    search_text have to agree with it or they become the exfiltration path.
    """
    outside = Path(tempfile.mkdtemp())
    workspace = outside / "ws"
    workspace.mkdir()
    (outside / "secret_outside.txt").write_text("EXFILTRATED")
    (workspace / "innocent.txt").symlink_to(outside / "secret_outside.txt")
    (workspace / "real.txt").write_text("legitimate")

    from harness.tools import list_files, search_text

    listing = list_files("**/*", root=workspace)
    assert "real.txt" in listing
    assert "innocent.txt" not in listing
    assert "no matches" in search_text("EXFILTRATED", root=workspace)


def test_exploration_tools_refuse_symlinks_to_secret_files():
    """A link inside the workspace pointing at .env is still a secret read."""
    workspace = Path(tempfile.mkdtemp())
    (workspace / ".env").write_text("LLM_API_KEY=secret")
    (workspace / "harmless.txt").symlink_to(workspace / ".env")

    from harness.tools import list_files, search_text

    assert "harmless.txt" not in list_files("**/*", root=workspace)
    assert "no matches" in search_text("LLM_API_KEY", root=workspace)
