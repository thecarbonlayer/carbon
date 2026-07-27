"""Subagents (ch-11).

Split work into bounded loops, each a fresh agent with its own isolated context
and tools. A subagent returns the answer, not its transcript, so the main
window stays clean. Independent subtasks fan out in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from harness.policy import Policy
from harness.tools import Tool, ToolRegistry, default_tools
from model import Provider

DEFAULT_WORKER_SYSTEM = "You are a focused worker. Do exactly the subtask and answer concisely."


def run_subagent(
    task: str,
    *,
    system: str | None = None,
    model: str | None = None,
    provider: Provider | None = None,
    tools: ToolRegistry | None = None,
    agents_dir: str = ".",
    policy: Policy | None = None,
) -> str:
    """Run one subtask in an isolated Agent. Read-only unless told otherwise.

    A worker is a full Agent with its own Policy, and nothing about the parent's
    gate travels across that boundary. Defaulting ``policy`` to None would mean an
    empty Policy — every tool in ``tools`` allowed, no approval — so a parent
    running fail-closed could delegate the very write it would refuse. The default
    is therefore ``Policy(read_only=True)``: mutation is opt-in, and a caller that
    wants a worker to write must hand it a policy that says so.
    """
    from harness.agent import Agent  # lazy: avoids an import cycle at module load

    sub = Agent(
        system=system or DEFAULT_WORKER_SYSTEM,
        tools=tools or default_tools(),
        model=model,
        provider=provider,
        agents_dir=agents_dir,
        policy=policy or Policy(read_only=True),
    )
    return sub.send(task)


def fan_out(
    tasks: list[str],
    *,
    model: str | None = None,
    provider: Provider | None = None,
    tools: ToolRegistry | None = None,
    agents_dir: str = ".",
    policy: Policy | None = None,
    max_workers: int = 4,
) -> list[str]:
    """Run subtasks in parallel, each in its own isolated subagent. Order preserved."""
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        return list(
            pool.map(
                lambda t: run_subagent(
                    t,
                    model=model,
                    provider=provider,
                    tools=tools,
                    agents_dir=agents_dir,
                    policy=policy,
                ),
                tasks,
            )
        )


def delegate_tool(
    model: str | None = None,
    *,
    provider: Provider | None = None,
    tools: ToolRegistry | None = None,
    agents_dir: str = ".",
    policy: Policy | None = None,
) -> Tool:
    """A tool that lets a main agent delegate a self-contained subtask to a subagent."""

    def delegate(task: str) -> str:
        return run_subagent(
            task,
            model=model,
            provider=provider,
            tools=tools,
            agents_dir=agents_dir,
            policy=policy,
        )

    return Tool(
        name="delegate",
        description="Delegate a self-contained subtask to a fresh subagent and get its result.",
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
        func=delegate,
    )


def fan_out_tool(
    model: str | None = None,
    *,
    provider: Provider | None = None,
    tools: ToolRegistry | None = None,
    agents_dir: str = ".",
    policy: Policy | None = None,
) -> Tool:
    """A tool that lets the model split work into independent subtasks and run them
    in parallel, each in its own isolated subagent. Results come back labeled and
    ordered, so the model can read them as one block."""

    def fan_out_call(tasks: list[str]) -> str:
        # The model sometimes passes a JSON string instead of a list; iterating that
        # would spawn one subagent per character. Require a real list of strings.
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            return "error: `tasks` must be a list of strings"
        results = fan_out(
            tasks,
            model=model,
            provider=provider,
            tools=tools,
            agents_dir=agents_dir,
            policy=policy,
        )
        return "\n\n".join(
            f"[subtask {i}] {task}\n{result}"
            for i, (task, result) in enumerate(zip(tasks, results, strict=False), 1)
        )

    return Tool(
        name="fan_out",
        description=(
            "Run several independent subtasks in parallel, each in its own fresh "
            "subagent, and get back their labeled results. Use for work that splits "
            "cleanly into pieces that don't depend on each other."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The independent subtasks to run in parallel.",
                }
            },
            "required": ["tasks"],
        },
        func=fan_out_call,
    )
