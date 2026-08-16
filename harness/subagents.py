"""Subagents (ch-11).

Split work into bounded loops, each a fresh agent with its own isolated context
and tools. A subagent returns the answer, not its transcript, so the main
window stays clean. Independent subtasks fan out in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from harness.harness_config import TruncationPolicy
from harness.policy import Policy
from harness.session_env import SessionEnvironment
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
    tool_output: TruncationPolicy | None = None,
    session_env: SessionEnvironment | None = None,
) -> str:
    """Run one subtask in an isolated Agent. Read-only unless told otherwise.

    A worker is a full Agent with its own Policy, and nothing about the parent's
    gate travels across that boundary. Defaulting ``policy`` to None would mean an
    empty Policy — every tool in ``tools`` allowed, no approval — so a parent
    running fail-closed could delegate the very write it would refuse. The default
    is therefore ``Policy(read_only=True)``: mutation is opt-in, and a caller that
    wants a worker to write must hand it a policy that says so.

    ``session_env`` is the parent's session scratch, passed straight through to the
    worker ``Agent`` and to the default tool registry's ``scratch_root``. One
    session, one scratch inventory: a parent running ``offload_to_file`` spills into
    its own scratch and hands the worker a ``scratch://`` footer naming a file
    there, so a parent's footer resolves inside a worker only if the worker's own
    ``read_file`` resolves against that SAME scratch — not a workspace path (there
    never was one to hit) and not a scratch of the worker's own. No env supplied,
    no inheritance: the worker opens its own, and this call closes it below, the
    same as any construction site closes what it owns.

    ``tool_output`` is the parent's truncation policy, inherited for the same
    reason: a worker reading the same oversized results the parent does should cut
    them the same way, not silently drop the middle under whatever the surface
    happens to default to.
    """
    from harness.agent import Agent  # lazy: avoids an import cycle at module load

    # Agent-first, tools-after (the same ordering run_once uses, harness/agent.py):
    # constructed before the default registry, not inside the ``tools=`` expression.
    # ``session_env`` (the parameter) is only ever a real value when the CALLER
    # supplied one; when it's None, the worker's actual scratch doesn't exist until
    # ``Agent.__init__`` creates it. Building ``default_tools(scratch_root=...)``
    # from the parameter instead of from ``sub.session_env`` left a no-env worker's
    # own registry permanently bound to ``scratch_root=None`` — its own read_file
    # could never resolve the scratch:// footer its own offloads had just written.
    # ``sub.session_env`` is correct in both cases: the parent's env when supplied,
    # or the one this Agent just created for itself when it wasn't.
    sub = Agent(
        system=system or DEFAULT_WORKER_SYSTEM,
        tools=tools,
        model=model,
        provider=provider,
        agents_dir=agents_dir,
        policy=policy or Policy(read_only=True),
        tool_output=tool_output,
        session_env=session_env,
    )
    if tools is None:
        # read_file/list_files/search_text are rooted at agents_dir, not the process
        # cwd, so this worker sees the tree it was actually asked to look at —
        # unrelated to scratch_root, which is a different root entirely (see above).
        sub.tools = default_tools(agents_dir, scratch_root=sub.session_env.scratch_root)
    try:
        return sub.send(task)
    finally:
        # This call constructed ``sub``, so this call owns closing it. Agent.close()
        # already tracks ownership by how ``sub`` was built (a supplied session_env
        # is a no-op here; one ``sub`` created itself is removed) — nothing here
        # re-decides that, it only guarantees close() actually runs.
        sub.close()


def fan_out(
    tasks: list[str],
    *,
    model: str | None = None,
    provider: Provider | None = None,
    tools: ToolRegistry | None = None,
    agents_dir: str = ".",
    policy: Policy | None = None,
    tool_output: TruncationPolicy | None = None,
    session_env: SessionEnvironment | None = None,
    max_workers: int = 4,
) -> list[str]:
    """Run subtasks in parallel, each in its own isolated subagent. Order preserved.

    ``session_env`` is forwarded to every worker unchanged — one parent scratch
    shared by all of them, since it names one session's inventory, not one per
    worker. Each ``run_subagent`` call still owns closing its own worker (see
    there): a supplied env is shared, and sharing is not ownership, so this
    function never closes it either — that stays the caller's job, same as any
    borrowed ``SessionEnvironment``. With no env supplied, every worker opens (and
    closes) its own, exactly as a single ``run_subagent`` call would.
    """
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
                    tool_output=tool_output,
                    session_env=session_env,
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
    tool_output: TruncationPolicy | None = None,
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
            tool_output=tool_output,
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
    tool_output: TruncationPolicy | None = None,
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
            tool_output=tool_output,
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
