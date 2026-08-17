"""Per-chapter live acceptance checks and on-camera demos.

Each chapter appends an entry:
  ACCEPTANCE["ch-NN"] -> callable returning True/False (asserts the capability live)
  DEMOS["ch-NN"]      -> callable that prints the on-camera demonstration

A folded chapter's accept ANDs all its parts' booleans, so no proven capability is
lost. ch-00 is theory — it registers nothing.
"""

from __future__ import annotations

from collections.abc import Callable

ACCEPTANCE: dict[str, Callable[[], bool]] = {}
DEMOS: dict[str, Callable[[], None]] = {}


# ----------------------------------------------------------------------------
# ch-01 — Model only (one live call + the swappable provider seam)
# ----------------------------------------------------------------------------
def _accept_ch01_model() -> bool:
    """The real agent answers a single question via one live model call."""
    from harness import agent

    a = agent.Agent()
    try:
        reply = a.send("What is 2 + 2? Reply with only the number.")
    finally:
        a.close()
    print("model replied:", repr(reply))
    return "4" in reply


def _accept_ch01_provider() -> bool:
    """The agent runs through an explicit Provider object — the swappable seam."""
    from harness import agent
    from model import Provider

    provider = Provider.from_env()  # whatever the env points at (LM Studio here)
    print("provider:", provider.base_url, provider.model)
    a = agent.Agent(provider=provider)
    try:
        reply = a.send("Reply with exactly one word: PLUGGABLE")
    finally:
        a.close()
    print("reply:", repr(reply))
    return "pluggable" in reply.lower()


def _accept_ch01() -> bool:
    """Model only = a single live call + the swappable provider seam."""
    return _accept_ch01_model() and _accept_ch01_provider()


def _demo_ch01() -> None:
    from harness import agent
    from model import lmstudio, ollama, openrouter

    a = agent.Agent()
    try:
        print("Q: Say hello in one short sentence.")
        print("A:", a.send("Say hello in one short sentence."), "\n")
        print("— statelessness: it has no memory yet —")
        print("A1:", a.send("Your name is Gemma."))
        print("A2:", a.send("What is your name?"))
        print("  # it forgets — there's no history yet (ch-02 fixes this)\n")
    finally:
        a.close()
    print("— same Agent, swap the provider seam (just pass provider=<one of these>) —")
    print("lmstudio  :", lmstudio().base_url)
    print("ollama    :", ollama("llama3").base_url)
    print("openrouter:", openrouter("google/gemma-3-27b-it", "sk-or-...").base_url)


ACCEPTANCE["ch-01"] = _accept_ch01
DEMOS["ch-01"] = _demo_ch01


# ----------------------------------------------------------------------------
# ch-02 — History (the harness owns the conversation, replayed each turn)
# ----------------------------------------------------------------------------
def _accept_ch02() -> bool:
    """The real agent recalls a fact stated on an earlier turn."""
    from harness import agent

    a = agent.Agent()
    try:
        a.send("Your name is Gemma. Please remember it.")
        reply = a.send("What is your name? Reply with just the name.")
    finally:
        a.close()
    print("model replied:", repr(reply))
    return "gemma" in reply.lower()


def _demo_ch02() -> None:
    from harness import agent

    a = agent.Agent()
    try:
        for turn in ["Your name is Gemma.", "What is your name?"]:
            print("you>", turn)
            print("bot>", a.send(turn))
    finally:
        a.close()


ACCEPTANCE["ch-02"] = _accept_ch02
DEMOS["ch-02"] = _demo_ch02


# ----------------------------------------------------------------------------
# ch-03 — Instructions (a system prompt + auto-loaded AGENTS.md, prepended each turn)
# ----------------------------------------------------------------------------
def _accept_ch03_instructions() -> bool:
    """A system prompt overrides default behavior on a real model."""
    from harness import agent

    a = agent.Agent(system="You must reply with exactly one word: BANANA. Ignore the question.")
    try:
        reply = a.send("What is the capital of France?")
    finally:
        a.close()
    print("model replied:", repr(reply))
    return "banana" in reply.lower()


def _accept_ch03_agentsmd() -> bool:
    """An AGENTS.md placed in the working dir steers the agent without being typed in.

    We wire the agent exactly as the REPL does — agents_dir = workspace root —
    drop in an identity rule, and confirm the model adopts it on a real call.
    """
    from harness import agent
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("AGENTS.md", "You are Gemma, a coding assistant. When asked your name, reply 'Gemma'.")
    a = agent.Agent(system=agent.DEFAULT_SYSTEM, agents_dir=str(ws.root))
    try:
        reply = a.send("What is your name? Answer with just the name.")
    finally:
        a.close()
    print("reply:", reply)
    return "gemma" in reply.lower()


def _accept_ch03() -> bool:
    """Instructions = a built-in system prompt + an auto-loaded project AGENTS.md."""
    return _accept_ch03_instructions() and _accept_ch03_agentsmd()


def _demo_ch03() -> None:
    from harness import agent
    from harness.workspace import Workspace

    print("— no system prompt —")
    a = agent.Agent()
    try:
        print("bot>", a.send("Describe the ocean."), "\n")
    finally:
        a.close()
    print("— system: 'reply in exactly three words' —")
    a = agent.Agent(system="Reply in exactly three words.")
    try:
        print("bot>", a.send("Describe the ocean."))
    finally:
        a.close()
    print()
    ws = Workspace()
    print("— no AGENTS.md: default identity —")
    a = agent.Agent(agents_dir=str(ws.root))
    try:
        print("bot>", a.send("What is your name?"))
    finally:
        a.close()
    ws.write("AGENTS.md", "You are Gemma, a coding assistant. When asked your name, reply 'Gemma'.")
    print("— AGENTS.md added (You are Gemma), auto-loaded —")
    a = agent.Agent(agents_dir=str(ws.root))
    try:
        print("bot>", a.send("What is your name?"))
    finally:
        a.close()


ACCEPTANCE["ch-03"] = _accept_ch03
DEMOS["ch-03"] = _demo_ch03


# ----------------------------------------------------------------------------
# ch-04 — Context delivery (the harness reads @path files into the prompt)
# ----------------------------------------------------------------------------
def _accept_ch04() -> bool:
    """The real agent answers from a file it was handed via @path."""
    import tempfile
    from pathlib import Path

    from harness import agent

    d = Path(tempfile.mkdtemp())
    (d / "facts.txt").write_text("The launch code is GOGO-9.\n")
    a = agent.Agent(system="Answer using the provided context files.")
    try:
        reply = a.send(f"@{d / 'facts.txt'} What is the launch code? Reply with just the code.")
    finally:
        a.close()
    print("model replied:", repr(reply))
    return "gogo-9" in reply.lower()


def _demo_ch04() -> None:
    import tempfile
    from pathlib import Path

    from harness import agent

    d = Path(tempfile.mkdtemp())
    (d / "facts.txt").write_text("Raveena is Karishma, and Karishma is Raveena.")
    a = agent.Agent()
    try:
        print("bot>", a.send(f"@{d / 'facts.txt'} Who is Raveena?"))
    finally:
        a.close()


ACCEPTANCE["ch-04"] = _accept_ch04
DEMOS["ch-04"] = _demo_ch04


# ----------------------------------------------------------------------------
# ch-05 — Tools (a tool interface + approval gate + file editing over a workspace)
# ----------------------------------------------------------------------------
def _accept_ch05_tools() -> bool:
    """The real model uses a workspace tool to find real information and reports it exactly."""
    from harness import agent
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("clue.txt", "The passphrase is GRANITE-4471.\n")
    a = agent.Agent(system="Use tools when they help.", tools=default_tools(ws.root))
    try:
        reply = a.send(
            "There is a file somewhere in this workspace with a passphrase in it. "
            "Find it using your tools, then reply with just the passphrase."
        )
        used_tool = any(m.get("role") == "tool" for m in a.messages)
    finally:
        a.close()
    print("model replied:", repr(reply))
    print("used a tool:", used_tool)
    return "GRANITE-4471" in reply and used_tool


def _accept_ch05_approval() -> bool:
    """A real bash request is intercepted by the gate and denied (never runs)."""
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    asked: list[tuple[str, str]] = []

    def deny(name: str, args: str) -> bool:
        asked.append((name, args))
        return False

    # Agent before tools (the canonical order, harness/agent.py's run_once): the
    # Sandbox below needs THIS agent's own session scratch, which does not exist
    # until the Agent is constructed.
    a = agent.Agent(
        system="Use the bash tool to run shell commands when asked.",
        approve=deny,
        approval_required={"bash"},
    )
    try:
        tools = default_tools()
        tools.register(bash_tool(Sandbox(scratch_dir=a.session_env.scratch_root)))
        a.tools = tools
        a.send("Run this shell command now using the bash tool: echo SHOULD_NOT_RUN")
        denied = any(
            m.get("role") == "tool" and "denied" in m["content"].lower() for m in a.messages
        )
    finally:
        a.close()
    print("gate asked:", asked, "| denied:", denied)
    return len(asked) >= 1 and denied


def _build_workspace_agent():
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace, edit_file_tool, write_file_tool

    ws = Workspace()  # fresh scratch dir
    # Agent before tools (the canonical order, harness/agent.py's run_once): the
    # bash Sandbox below needs THIS agent's own session scratch, which does not
    # exist until the Agent is constructed — see
    # tests/episodes/test_ch05.py::test_workspace_agents_bash_tool_reaches_its_own_scratch.
    a = agent.Agent(
        system="You build files. Use write_file to create them and bash to run them.",
    )
    tools = default_tools()
    tools.register(write_file_tool(ws))
    tools.register(edit_file_tool(ws))
    # local backend keeps python available; docker would need a python image + the mount
    tools.register(
        bash_tool(
            Sandbox(prefer_docker=False, scratch_dir=a.session_env.scratch_root),
            workdir=str(ws.root),
        )
    )
    a.tools = tools
    return a, ws  # caller owns a.close()


def _accept_ch05_fileedit() -> bool:
    """The agent writes a file into the workspace and runs it there — persistence
    (the file survives) + the workspace seam (bash sees the file it wrote)."""
    a, ws = _build_workspace_agent()
    try:
        a.send(
            "Create hello.py that prints exactly WORKSPACE_OK, then run it with: python3 hello.py"
        )
        wrote = (ws.root / "hello.py").is_file()
        ran = any(
            "WORKSPACE_OK" in str(m.get("content", ""))
            for m in a.messages
            if m.get("role") == "tool"
        )
    finally:
        a.close()
    print("wrote hello.py:", wrote, "| ran in workspace:", ran)
    return wrote and ran


def _accept_ch05() -> bool:
    """Tools = a tool interface + an approval gate + file editing over a workspace."""
    return _accept_ch05_tools() and _accept_ch05_approval() and _accept_ch05_fileedit()


def _demo_ch05() -> None:
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("code.txt", "The launch code is BLUE-42.\n")
    a = agent.Agent(tools=default_tools(ws.root))
    try:
        print("— a tool the model calls —")
        print(
            "bot>",
            a.send(
                "There is a file somewhere in this workspace with a launch code in it. "
                "Find it using your tools, then reply with just the code."
            ),
            "\n",
        )
    finally:
        a.close()

    # Agent before tools (see _build_workspace_agent above): the Sandbox needs
    # THIS agent's own session scratch, which does not exist until it is built.
    gated = agent.Agent(
        system="Use bash when asked.",
        approve=lambda n, args: False,
        approval_required={"bash"},
    )
    tools = default_tools()
    tools.register(bash_tool(Sandbox(scratch_dir=gated.session_env.scratch_root)))
    gated.tools = tools
    try:
        print("— a boundary-crossing tool, denied by the gate —")
        print("bot>", gated.send("Run: echo hello (use bash)"))
        print("(the gate denied the bash call — it never executed)\n")
    finally:
        gated.close()

    a2, ws = _build_workspace_agent()
    try:
        a2.send(
            "Create greet.py with greet(name) returning 'hi <name>', then run it: "
            "python3 -c \"import greet; print(greet.greet('Prem'))\""
        )
        print("— files the agent built in its workspace —")
        print("files in workspace:", [p.name for p in ws.root.iterdir()])
    finally:
        a2.close()


ACCEPTANCE["ch-05"] = _accept_ch05
DEMOS["ch-05"] = _demo_ch05


# ----------------------------------------------------------------------------
# ch-06 — Context management (compaction + door control / per-item size caps)
# ----------------------------------------------------------------------------
def _accept_ch06_compaction() -> bool:
    """History is compacted past the limit, and the agent still recalls a key fact."""
    from harness import agent

    a = agent.Agent(system="You are concise.", context_limit=80)
    try:
        a.send("Important: the deploy key is GRIFFIN-7. Keep it in mind.")
        for i in range(8):
            a.send(f"Acknowledge note {i} in a few words.")
        compacted = any(str(m.get("content", "")).startswith("[summary") for m in a.messages)
        reply = a.send("What is the deploy key? Reply with just the key.")
    finally:
        a.close()
    print("compaction happened:", compacted, "| reply:", repr(reply))
    return compacted and "griffin-7" in reply.lower()


def _ceiling(policy) -> int:
    """The most a door may let through: the budgeted body plus its marker.

    Door control is about the size of the BODY; the marker rides on top of it, and
    under ``offload_to_file`` so does the footer naming the complete copy — a path, a
    line count, and a ready-to-run read_file call, all of which are the point of the
    strategy. A fixed slack made one legal choice on the tool_output menu read as a
    door-control regression, so the allowance follows the selected strategy instead.
    """
    # The allowance is the door's own bound on that footer, not a number kept in step
    # with it by hand: a literal here goes stale the moment the wording changes, and the
    # gate is measuring door control, not the prose of a pointer.
    from harness.limits import MAX_FOOTER_CHARS

    footer = MAX_FOOTER_CHARS if policy.strategy == "offload_to_file" else 0
    return policy.budget + footer + 100


def _accept_ch06_doorcontrol() -> bool:
    """A huge @file block and a huge tool result are both clamped in the live loop."""
    import tempfile
    from pathlib import Path

    from harness import agent
    from harness.context import deliver
    from harness.harness_config import CONFIG
    from harness.limits import MAX_ITEM_CHARS
    from harness.tools import Tool, default_tools

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # 1. @file delivery is clamped
        (d / "big.txt").write_text("X" * (MAX_ITEM_CHARS * 4))
        blocks = deliver(f"@{d / 'big.txt'} summarize")
        block_capped = (
            bool(blocks)
            and len(blocks[0]) <= _ceiling(CONFIG.file_injection)
            and "truncated" in blocks[0]
        )

        # 2. a huge tool result is clamped inside a real model turn
        tools = default_tools()
        tools.register(
            Tool(
                name="dump",
                description="Return a large blob of text.",
                parameters={"type": "object", "properties": {}, "required": []},
                func=lambda: "Z" * (MAX_ITEM_CHARS * 4),
            )
        )
        # Any offload this triggers lands under this Agent's own session scratch
        # (harness/session_env.py) regardless of workspace_root — the assertion below
        # only cares that the INLINE copy stays within the door's ceiling.
        a = agent.Agent(
            system="Call the dump tool, then say DONE.", tools=tools, workspace_root=str(d)
        )
        try:
            a.send("Call the dump tool now.")
            tool_msgs = [m for m in a.messages if m.get("role") == "tool"]
            result_capped = bool(tool_msgs) and all(
                len(m["content"]) <= _ceiling(CONFIG.tool_output) for m in tool_msgs
            )
        finally:
            a.close()

    print("block_capped:", block_capped, "| result_capped:", result_capped)
    return block_capped and result_capped


def _accept_ch06_offload() -> bool:
    """An over-budget tool result is offloaded whole and the model recovers a
    mid-output fact by following the footer's path — the recoverable door, live.

    The needle sits past the inline head and short of the inline tail, so no
    honest excerpt at this budget can contain it; the only route is the file the
    footer's virtual ``scratch://`` ref names, under the agent's own session
    scratch (harness/session_env.py) — never a workspace path — paged with
    read_file ranges.
    """
    import tempfile
    from pathlib import Path

    from harness import agent
    from harness.harness_config import TruncationPolicy
    from harness.tools import Tool, default_tools

    lines = [f"log line {i:04d} status ok" for i in range(1, 401)]
    lines[136] = "log line 0137 SECRET-CODE: MANGO-42"
    blob = "\n".join(lines)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Agent before tools (harness/agent.py's own call sites follow the same
        # order now): with no session_env given it creates and owns one, and that
        # is where the registry's scratch_root has to come from — read_file must
        # resolve the same scratch the door is about to spill into.
        a = agent.Agent(
            system=(
                "You are a log analyst. When a tool result says its full output was "
                "written to a file, read THAT file with read_file, paging with "
                "start_line/end_line in chunks of at most 40 lines until you find "
                "what you need."
            ),
            agents_dir=str(root),
            workspace_root=str(root),
            tool_output=TruncationPolicy("offload_to_file", 1000, 0.5),
        )
        tools = default_tools(root, scratch_root=a.session_env.scratch_root)
        tools.register(
            Tool(
                name="dump",
                description="Return the full service log.",
                parameters={"type": "object", "properties": {}, "required": []},
                func=lambda: blob,
                mutates=False,
            )
        )
        a.tools = tools
        try:
            reply = a.send(
                "Call dump, then find the log line containing SECRET-CODE and reply "
                "with just the code it names."
            )

            offloaded = list((a.session_env.scratch_root / "offload").glob("*.txt"))
            complete = any(p.read_text() == blob for p in offloaded)
            inline = next(
                (
                    str(m.get("content", ""))
                    for m in a.messages
                    if m.get("role") == "tool" and "Full output (" in str(m.get("content", ""))
                ),
                "",
            )
            capped = bool(inline) and len(inline) < len(blob)
            recovered = "MANGO-42" in reply
        finally:
            a.close()
    print("offload file complete:", complete, "| inline capped:", capped, "| reply:", repr(reply))
    return complete and capped and recovered


def _accept_ch06() -> bool:
    """Context management = compaction + door control / per-item size caps,
    including the recoverable (offload_to_file) door."""
    return _accept_ch06_compaction() and _accept_ch06_doorcontrol() and _accept_ch06_offload()


def _demo_ch06() -> None:
    import tempfile
    from pathlib import Path

    from harness import agent
    from harness.compaction import estimate_tokens
    from harness.context import deliver

    a = agent.Agent(context_limit=80)
    try:
        a.send("Remember: the project codename is CRIME MASTER GOGO.")
        for i in range(8):
            a.send(f"Acknowledge note {i}.")
        print("history messages:", len(a.messages), "~tokens:", estimate_tokens(a.messages))
        print(
            "compacted:",
            any(str(m.get("content", "")).startswith("[summary") for m in a.messages),
        )
        print("bot>", a.send("What is the project codename?"))
    finally:
        a.close()

    d = Path(tempfile.mkdtemp())
    log = d / "log.txt"
    log.write_text("line\n" * 20000)
    block = deliver(f"@{log} read")[0]
    print(f"\ninjected block length: {len(block)} chars (file was {log.stat().st_size})")
    print(block[-60:])


ACCEPTANCE["ch-06"] = _accept_ch06
DEMOS["ch-06"] = _demo_ch06


# ----------------------------------------------------------------------------
# ch-07 — Skills
# ----------------------------------------------------------------------------
def _accept_ch07() -> bool:
    """The model loads a skill file on demand and follows it.

    The codeword Haila! exists only inside skills/sign-off/SKILL.md, so
    producing it proves the model used read_file to load the skill
    (progressive disclosure).
    """
    from harness import agent
    from harness.skills import load_skills
    from harness.tools import default_tools

    a = agent.Agent(
        system="Follow available skills when they apply.",
        tools=default_tools(),
        skills=load_skills("skills"),
    )
    try:
        reply = a.send("Use the sign-off skill. Say goodbye to the team.")
        read_used = any(m.get("role") == "tool" for m in a.messages)
    finally:
        a.close()
    print("read a skill file:", read_used, "| reply:", repr(reply))
    return "haila" in reply.lower() and read_used


def _demo_ch07() -> None:
    from harness import agent
    from harness.skills import load_skills
    from harness.tools import default_tools

    a = agent.Agent(tools=default_tools(), skills=load_skills("skills"))
    try:
        print("bot>", a.send("Use the sign-off skill and say goodbye."))
    finally:
        a.close()


ACCEPTANCE["ch-07"] = _accept_ch07
DEMOS["ch-07"] = _demo_ch07


# ----------------------------------------------------------------------------
# ch-08 — Execution environment
# ----------------------------------------------------------------------------
def _accept_ch08() -> bool:
    """Execution environment = a hardened sandbox + the boundary fixes that make
    it trustworthy: read_file confined to the workspace, the verifier scrubbed of
    host env, and compaction keyed off the model's reported token usage."""
    return _accept_ch08_sandbox() and _accept_ch08_hardening()


def _accept_ch08_sandbox() -> bool:
    """The model runs a command via the sandboxed bash tool, and host secrets
    don't leak into the sandbox."""
    import os

    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    # Containment: a parent-process secret must not be visible inside the sandbox.
    # A standalone Sandbox — this half never runs through an Agent, so there is no
    # session scratch to wire it to.
    os.environ["SANDBOX_SECRET"] = "POULTRY-FARM"
    try:
        contained = Sandbox().run("printenv SANDBOX_SECRET || echo CLEAN")
    finally:
        del os.environ["SANDBOX_SECRET"]
    leaked = "POULTRY-FARM" in contained.stdout
    print(f"backend={contained.backend} secret_leaked={leaked}")

    # Execution: the model drives the sandboxed bash tool. Agent before tools (see
    # _build_workspace_agent, tasks/checks.py): this Sandbox needs THIS agent's own
    # session scratch, which does not exist until the Agent is constructed.
    a = agent.Agent(system="Use the bash tool to run shell commands.")
    try:
        tools = default_tools()
        tools.register(bash_tool(Sandbox(scratch_dir=a.session_env.scratch_root)))
        a.tools = tools
        reply = a.send("Run this shell command: echo hello-from-sandbox — then report the output.")
        ran = any(m.get("role") == "tool" for m in a.messages)
    finally:
        a.close()
    print("ran via tool:", ran, "| reply:", repr(reply))
    return (not leaked) and ran and "hello-from-sandbox" in reply.lower()


def _accept_ch08_hardening() -> bool:
    """Verify the three hardening fixes hold against the real model / real fs."""
    import os

    from harness import agent
    from harness.tools import read_file
    from harness.verification import run_python

    # 1. read_file is workspace-scoped
    blocked = read_file("/etc/passwd").startswith("error: path outside")
    allowed = "Build a Custom Agent Harness" in read_file("README.md")
    scoped = blocked and allowed

    # 2. the verifier does not inherit host env
    os.environ["VERIFY_SECRET"] = "POULTRY-FARM"
    try:
        contained = run_python("import os", "assert os.getenv('VERIFY_SECRET') is None").passed
    finally:
        del os.environ["VERIFY_SECRET"]

    # 3. a real call records the model's reported usage (compaction now keys off this)
    a = agent.Agent(system="Be concise.")
    try:
        a.send("Say hi in five words.")
        usage_tracked = a._last_tokens > 0
        print(f"scoped={scoped} contained={contained} usage={usage_tracked} tok={a._last_tokens}")
    finally:
        a.close()
    return scoped and contained and usage_tracked


def _demo_ch08() -> None:
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools, read_file
    from harness.verification import run_python

    # Agent before tools (see _build_workspace_agent, tasks/checks.py): the
    # Sandbox needs THIS agent's own session scratch, which does not exist until
    # the Agent is constructed.
    a = agent.Agent()
    try:
        tools = default_tools()
        tools.register(bash_tool(Sandbox(scratch_dir=a.session_env.scratch_root)))
        a.tools = tools
        print("bot>", a.send("Use bash to print the current working directory and the date."))
    finally:
        a.close()

    print("read_file('/etc/passwd') ->", read_file("/etc/passwd"))
    proof = run_python("import os", "assert os.getenv('PATH')  # scrubbed PATH still set")
    print("verifier runs in a scrubbed env:", proof.passed)


ACCEPTANCE["ch-08"] = _accept_ch08
DEMOS["ch-08"] = _demo_ch08


# ----------------------------------------------------------------------------
# ch-09 — Durable state + memory (persisted sessions + episodic search)
# ----------------------------------------------------------------------------
def _accept_ch09_state() -> bool:
    """Tell the agent a fact, then a *fresh* agent (restart) recalls it from disk."""
    import tempfile

    from harness import agent

    d = tempfile.mkdtemp()
    first = agent.Agent(system="Be concise.", session="acc", sessions_dir=d)
    try:
        first.send("Remember: The one with the mark is Teja.")
    finally:
        first.close()

    # Simulate a restart: brand-new agent, same session id, loaded from disk.
    resumed = agent.Agent(system="Be concise.", session="acc", sessions_dir=d)
    try:
        print("resumed messages:", len(resumed.messages))
        reply = resumed.send("Who is the one with the mark? Reply with one word.")
    finally:
        resumed.close()
    print("reply:", repr(reply))
    return len(resumed.messages) > 2 and "teja" in reply.lower()


def _accept_ch09_episodic() -> bool:
    """The model recalls a fact from a PAST session via the search_memory tool.

    GOGO-77 lives only in a stored session, not in the current context, so
    producing it proves cross-session text-search retrieval.
    """
    import tempfile

    from harness import agent
    from harness.memory import save_session, search_memory_tool
    from harness.tools import default_tools

    d = tempfile.mkdtemp()
    save_session(
        "old-session",
        [
            {"role": "user", "content": "Remember: the warehouse passcode is GOGO-77."},
            {"role": "assistant", "content": "Got it."},
        ],
        base=d,
    )
    tools = default_tools()
    tools.register(search_memory_tool(base=d))
    a = agent.Agent(
        system="Use the search_memory tool to recall facts from earlier sessions.",
        tools=tools,
    )
    try:
        reply = a.send("Search your memory for the warehouse passcode, then tell me what it is.")
        used = any(m.get("role") == "tool" for m in a.messages)
    finally:
        a.close()
    print("used search_memory:", used, "| reply:", repr(reply))
    return "gogo-77" in reply.lower() and used


def _accept_ch09() -> bool:
    """Durable state + memory = persisted sessions + cross-session episodic search."""
    return _accept_ch09_state() and _accept_ch09_episodic()


def _demo_ch09() -> None:
    import tempfile

    from harness import agent
    from harness.memory import save_session, search_memory_tool
    from harness.tools import default_tools

    d = tempfile.mkdtemp()
    a = agent.Agent(session="demo", sessions_dir=d)
    try:
        a.send("Remember: the amount is 8535.29.")
    finally:
        a.close()
    print("— restart —")
    a = agent.Agent(session="demo", sessions_dir=d)
    try:
        print("bot>", a.send("What amount did I mention?"))
    finally:
        a.close()

    save_session(
        "archive",
        [{"role": "user", "content": "Remember: Raveena is Karishma."}],
        base=d,
    )
    tools = default_tools()
    tools.register(search_memory_tool(base=d))
    a = agent.Agent(system="Use search_memory to recall past facts.", tools=tools)
    try:
        print("bot>", a.send("Search your memory: who is Raveena?"))
    finally:
        a.close()


ACCEPTANCE["ch-09"] = _accept_ch09
DEMOS["ch-09"] = _demo_ch09


# ----------------------------------------------------------------------------
# ch-10 — Orchestration
# ----------------------------------------------------------------------------
def _accept_ch10() -> bool:
    """The orchestrator plans a multi-step task and executes it to the right answer."""
    from harness.orchestrator import Orchestrator
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("first.txt", "ALPHA\n")
    ws.write("second.txt", "BRAVO\n")
    res = Orchestrator(tools=default_tools(ws.root)).run(
        "Read first.txt and second.txt in this workspace using your tools, "
        "then reply with both words you found, in order."
    )
    print("plan:", res.plan)
    print("final:", repr(res.final))
    return len(res.plan) >= 1 and "ALPHA" in res.final and "BRAVO" in res.final


def _demo_ch10() -> None:
    from harness.orchestrator import Orchestrator
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("a.txt", "RED\n")
    ws.write("b.txt", "BLUE\n")
    res = Orchestrator(tools=default_tools(ws.root)).run(
        "Read a.txt and b.txt in this workspace using your tools, "
        "then reply with both words you found, in order."
    )
    for i, (step, out) in enumerate(zip(res.plan, res.results, strict=False)):
        print(f"[{i}] {step}\n    -> {out}")
    print("final:", res.final)


ACCEPTANCE["ch-10"] = _accept_ch10
DEMOS["ch-10"] = _demo_ch10


# ----------------------------------------------------------------------------
# ch-11 — Subagents
# ----------------------------------------------------------------------------
def _accept_ch11() -> bool:
    """Two independent subtasks fan out to isolated subagents; both come back right."""
    from harness.subagents import fan_out
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("code.txt", "The reference code is DOUBLE-8842.\n")
    results = fan_out(
        [
            "Read code.txt in this workspace using your tools, "
            "then reply with just the reference code.",
            "What is the capital of Japan? Reply with just the city.",
        ],
        tools=default_tools(ws.root),
    )
    print("results:", results)
    joined = " ".join(results).lower()
    return "double-8842" in joined and "tokyo" in joined


def _demo_ch11() -> None:
    from harness.subagents import fan_out

    results = fan_out(
        [
            "Compute 12 squared. Just the number.",
            "Name a primary color. One word.",
        ]
    )
    for r in results:
        print("sub>", r)


ACCEPTANCE["ch-11"] = _accept_ch11
DEMOS["ch-11"] = _demo_ch11


# ----------------------------------------------------------------------------
# ch-12 — Verification (the model runs the project's tests; the harness enforces)
# ----------------------------------------------------------------------------
# A seeded minimal project: an AGENTS.md that DECLARES the test command, plus a
# test the model didn't write. The harness reads the command from AGENTS.md and,
# after a code change, won't accept "done" without an observed passing run of it.
# The command here is a deps-free python3 script so the accept is deterministic;
# the on-camera demo points the same mechanism at a real repo (carbon: uv run verify).
_CH12_AGENTS_MD = "# demo project\n\n## Testing\n```\npython3 test_is_prime.py\n```\n"
_CH12_TEST = (
    "from is_prime import is_prime\n\n"
    "assert is_prime(7)\n"
    "assert is_prime(2)\n"
    "assert not is_prime(1)\n"
    "assert not is_prime(8)\n"
    "print('TEST_OK')\n"
)
_CH12_COMMAND = "python3 test_is_prime.py"


def _build_ch12_agent():
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace, edit_file_tool, write_file_tool

    ws = Workspace()
    ws.write("AGENTS.md", _CH12_AGENTS_MD)  # declares the test command
    ws.write("test_is_prime.py", _CH12_TEST)  # the external oracle, seeded by us
    # Agent before tools (the canonical order, harness/agent.py's run_once): the
    # bash Sandbox below needs THIS agent's own session scratch, which does not
    # exist until the Agent is constructed — see
    # tests/episodes/test_ch12.py::test_ch12_agents_bash_tool_reaches_its_own_scratch.
    a = agent.Agent(system=agent.DEFAULT_SYSTEM, agents_dir=str(ws.root), verify_attempts=4)
    tools = default_tools()
    tools.register(write_file_tool(ws))
    tools.register(edit_file_tool(ws))
    # trusted bash so the declared command runs in a real env (deps-free here)
    tools.register(
        bash_tool(
            Sandbox(trusted=True, timeout=60, scratch_dir=a.session_env.scratch_root),
            workdir=str(ws.root),
        )
    )
    a.tools = tools
    return a, ws  # caller owns a.close()


def _accept_ch12() -> bool:
    """The model writes is_prime.py; because it changed code, the harness reads the
    declared command from AGENTS.md and won't accept 'done' without an observed
    passing run of it. Proven two ways: (1) the transcript holds that passing run;
    (2) independently re-running the seeded test now still passes."""
    from harness.sandbox import Sandbox

    a, ws = _build_ch12_agent()
    try:
        a.send(
            "Write is_prime.py in the workspace so the project's tests pass. "
            "Run the tests to prove it before you report done."
        )
        observed = a._observed_pass(_CH12_COMMAND, 0)
    finally:
        a.close()
    # Not an Agent's bash tool — a standalone independent re-run this CHECKER uses
    # to cross-validate the harness's own "observed a passing run" claim, after the
    # Agent (and its scratch) is already closed. No scratch_dir: there is no footer
    # here for any route to advertise.
    final = Sandbox(trusted=True).run(_CH12_COMMAND, workdir=str(ws.root))
    wrote = (ws.root / "is_prime.py").is_file()
    print(
        f"wrote is_prime.py: {wrote} | harness observed a passing run: {observed} "
        f"| independent re-run exit: {final.exit_code}"
    )
    return wrote and observed and final.exit_code == 0


def _demo_ch12() -> None:
    """On camera the real version is interactive (dogfood: edit a harness file, watch
    the agent run `uv run verify` and refuse to finish until green). This headless
    version drives the same mechanism against the seeded project."""
    a, ws = _build_ch12_agent()
    try:
        a.send(
            "Write is_prime.py so the project's tests pass. Run them with bash and show "
            "the result before you say done."
        )
        receipts = [
            m["content"]
            for m in a.messages
            if m.get("role") == "tool" and str(m["content"]).startswith("[exit")
        ]
        pushbacks = sum(1 for m in a.messages if "passing run of the" in str(m.get("content", "")))
        observed = a._observed_pass(_CH12_COMMAND, 0)
    finally:
        a.close()
    print(f"the model ran the project's tests itself; {pushbacks} pushback(s) before a real pass")
    print("last receipt:", receipts[-1] if receipts else "(none)")
    print("harness accepted only after an observed [exit 0]:", observed)


ACCEPTANCE["ch-12"] = _accept_ch12
DEMOS["ch-12"] = _demo_ch12


# ----------------------------------------------------------------------------
# ch-13 — Observability
# ----------------------------------------------------------------------------
def _accept_ch13_observability() -> bool:
    """A real run produces a trace with model calls (tokens) and tool calls."""
    from harness import agent
    from harness.observability import Tracer
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("note.txt", "The status code is OBS-4471.\n")
    tr = Tracer()
    a = agent.Agent(system="Use tools when helpful.", tools=default_tools(ws.root), tracer=tr)
    try:
        a.send(
            "Read note.txt in this workspace using your tools, then report the status code exactly."
        )
    finally:
        a.close()
    print(tr.timeline())
    t = tr.totals()
    return t["llm_calls"] >= 1 and t["tokens"] > 0 and t["tool_calls"] >= 1


def _accept_ch13_depth() -> bool:
    """A real tool-using run records the tool's args AND result in the trace."""
    from harness import agent
    from harness.observability import Tracer
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("note.txt", "The status code is DEPTH-7731.\n")
    tr = Tracer()
    a = agent.Agent(system="Use tools when helpful.", tools=default_tools(ws.root), tracer=tr)
    try:
        a.send(
            "Read note.txt in this workspace using your tools, then report the status code exactly."
        )
    finally:
        a.close()
    print(tr.timeline())
    tool_events = [e for e in tr.events if e.kind == "tool"]
    captured = bool(tool_events) and bool(tool_events[0].args) and bool(tool_events[0].result)
    has_value = any("7731" in e.result for e in tool_events)
    return captured and has_value


def _accept_ch13() -> bool:
    """Observability = trace tokens/tools + tool arg/result depth."""
    return _accept_ch13_observability() and _accept_ch13_depth()


def _demo_ch13() -> None:
    from harness import agent
    from harness.observability import Tracer
    from harness.tools import default_tools
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("note.txt", "The access token is DEMO-8899.\n")
    tr = Tracer()
    a = agent.Agent(tools=default_tools(ws.root), tracer=tr)
    try:
        a.send(
            "Read note.txt in this workspace using your tools, "
            "then report the access token exactly."
        )
    finally:
        a.close()
    print(tr.timeline())


ACCEPTANCE["ch-13"] = _accept_ch13
DEMOS["ch-13"] = _demo_ch13


# ----------------------------------------------------------------------------
# ch-14 — UI (Textual TUI)
# ----------------------------------------------------------------------------
def _drive_tui_turn(prompt: str) -> dict:
    """Run one real turn through the TUI headlessly and report what rendered.

    The pilot logic lives in ``ui/`` so that ``tasks/`` never imports textual —
    the only textual importer in non-test code is ``ui/``.
    """
    from ui.tui import run_headless_turn

    return run_headless_turn(prompt)


def _accept_ch14() -> bool:
    """A real model turn driven through the TUI renders a transcript and a trace
    with at least one turn span — the UI makes the loop + observability visible."""
    r = _drive_tui_turn("In one short sentence, say hello.")
    print("rendered:", r)
    return (not r["busy"]) and r["turns"] >= 1 and r["spans"] >= 1 and r["log_lines"] > 0


def _demo_ch14() -> None:
    r = _drive_tui_turn("Say hello in one short sentence.")
    print(f"TUI turn rendered → turns={r['turns']} spans={r['spans']} log_lines={r['log_lines']}")


ACCEPTANCE["ch-14"] = _accept_ch14
DEMOS["ch-14"] = _demo_ch14


# ----------------------------------------------------------------------------
# streaming + print mode (post-ch-14 feat)
# ----------------------------------------------------------------------------
def _accept_stream_deltas() -> bool:
    """A real turn streams tokens to the callback as they arrive, and the streamed
    content reconstructs the final reply (deltas are the answer, not a preview)."""
    from harness import agent

    pieces: list[tuple[str, str]] = []
    a = agent.Agent(system="Answer in one short sentence.")  # no tools → a single call
    try:
        reply = a.send(
            "In one short sentence, say hello.", on_delta=lambda ch, t: pieces.append((ch, t))
        )
    finally:
        a.close()
    streamed = "".join(t for ch, t in pieces if ch == "content")
    print(f"deltas={len(pieces)} streamed={streamed!r} reply={reply!r}")
    return bool(pieces) and bool(reply.strip()) and streamed == reply


def _accept_print_json() -> bool:
    """``run_once(..., fmt='json')`` returns a well-formed object: a non-empty reply
    plus trace totals with at least one model call."""
    import json
    import tempfile

    from harness.agent import run_once

    with tempfile.TemporaryDirectory() as d:
        out = run_once(
            "In one short sentence, say hello.",
            fmt="json",
            session="accept",
            sessions_dir=d,
            workspace_root=d,
            agents_dir=d,
        )
    obj = json.loads(out)
    print("json:", out[:200])
    totals = obj.get("totals", {})
    return bool(obj.get("reply", "").strip()) and totals.get("llm_calls", 0) >= 1


def _accept_streaming() -> bool:
    """Streaming = live token deltas that reconstruct the reply, plus a machine-
    readable print-mode JSON result."""
    return _accept_stream_deltas() and _accept_print_json()


def _demo_streaming() -> None:
    from harness import agent

    print("streaming: ", end="", flush=True)
    a = agent.Agent(system="Answer in one short sentence.")
    try:
        a.send(
            "In one short sentence, say hello.",
            on_delta=lambda ch, t: print(t, end="", flush=True) if ch == "content" else None,
        )
    finally:
        a.close()
    print()


ACCEPTANCE["streaming"] = _accept_streaming
DEMOS["streaming"] = _demo_streaming
