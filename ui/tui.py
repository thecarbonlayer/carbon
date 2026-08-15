"""UI (ch-14) — a Textual TUI over the existing agent.

The thesis is *Agent = Model + Harness + UI*. Every prior chapter built harness;
this one adds the UI, and its whole job is to make *one* agent's primitives
*visible*: the loop (a transcript), observability (a live trace of nested spans
with tokens + cost), and the approval gate (a modal that now shows a diff for file
edits, not just a bash command). The course builds a single agent, so the UI is a
single agent — two panes, conversation and trace. Durable state is still here (the
one session persists and resumes on launch), exposed as ``/reset`` and ``/new``
commands rather than a session switcher, which would imply many agents. A specific
past run is reopened by naming it at launch (``uv run tui <session>``), not picked
from a list.

It does not reimplement the agent. It runs the same ``Agent`` in a worker thread
and renders what the ``Tracer`` records. The one subtlety is the seam: the agent
loop is synchronous and its approval hook blocks, while Textual is async — so the
turn runs off the UI thread and the approval callback bridges back with
``call_from_thread`` + a ``threading.Event``, preserving the fail-closed contract.
"""

from __future__ import annotations

import difflib
import json
import threading
import time

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Static, Tree

import harness.agent as agent_mod
from harness.harness_config import CONFIG
from harness.memory import DEFAULT_DIR, delete_session
from harness.observability import Tracer
from harness.skills import load_skills
from harness.workspace import Workspace
from model import Provider
from model.pricing import format_cost

APPROVAL_TOOLS = CONFIG.approval_tools  # from the editable surface — one source of truth
_STATUS_COLOR = {"denied": "yellow", "error": "red", "fail": "red", "pass": "green", "ok": ""}
_KIND_ICON = {"llm": "◆", "tool": "›", "verify": "✓", "plan": "▷"}


def _unified_diff(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def approval_preview(name: str, args_json: str, workspace: Workspace | None) -> dict:
    """Build what the approval modal shows: a bash command, or a unified diff for
    a file write/edit (computed from the tool args + current workspace state)."""
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        args = {}
    if name == "bash":
        return {
            "title": "approval required · bash",
            "kind": "bash",
            "body": args.get("command", ""),
        }
    if name in ("write_file", "edit_file"):
        path = args.get("path", "")
        raw = workspace.read(path) if workspace else "error: no such file"
        missing = raw.startswith("error:")
        current = "" if missing else raw
        title = f"approval required · {name} · {path}"
        if name == "edit_file":
            # Preview exactly what ws.edit will do — including its failure modes, so
            # the modal never shows "(no change)" for an edit that will actually error.
            old = args.get("old", "")
            if missing:
                return {
                    "title": title,
                    "kind": "other",
                    "body": f"edit will fail: no such file: {path}",
                }
            if old == "":
                return {
                    "title": title,
                    "kind": "other",
                    "body": "edit will fail: `old` must be non-empty",
                }
            if old not in current:
                return {
                    "title": title,
                    "kind": "other",
                    "body": f"edit will fail: text to replace not found in {path}",
                }
            new = current.replace(old, args.get("new", ""), 1)
        else:
            new = args.get("content", "")
        return {
            "title": title,
            "kind": "diff",
            "body": _unified_diff(current, new, path) or "(no change)",
        }
    return {"title": f"approval required · {name}", "kind": "other", "body": args_json}


class ApprovalModal(ModalScreen[bool]):
    """The approval gate as a modal. Pauses the turn until the user answers; fail-closed."""

    BINDINGS = [("a", "allow", "allow"), ("d", "deny", "deny"), ("escape", "deny", "deny")]

    def __init__(self, preview: dict) -> None:
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        p = self.preview
        with Vertical(id="approval"):
            yield Static(p["title"], id="approval-title")
            if p["kind"] == "diff":
                yield Static(Syntax(p["body"], "diff", word_wrap=True), id="approval-body")
            elif p["kind"] == "bash":
                yield Static(Syntax(p["body"], "bash", word_wrap=True), id="approval-body")
            else:
                yield Static(p["body"], id="approval-body")
            yield Static("runs fail-closed · a allow · d deny", id="approval-note")
            with Horizontal(id="approval-actions"):
                yield Button("allow · a", id="allow", variant="success")
                yield Button("deny · d", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class AgentTUI(App):
    """Two panes (conversation · trace) + header + footer — one agent, made visible."""

    CSS = """
    #header { height: 1; padding: 0 1; background: $panel; color: $accent; }
    #body { height: 1fr; }
    .pane { height: 1fr; }
    #conversation { width: 1fr; border-right: solid $panel; }
    #trace { width: 46; }
    .pane-body { height: 1fr; }
    .pane-foot { height: 3; dock: bottom; border-top: solid $panel; padding: 0 1; }
    #prompt { border: none; border-top: solid $panel; height: 3; padding: 0 1; }
    #trace-foot { content-align: left middle; color: $text-muted; }
    #log { height: 1fr; }
    .msg { height: auto; padding: 1 2; margin: 0 1 1 1; }
    .msg-role { text-style: bold; margin-bottom: 1; }
    .msg-user { background: $surface-lighten-1; }
    .msg-user .msg-role { color: $text; }
    .msg-agent { background: $panel; }
    .msg-agent .msg-role { color: $accent; }
    .msg-reason { color: $text-muted; }
    ApprovalModal { align: center middle; }
    #approval { width: 76; height: auto; max-height: 80%; border: thick $warning; padding: 1 2; }
    #approval-title { color: $warning; text-style: bold; }
    #approval-note { color: $text-muted; margin-top: 1; }
    #approval-actions { height: auto; margin-top: 1; }
    #approval-actions Button { margin-right: 2; }
    """

    BINDINGS = [
        ("ctrl+n", "new_session", "new"),
        ("ctrl+q", "quit", "quit"),
    ]

    def __init__(
        self,
        agent: agent_mod.Agent | None = None,
        provider: Provider | None = None,
        workspace: Workspace | None = None,
        sessions_dir: str = DEFAULT_DIR,
        session: str = "cli",
        extensions: bool = False,
    ) -> None:
        super().__init__()
        self.provider = provider or Provider.from_env()
        self.workspace = workspace or Workspace()
        self.sessions_dir = sessions_dir
        self.extensions = extensions
        self._busy = False
        # Streaming state: the live agent block being filled this turn (None between
        # turns). ``_stream_delta`` mounts it on the first token; ``_turn_done``
        # finalizes it in place so the streamed block is never duplicated.
        self._live_body: Static | None = None
        self._live_reason: Static | None = None
        self._live_content = ""
        self._live_reason_text = ""
        self.agent = agent or self._build_agent(session)

    # --- agent construction -------------------------------------------------
    def _build_agent(self, session: str) -> agent_mod.Agent:
        # One builder for print mode, the REPL, and the TUI. Building the belt here
        # by hand is how the TUI's delegates ended up reading the process cwd instead
        # of the worktree — the detached-worktree mismatch. Read from the same tree we
        # write to, and hand workers the same binding. Trusted bash: the agent works
        # on your real project, so your test command runs for real, approval-gated.
        tools = agent_mod._coding_tools(
            self.workspace,
            exclude_session=session,  # recall across *other* sessions
            provider=self.provider,
            model=self.provider.model,
            sessions_dir=self.sessions_dir,
        )
        if self.extensions:
            from harness.extensions import load_extensions

            load_extensions(tools, *agent_mod._extension_dirs(self.workspace.root))
        tracer = Tracer(model=self.provider.model)
        return agent_mod.Agent(
            system=agent_mod.DEFAULT_SYSTEM,
            tools=tools,
            skills=load_skills("skills"),
            session=session,
            sessions_dir=self.sessions_dir,
            provider=self.provider,
            model=self.provider.model,
            tracer=tracer,
            approve=self._approve,
            approval_required=APPROVAL_TOOLS,
            agents_dir=str(self.workspace.root),  # AGENTS.md lives where the agent works
            workspace_root=str(self.workspace.root),  # …and so does anything it spills
        )

    # --- layout -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="body"):
            with Vertical(id="conversation", classes="pane"):
                yield VerticalScroll(id="log", classes="pane-body")
                yield Input(
                    placeholder="type a message…   ·   /plan <task>   /reset   /new",
                    id="prompt",
                    classes="pane-foot",
                )
            with Vertical(id="trace", classes="pane"):
                yield Tree("trace", id="trace-tree", classes="pane-body")
                yield Static("0 tok · $0.0000", id="trace-foot", classes="pane-foot")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#trace-tree", Tree).show_root = False
        self._refresh_header()
        self._render_history()
        self._rebuild_trace()
        self.query_one("#prompt", Input).focus()

    # --- rendering ----------------------------------------------------------
    def _refresh_header(self) -> None:
        a = self.agent
        text = (
            f"⚙ {self.provider.model}   ·   session: {a.session}   ·   "
            f"skills: {len(a.skills)}   ·   ctx {a._last_tokens}/{a.context_limit}"
        )
        self.query_one("#header", Static).update(text)

    def _render_history(self) -> None:
        self.query_one("#log", VerticalScroll).remove_children()
        for m in self.agent.messages:
            role = m.get("role")
            content = str(m.get("content", "") or "")
            if role == "user" and not content.startswith("Context file:"):
                self._write_user(content)
            elif role == "assistant" and content:
                self._write_agent(content)

    def _mount_block(self, role_label: str, body, role_class: str) -> None:
        """Append a message block (role label + body) and scroll to it. The body is
        any Rich renderable — plain str for user input, ``Markdown`` for agent replies,
        ``Syntax`` for approval diffs."""
        log = self.query_one("#log", VerticalScroll)
        block = Vertical(
            Label(role_label, classes="msg-role"),
            Static(body, classes="msg-body"),
            classes=f"msg {role_class}",
        )
        log.mount(block)
        log.scroll_end(animate=False)

    def _write_user(self, text: str) -> None:
        self._mount_block("you ▸", text, "msg-user")

    def _write_agent(self, text: str) -> None:
        self._mount_block("agent ▸", Markdown(text), "msg-agent")

    # --- streaming ----------------------------------------------------------
    def _mount_live_block(self) -> None:
        """Mount the empty agent block that this turn's tokens stream into: a dimmed
        reasoning line above a Markdown body for the visible answer."""
        self._live_reason = Static("", classes="msg-reason")
        self._live_body = Static(Markdown(""), classes="msg-body")
        block = Vertical(
            Label("agent ▸", classes="msg-role"),
            self._live_reason,
            self._live_body,
            classes="msg msg-agent",
        )
        log = self.query_one("#log", VerticalScroll)
        log.mount(block)
        log.scroll_end(animate=False)

    def _stream_delta(self, channel: str, text: str) -> None:
        """Append one streamed token to the live block (mounting it on first use).
        Reasoning renders dimmed and is cleared once the visible answer begins."""
        if self._live_body is None:
            self._mount_live_block()
        if channel == "reasoning":
            self._live_reason_text += text
            if self._live_reason is not None:
                self._live_reason.update(Text(self._live_reason_text, style="dim"))
        else:  # content — the visible answer
            self._live_content += text
            if self._live_reason is not None:
                self._live_reason.update("")  # thinking done; hand the block to the answer
            if self._live_body is not None:
                self._live_body.update(Markdown(self._live_content))
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    def _reset_live(self) -> None:
        self._live_body = None
        self._live_reason = None
        self._live_content = ""
        self._live_reason_text = ""

    def _write_diff(self, diff_text: str) -> None:
        self._mount_block("agent ▸", Syntax(diff_text, "diff", word_wrap=True), "msg-agent")

    def _span_label(self, e) -> Text:
        icon = _KIND_ICON.get(e.kind, "·")
        t = Text(f"{icon} {e.label}  {e.seconds * 1000:.0f} ms")
        color = _STATUS_COLOR.get(e.status, "")
        if color:
            t.stylize(color)
        return t

    def _rebuild_trace(self) -> None:
        tree = self.query_one("#trace-tree", Tree)
        tree.clear()
        events = list(self.agent.tracer.events) if self.agent.tracer else []
        by_turn: dict[int, list] = {}
        for e in events:
            by_turn.setdefault(e.turn, []).append(e)
        for turn in sorted(by_turn):
            evs = by_turn[turn]
            secs = sum(e.seconds for e in evs)
            tnode = tree.root.add(f"turn {turn}  ·  {secs * 1000:.0f} ms", expand=True)
            for e in evs:
                enode = tnode.add(self._span_label(e))
                if e.kind == "llm" or e.cost:
                    enode.add_leaf(f"{e.tokens} tok · {format_cost(e.cost)}")
                if e.args:
                    enode.add_leaf(f"args: {e.args}")
                if e.result:
                    enode.add_leaf(f"→ {e.result}")
        t = self.agent.tracer.totals() if self.agent.tracer else {"tokens": 0, "cost": 0.0}
        self.query_one("#trace-foot", Static).update(
            f"{t['tokens']} tok · {format_cost(t['cost'])}"
        )

    # --- turn lifecycle -----------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return
        event.input.value = ""
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._busy = True
        self._write_user(text)
        self._run_turn(text)

    @work(thread=True, exclusive=True)
    def _run_turn(self, text: str) -> None:
        def sink(channel: str, piece: str) -> None:
            # The turn runs off the UI thread; bridge each token back to it.
            self.call_from_thread(self._stream_delta, channel, piece)

        try:
            reply = self.agent.send(text, on_delta=sink)
        except Exception as exc:  # surface, don't crash the UI
            reply = f"[error] {exc}"
        self.call_from_thread(self._turn_done, reply)

    def _turn_done(self, reply: str) -> None:
        if self._live_body is not None:
            # Finalize the streamed block with the canonical reply, in place — never
            # mount a second block for what the user already watched stream in.
            if self._live_reason is not None:
                self._live_reason.update("")
            self._live_body.update(Markdown(reply))
            self._reset_live()
        else:
            self._write_agent(reply)  # nothing streamed → mount the reply now
        self._rebuild_trace()
        self._refresh_header()
        self._busy = False
        self.query_one("#prompt", Input).focus()

    # --- approval bridge (worker thread -> UI thread -> back) ---------------
    def _approve(self, name: str, args: str) -> bool:
        preview = approval_preview(name, args, self.workspace)
        done = threading.Event()
        box: dict[str, bool] = {}

        def on_dismiss(allowed: bool | None) -> None:
            box["v"] = bool(allowed)
            done.set()

        def show() -> None:
            self.push_screen(ApprovalModal(preview), on_dismiss)

        self.call_from_thread(show)
        done.wait()
        allowed = box.get("v", False)
        if allowed and preview["kind"] == "diff":
            self.call_from_thread(self._write_diff, preview["body"])
        return allowed

    # --- session commands (single agent: clear it, or start fresh) ----------
    def _handle_command(self, text: str) -> None:
        cmd = text.split()[0].lower()
        if cmd == "/reset":
            self._reset_session()
        elif cmd == "/new":
            self._new_session()
        elif cmd == "/plan":
            task = text[len("/plan") :].strip()
            if not task:
                self._write_agent("usage: `/plan <task>`")
                return
            self._busy = True
            self._write_user(text)
            self._run_plan(task)
        else:
            self._write_agent(f"unknown command `{cmd}` · try `/plan`, `/reset`, or `/new`")

    @work(thread=True, exclusive=True)
    def _run_plan(self, task: str) -> None:
        """Run the ch-10 orchestrator off the UI thread: plan the task into steps and
        execute them. The planner's ``plan`` span nests under its own turn in the
        tracer, so it lands in the trace pane alongside the agent's turns."""
        from harness.orchestrator import Orchestrator

        error: str | None = None
        result = None
        try:
            if self.agent.tracer is not None:
                self.agent.tracer.turn_start()
            orch = Orchestrator(model=self.provider.model, tracer=self.agent.tracer)
            result = orch.run(task)
        except Exception as exc:  # surface, don't crash the UI
            error = str(exc)
        self.call_from_thread(self._plan_done, result, error)

    def _plan_done(self, result, error: str | None) -> None:
        if error is not None:
            self._write_agent(f"[plan error] {error}")
        else:
            lines = ["**plan**"]
            lines += [f"{i}. {step}" for i, step in enumerate(result.plan, 1)]
            lines += ["", "**results**"]
            for i, (step, res) in enumerate(zip(result.plan, result.results, strict=False), 1):
                lines.append(f"{i}. {step}")
                lines.append(f"   → {res}")
            self._write_agent("\n".join(lines))
        self._rebuild_trace()
        self._refresh_header()
        self._busy = False
        self.query_one("#prompt", Input).focus()

    def _open_session(self, session: str) -> None:
        """Point the UI at ``session`` (already on disk or brand-new) and re-render."""
        self.agent = self._build_agent(session)
        self._render_history()
        self._rebuild_trace()
        self._refresh_header()
        self.query_one("#prompt", Input).focus()

    def _reset_session(self) -> None:
        """Clear the current session's history + trace, keeping the same name."""
        session = self.agent.session or "cli"
        delete_session(session, self.sessions_dir)
        self._open_session(session)

    def _new_session(self) -> None:
        """Start a fresh session (new name) — a clean transcript and trace."""
        self._open_session(f"chat-{int(time.time())}")

    def action_new_session(self) -> None:
        self._new_session()


# --- headless drive (for the ch-14 accept / off-UI-thread tooling) ----------
def run_headless_turn(prompt: str, *, sessions_dir: str | None = None) -> dict:
    """Run one real turn through the TUI headlessly (Textual pilot) and report
    what rendered — the live model is called via the same worker-thread path the
    real app uses.

    This is the public seam for ``tasks/`` (the ch-14 accept) so that ``tasks/``
    never has to import textual itself: the only textual importer stays ``ui/``.
    """
    import asyncio
    import tempfile

    sessions = sessions_dir or tempfile.mkdtemp(prefix="tui-sessions-")

    async def run() -> dict:
        app = AgentTUI(sessions_dir=sessions)
        async with app.run_test() as pilot:
            app.query_one("#prompt", Input).value = prompt
            await pilot.press("enter")
            for _ in range(600):  # wait up to ~60s for the worker turn to finish
                if not app._busy:
                    break
                await pilot.pause(0.1)
            tree = app.query_one("#trace-tree", Tree)
            return {
                "turns": len(tree.root.children),
                "spans": sum(len(n.children) for n in tree.root.children),
                "log_lines": len(app.query(".msg")),  # rendered message blocks
                "busy": app._busy,
            }

    return asyncio.run(run())
