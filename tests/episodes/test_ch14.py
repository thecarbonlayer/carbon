"""ch-14 — UI (Textual TUI).

Capability: a two-pane TUI over the *single* agent that makes the harness's
primitives visible — a transcript (the loop) and a trace of nested spans with
tokens + cost (observability), plus an approval modal that shows a diff for file
edits (the gate). Durable state persists the one session and is exposed as
``/reset`` and ``/new`` commands. The agent/model is mocked; the UI is driven
headlessly with Textual's pilot.
"""

import asyncio
from unittest.mock import patch

import harness.agent as agent_mod
from harness.memory import delete_session, list_sessions, load_session, save_session
from harness.observability import Tracer
from model import LLMResponse
from model.pricing import cost, format_cost
from ui.tui import AgentTUI, ApprovalModal, approval_preview


# --- pure pieces --------------------------------------------------------------
def test_cost_local_is_free_hosted_is_metered():
    assert cost("google/gemma-4-26b-a4b", 1000, 1000) == 0.0  # local/unknown → free
    assert cost("openai/gpt-4o-mini", 1_000_000, 0) == 0.15  # priced from the table
    assert format_cost(0.0) == "$0.0000"


def test_list_sessions_counts_and_orders(tmp_path):
    save_session("old", [{"role": "user", "content": "a"}], base=tmp_path)
    save_session(
        "new",
        [{"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}],
        base=tmp_path,
    )
    sessions = list_sessions(tmp_path)
    assert {s["name"] for s in sessions} == {"old", "new"}
    counts = {s["name"]: s["messages"] for s in sessions}
    assert counts == {"old": 1, "new": 2}


def test_tracer_records_verify_cost_and_turns():
    seen = []
    tr = Tracer(model="openai/gpt-4o-mini", on_event=seen.append)
    tr.turn_start()
    tr.record_llm({"prompt_tokens": 1_000_000, "completion_tokens": 0}, 0.1)
    tr.record_verify(False, 0.01, "boom")
    tr.record_verify(True, 0.01, "")
    assert len(seen) == 3  # on_event fired per step
    verify = [e for e in tr.events if e.kind == "verify"]
    assert [e.status for e in verify] == ["fail", "pass"]
    assert all(e.turn == 1 for e in tr.events)  # nested under turn 1
    assert tr.totals()["cost"] == 0.15  # priced from usage


def test_trace_persists_across_restart(tmp_path):
    # turn 1: a tracer-bearing agent records and saves its trace
    tr1 = Tracer(model="google/gemma-4-26b-a4b")
    with patch.object(
        agent_mod, "chat", return_value=LLMResponse(content="ok", usage={"total_tokens": 9})
    ):
        agent_mod.Agent(session="cli", sessions_dir=str(tmp_path), tracer=tr1).send("hi")
    assert tr1.events  # recorded in-memory

    # restart: a fresh agent + fresh tracer for the same session restores the trace
    tr2 = Tracer(model="google/gemma-4-26b-a4b")
    agent_mod.Agent(session="cli", sessions_dir=str(tmp_path), tracer=tr2)
    assert [e.kind for e in tr2.events] == [e.kind for e in tr1.events]
    assert tr2._turn == tr1._turn  # turn numbering continues, not reset


def test_trace_files_do_not_pollute_session_list(tmp_path):
    tr = Tracer()
    with patch.object(agent_mod, "chat", return_value=LLMResponse(content="ok")):
        agent_mod.Agent(session="cli", sessions_dir=str(tmp_path), tracer=tr).send("hi")
    names = [s["name"] for s in list_sessions(tmp_path)]
    assert names == ["cli"]  # the traces/ subdir is not mistaken for a session


def test_approval_preview_bash_and_diff(tmp_path):
    from harness.workspace import Workspace

    ws = Workspace(root=tmp_path)
    ws.write("calc.py", "def add(a, b):\n    return a-b\n")

    bash = approval_preview("bash", '{"command": "rm -rf build"}', ws)
    assert bash["kind"] == "bash" and "rm -rf build" in bash["body"]

    edit = approval_preview("edit_file", '{"path": "calc.py", "old": "a-b", "new": "a + b"}', ws)
    assert edit["kind"] == "diff"
    assert "-    return a-b" in edit["body"] and "+    return a + b" in edit["body"]


def test_delete_session_clears_messages_and_trace(tmp_path):
    """Reviewer follow-up (Task 3): delete_session is the ONE production call site
    that deletes a session (ui/tui.py's /reset), so it has to remove the durable
    scratch too — otherwise a reset leaves the previous session's complete tool
    outputs (limits.py: unredacted bytes, "a printenv, a token-bearing log") sitting
    on disk with nothing pointing at them anymore."""
    tr = Tracer(model="google/gemma-4-26b-a4b")
    with patch.object(agent_mod, "chat", return_value=LLMResponse(content="ok")):
        agent = agent_mod.Agent(session="cli", sessions_dir=str(tmp_path), tracer=tr)
        agent.send("hi")
    scratch = agent.session_env.scratch_root
    (scratch / "offload").mkdir(parents=True)
    (scratch / "offload" / "secret.txt").write_text("token-bearing log")
    agent.close()  # durable: a no-op on the scratch (Task 3) — it must still be here
    assert load_session("cli", tmp_path)  # persisted
    assert scratch.exists()  # sanity: durable scratch really did survive close()

    delete_session("cli", tmp_path)
    assert load_session("cli", tmp_path) == []  # messages gone
    assert list_sessions(tmp_path) == []  # trace file gone too, nothing lingers
    assert not scratch.exists()  # …and now so is the scratch that held its spills
    delete_session("cli", tmp_path)  # idempotent — no error on a missing session


# --- the TUI, driven headlessly ----------------------------------------------
def test_two_panes_mount(tmp_path):
    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test():
            for sel in ("#conversation", "#trace", "#header", "#prompt"):
                assert app.query_one(sel)
            assert not app.query("#sessions")  # the sessions pane is gone

    asyncio.run(run())


def test_reset_and_new_session_commands(tmp_path):
    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test() as pilot:
            with patch.object(
                agent_mod,
                "chat",
                return_value=LLMResponse(content="hello", usage={"total_tokens": 5}),
            ):
                app._turn_done(app.agent.send("hi"))  # one turn on session "cli"
            await pilot.pause()
            assert app.query(".msg-agent") and app.agent.messages

            app._handle_command("/reset")  # clears the same session in place
            await pilot.pause()
            assert app.agent.session == "cli"
            assert app.agent.messages == []
            assert not app.query(".msg")  # transcript wiped
            assert load_session("cli", tmp_path) == []

            app._handle_command("/new")  # fresh session, new name
            await pilot.pause()
            assert app.agent.session.startswith("chat-")
            assert app.agent.messages == []

    asyncio.run(run())


def test_turn_renders_transcript_and_trace_spans(tmp_path):
    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test():
            with patch.object(
                agent_mod,
                "chat",
                return_value=LLMResponse(content="hello", usage={"total_tokens": 5}),
            ):
                reply = app.agent.send("hi")  # populates messages + tracer (turn 1)
            app._turn_done(reply)  # render transcript + rebuild trace
            tree = app.query_one("#trace-tree")
            assert len(tree.root.children) == 1  # one turn span
            assert len(tree.root.children[0].children) >= 1  # at least the llm step
            assert app.query(".msg-agent")  # transcript rendered the reply as a block

    asyncio.run(run())


def test_resolve_session_name_or_path():
    from tasks.tui import _resolve_session

    assert _resolve_session(None) == ("cli", None)  # default
    assert _resolve_session("chat-1") == ("chat-1", None)  # bare name → default dir
    assert _resolve_session(".sessions/chat-1.jsonl") == ("chat-1", ".sessions")  # path → stem+dir
    assert _resolve_session("/tmp/runs/foo.jsonl") == ("foo", "/tmp/runs")


def test_launch_resumes_named_session(tmp_path):
    save_session(
        "refactor-api",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}],
        base=tmp_path,
    )

    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path), session="refactor-api")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.agent.session == "refactor-api"  # opened the one we named
            assert len(app.agent.messages) == 2  # its history loaded
            assert app.query(".msg-user") and app.query(".msg-agent")  # and rendered

    asyncio.run(run())


def test_approval_modal_allow_and_deny(tmp_path):
    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test() as pilot:
            results = []
            app.push_screen(
                ApprovalModal({"title": "t", "kind": "bash", "body": "ls"}), results.append
            )
            await pilot.pause()
            await pilot.press("a")  # allow
            await pilot.pause()
            assert results == [True]

            app.push_screen(
                ApprovalModal({"title": "t", "kind": "bash", "body": "ls"}), results.append
            )
            await pilot.pause()
            await pilot.press("d")  # deny
            await pilot.pause()
            assert results == [True, False]

    asyncio.run(run())
