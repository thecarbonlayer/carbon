"""TUI streaming (post-ch-14 feat).

The Textual UI renders tokens into a live agent block as they arrive, and
finalizes that same block when the turn ends — it must not leave a half-streamed
block *and* mount a duplicate final one. Driven headlessly with Textual's pilot;
the model is mocked to fire the streaming callback.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import harness.agent as agent_mod
from harness.workspace import Workspace
from model import LLMResponse
from ui.tui import AgentTUI


def _streaming_chat(payload, *, on_delta=None, **kwargs):
    if on_delta:
        on_delta("reasoning", "hmm ")
        on_delta("content", "Hel")
        on_delta("content", "lo")
    return LLMResponse(content="Hello", usage={"total_tokens": 3})


def test_streaming_builds_a_live_block_then_finalizes_without_duplicating(tmp_path):
    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test() as pilot:
            with patch.object(agent_mod, "chat", side_effect=_streaming_chat):
                reply = app.agent.send("hi", on_delta=app._stream_delta)
            await pilot.pause()
            assert app._live_content == "Hello"  # content accumulated live
            assert app._live_reason_text == "hmm "  # reasoning streamed on its channel
            assert len(app.query(".msg-agent")) == 1  # exactly one streamed block

            app._turn_done(reply)
            await pilot.pause()
            assert len(app.query(".msg-agent")) == 1  # finalized in place, not duplicated
            assert app._live_body is None  # live state reset for the next turn

    asyncio.run(run())


def test_extensions_flag_loads_project_local_extensions(tmp_path, monkeypatch):
    """The TUI's ``--extensions`` wiring (final whole-branch review — the earlier
    plan wired print mode and the REPL but missed the TUI entirely): the flag
    must reach ``_build_agent`` and actually load a `.carbon/extensions/` tool,
    not just be stored and ignored."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    ext_dir = tmp_path / "workspace" / ".carbon" / "extensions"
    ext_dir.mkdir(parents=True)
    (ext_dir / "hello.py").write_text(
        "from harness.tools import Tool\n\n"
        "def setup(registry):\n"
        "    registry.register(Tool(\n"
        "        name='hello', description='d',\n"
        "        parameters={'type': 'object', 'properties': {}, 'required': []},\n"
        "        func=lambda: 'hi',\n"
        "        mutates=False,\n"
        "    ))\n"
    )

    async def run():
        app = AgentTUI(
            workspace=Workspace(root=str(tmp_path / "workspace")),
            sessions_dir=str(tmp_path / "sessions"),
            extensions=True,
        )
        async with app.run_test():
            assert app.extensions is True
            assert "hello" in app.agent.tools.names()

    asyncio.run(run())


def test_extensions_flag_defaults_to_off(tmp_path, monkeypatch):
    """Without the flag, the same `.carbon/extensions/` tool must not load —
    extensions are opt-in, not auto-discovered."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    ext_dir = tmp_path / "workspace" / ".carbon" / "extensions"
    ext_dir.mkdir(parents=True)
    (ext_dir / "hello.py").write_text(
        "from harness.tools import Tool\n\n"
        "def setup(registry):\n"
        "    registry.register(Tool(\n"
        "        name='hello', description='d',\n"
        "        parameters={'type': 'object', 'properties': {}, 'required': []},\n"
        "        func=lambda: 'hi',\n"
        "        mutates=False,\n"
        "    ))\n"
    )

    async def run():
        app = AgentTUI(
            workspace=Workspace(root=str(tmp_path / "workspace")),
            sessions_dir=str(tmp_path / "sessions"),
        )
        async with app.run_test():
            assert app.extensions is False
            assert "hello" not in app.agent.tools.names()

    asyncio.run(run())


def test_turn_done_without_streaming_still_mounts_a_block(tmp_path):
    """A turn that streamed nothing (e.g. on_delta unused) must still render its
    reply — the non-streaming path stays intact."""

    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test() as pilot:
            app._turn_done("plain reply")
            await pilot.pause()
            assert len(app.query(".msg-agent")) == 1

    asyncio.run(run())


def test_build_agent_closes_the_agent_it_just_built_when_setup_raises(tmp_path, monkeypatch):
    """``_build_agent`` constructs the Agent (allocating its scratch in
    ``Agent.__init__``) BEFORE its post-construction setup — ``_coding_tools``, and
    (when ``--extensions``) ``load_extensions``, which runs arbitrary user code from
    ``.carbon/extensions/``. Before this fix, a raise from that setup propagated
    straight out of ``_build_agent`` with ``agent.close()`` never called for the
    Agent it had just built.

    Spies on ``Agent.close`` directly rather than asserting the scratch directory
    is gone: every REAL TUI session is DURABLE (``session=`` is a non-empty string
    through the one shipped CLI entrypoint, ``tasks/tui.py``'s ``_resolve_session``,
    which maps even a missing/blank argument to ``"cli"``), and a durable
    ``cleanup()`` is BY DESIGN a no-op regardless of whether ``close()`` even runs
    (harness/session_env.py) — so a filesystem assertion here could not tell a
    fixed ``_build_agent`` from a broken one for the session shape production
    actually uses. What must hold unconditionally, durable or not, is that the
    Agent this method just constructed gets ``.close()``d when setup fails after it
    — matching the identical shape already established for ``run_once``/``_run_repl``
    (harness/agent.py).
    """
    import harness.agent as agent_mod
    from ui.tui import AgentTUI

    closed: list[bool] = []
    real_close = agent_mod.Agent.close

    def spy_close(self):
        closed.append(True)
        return real_close(self)

    monkeypatch.setattr(agent_mod.Agent, "close", spy_close)
    monkeypatch.setattr(
        agent_mod,
        "_coding_tools",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("setup blew up")),
    )

    try:
        AgentTUI(
            workspace=Workspace(root=str(tmp_path / "workspace")),
            sessions_dir=str(tmp_path / "sessions"),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the post-construction setup failure to propagate")

    assert closed, (
        "a raise during _build_agent's post-construction setup must close the Agent "
        "it already built, not leak it until the next process's scavenge()"
    )
