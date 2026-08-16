"""Boundary-case hardening (Phase 1 + Phase 2).

Defects that ``verify`` and ``accept`` don't exercise. Phase 1: crashes on a live
run (compaction orphaning tool-call groups, provider null fields, binary ``@file``,
a half-written session line, a quoted `.env`). Phase 2: trust + intent gaps
(reading secrets, dishonest approval preview, mispriced cost, non-atomic saves,
memory recall that ignores its own stated scope). Each fails pre-fix, passes after.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from harness import compaction
from harness.compaction import _clean_cut, compact
from harness.context import deliver
from harness.memory import load_session, save_session
from model import LLMResponse


# --- compaction: never orphan a tool-call group ------------------------------
def _roles_valid(messages: list[dict]) -> bool:
    """Every assistant.tool_calls is immediately followed by its tool results, and
    no tool message is dangling — the invariant an OpenAI-compatible API enforces."""
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            prev = messages[i - 1] if i > 0 else {}
            if prev.get("role") not in ("assistant", "tool"):
                return False
        if m.get("role") == "assistant" and m.get("tool_calls"):
            nxt = messages[i + 1] if i + 1 < len(messages) else {}
            if nxt.get("role") != "tool":
                return False
    return True


def _tool_session() -> list[dict]:
    # A first turn that uses a tool — the deterministic head-split case.
    return [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "result a"},
        {"role": "assistant", "content": "did it"},
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "b", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "b", "content": "result b"},
        {"role": "assistant", "content": "done"},
    ]


def test_compact_does_not_orphan_tool_calls_on_head_split():
    msgs = _tool_session()
    assert _roles_valid(msgs)  # input is well-formed
    with patch.object(compaction, "chat", return_value=LLMResponse(content="SUMMARY")):
        out = compact(msgs, keep_head=2, keep_tail=4)
    assert _roles_valid(out), out


def test_compact_does_not_orphan_tool_result_on_tail_split():
    # keep_tail lands the tail boundary on a `tool` message whose assistant is
    # about to be summarized away.
    msgs = _tool_session()
    with patch.object(compaction, "chat", return_value=LLMResponse(content="SUMMARY")):
        out = compact(msgs, keep_head=1, keep_tail=2)
    assert _roles_valid(out), out


def test_clean_cut_rejects_splitting_a_tool_group():
    msgs = _tool_session()
    assert _clean_cut(msgs, 2) is False  # index 2 is a tool result
    assert _clean_cut(msgs, 1) is True  # before the assistant(tool_calls) is fine


def test_compact_leaves_plain_history_boundaries_unchanged():
    # No tool calls → snapping is a no-op, original head/tail semantics hold.
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    with patch.object(compaction, "chat", return_value=LLMResponse(content="SUMMARY")):
        out = compact(msgs, keep_head=2, keep_tail=2)
    assert out[0] == msgs[0] and out[1] == msgs[1]
    assert out[-1] == msgs[-1] and out[-2] == msgs[-2]


# --- provider parsing: missing / null response fields ------------------------
def _fake_post(payload: dict):
    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return payload

    return _Resp()


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "rate limited"}},  # HTTP 200 with an error body
        {"choices": []},  # empty choices (content filter)
    ],
)
def test_no_choices_raises_clear_error(body):
    from model.openai_compatible import complete_openai
    from model.provider import Provider

    prov = Provider(base_url="http://x/v1", model="m")
    with patch("model.openai_compatible.httpx.post", return_value=_fake_post(body)):
        with pytest.raises(RuntimeError, match="no choices"):
            complete_openai(prov, [{"role": "user", "content": "hi"}])


def test_null_message_and_usage_do_not_crash():
    from model.openai_compatible import complete_openai
    from model.provider import Provider

    body = {"choices": [{"message": None, "finish_reason": "content_filter"}], "usage": None}
    prov = Provider(base_url="http://x/v1", model="m")
    with patch("model.openai_compatible.httpx.post", return_value=_fake_post(body)):
        resp = complete_openai(prov, [{"role": "user", "content": "hi"}])
    assert resp.content == ""
    assert resp.usage == {}  # never None — downstream token accounting relies on this


# --- context: a binary @file must not crash the turn -------------------------
def test_binary_attachment_is_skipped_not_crashed(tmp_path):
    img = tmp_path / "logo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binarybytes\xff\xfe")
    blocks = deliver(f"look at @{img} please")
    assert blocks == []  # skipped silently, no UnicodeDecodeError


# --- memory: one bad line must not brick resume ------------------------------
def test_load_session_survives_a_truncated_last_line(tmp_path):
    save_session(
        "s",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        base=tmp_path,
    )
    # Simulate a kill mid-write: append a half-written line.
    path = tmp_path / "s.jsonl"
    with path.open("a") as f:
        f.write('{"role": "assist')
    out = load_session("s", base=tmp_path)
    assert out == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


# --- .env parsing: quotes, comments, export ----------------------------------
def test_dotenv_strips_quotes_comments_and_export(tmp_path, monkeypatch):
    from model import provider as provider_mod

    env = tmp_path / ".env"
    env.write_text(
        'LLM_API_KEY="sk-secret-123"\n'
        "LLM_MODEL=gpt-4o # prod model\n"
        "export LLM_BASE_URL=https://api.example.com/v1\n"
    )
    monkeypatch.chdir(tmp_path)
    for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    cfg = provider_mod.Provider.from_env()
    assert cfg.api_key == "sk-secret-123"  # no literal quotes in the auth header
    assert cfg.model == "gpt-4o"  # inline comment stripped
    assert cfg.base_url == "https://api.example.com/v1"  # export honored


# ============================ Phase 2: trust + intent ========================


# --- read_file: confinement, secrets denylist, and root binding --------------
def test_read_file_refuses_env_and_keys(tmp_path):
    from harness.tools import read_file

    (tmp_path / ".env").write_text("LLM_API_KEY=sk-secret")
    (tmp_path / "id_ed25519").write_text("PRIVATE KEY")
    (tmp_path / "ok.txt").write_text("fine")
    assert "refusing to read secret" in read_file(".env", root=tmp_path)
    assert "refusing to read secret" in read_file("id_ed25519", root=tmp_path)
    assert read_file("ok.txt", root=tmp_path) == "fine"


def test_read_file_confined_to_its_root(tmp_path):
    from harness.tools import read_file

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    assert "outside workspace" in read_file(str(outside), root=tmp_path)


def test_read_file_tool_binds_root(tmp_path):
    from harness.tools import read_file_tool

    (tmp_path / "note.txt").write_text("hello")
    tool = read_file_tool(str(tmp_path))
    assert tool.func(path="note.txt") == "hello"


def test_read_file_resolves_scratch_ref_confined_to_scratch(tmp_path):
    from harness.tools import read_file

    scratch = tmp_path / "scratch"
    (scratch / "offload").mkdir(parents=True)
    (scratch / "offload" / "ab12.txt").write_text("the complete bytes")
    ws = tmp_path / "ws"
    ws.mkdir()
    body = read_file("scratch://offload/ab12.txt", root=ws, scratch_root=scratch)
    assert "the complete bytes" in body
    # escape attempts stay inside scratch
    err = read_file("scratch://../ws/anything", root=ws, scratch_root=scratch)
    assert err.startswith("error:")
    # no scratch configured -> a clear error, not a workspace fallback
    err2 = read_file("scratch://offload/ab12.txt", root=ws, scratch_root=None)
    assert err2.startswith("error:")


# --- approval preview is honest about a failing edit -------------------------
def test_approval_preview_shows_edit_failure_not_no_change(tmp_path):
    from harness.workspace import Workspace
    from ui.tui import approval_preview

    ws = Workspace(root=tmp_path)
    ws.write("a.txt", "hello world")
    # `old` not present → the real edit errors; the preview must say so, not "(no change)".
    args = '{"path": "a.txt", "old": "NOPE", "new": "x"}'
    preview = approval_preview("edit_file", args, ws)
    assert "not found" in preview["body"]
    assert preview["body"] != "(no change)"
    # missing file → also surfaced
    preview2 = approval_preview("edit_file", '{"path": "gone.txt", "old": "a", "new": "b"}', ws)
    assert "no such file" in preview2["body"]


# --- cost is priced from the model actually called ---------------------------
def test_record_llm_prices_from_request_model():
    from harness.observability import Tracer

    tracer = Tracer()  # no model set (the REPL's old shape) → used to cost $0
    tracer.turn_start()
    tracer.record_llm(
        {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
        0.1,
        request_model="gpt-4o",
    )
    assert tracer.totals()["cost"] > 0  # priced from request_model, not the empty self.model


# --- atomic save: a crash mid-write can't shred the prior good file ----------
def test_save_session_is_atomic(tmp_path):
    save_session("s", [{"role": "user", "content": "one"}], base=tmp_path)
    # Simulate os.replace failing mid-save; the original file must survive intact.
    with patch("harness.memory.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            save_session("s", [{"role": "user", "content": "two"}], base=tmp_path)
    assert load_session("s", base=tmp_path) == [{"role": "user", "content": "one"}]
    # no stray temp files left behind
    assert list(tmp_path.glob("*.tmp")) == []


def test_session_id_cannot_escape_base(tmp_path):
    from harness.memory import _path

    p = _path("../escaped", base=tmp_path)
    assert p.parent == tmp_path  # traversal stripped, stays inside base


# --- search_sessions honors its stated scope ---------------------------------
def test_search_excludes_current_session(tmp_path):
    from harness.memory import search_sessions

    save_session("cur", [{"role": "user", "content": "the passcode is GOGO"}], base=tmp_path)
    save_session("old", [{"role": "user", "content": "the passcode is GOGO"}], base=tmp_path)
    hits = search_sessions("passcode", base=tmp_path, exclude="cur")
    assert [h["session"] for h in hits] == ["old"]  # current session not surfaced


def test_search_tolerates_punctuation_in_query(tmp_path):
    from harness.memory import search_sessions

    save_session(
        "s", [{"role": "user", "content": "the warehouse passcode is GOGO-77"}], base=tmp_path
    )
    hits = search_sessions("warehouse passcode?", base=tmp_path)
    assert hits and "GOGO-77" in hits[0]["content"]


# --- estimate_tokens counts tool-call arguments ------------------------------
def test_estimate_counts_tool_call_args():
    from harness.compaction import estimate_tokens

    big_args = "x" * 400
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"arguments": big_args}}],
    }
    assert estimate_tokens([msg]) > 50  # the 400-char args are measured, not ignored


# --- fan_out rejects a non-list argument -------------------------------------
def test_fan_out_rejects_non_list():
    from harness.subagents import fan_out_tool

    tool = fan_out_tool()
    assert "must be a list" in tool.func(tasks="not a list")


# --- workspace edit rejects an empty `old` -----------------------------------
def test_edit_rejects_empty_old(tmp_path):
    from harness.workspace import Workspace

    ws = Workspace(root=tmp_path)
    ws.write("a.txt", "hello")
    assert "must be non-empty" in ws.edit("a.txt", "", "X")
    assert ws.read("a.txt") == "hello"  # unchanged, not prepended


# ======================= Phase 3: verification integrity =====================
#
# The two-gate rule is the repo's thesis. These lock the gate against the ways it
# could be satisfied WITHOUT the tests actually validating the final code.

import json as _json  # noqa: E402

import harness.agent as _agent_mod  # noqa: E402
from harness.tools import default_tools as _default_tools  # noqa: E402
from harness.workspace import Workspace, write_file_tool  # noqa: E402

_AGENTS = "## Testing\n```\npython3 test_thing.py\n```\n"


def _call(cid, name, args):
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)},
            }
        ],
    )


def _gate_agent(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.write("AGENTS.md", _AGENTS)
    ws.write("test_thing.py", "print('ok')\n")
    tools = _default_tools()
    tools.register(write_file_tool(ws))
    from harness.sandbox import Sandbox, bash_tool

    tools.register(bash_tool(Sandbox(trusted=True), workdir=str(ws.root)))
    return _agent_mod.Agent(tools=tools, agents_dir=str(ws.root), verify_attempts=2)


def test_gate_rejects_echoed_command(tmp_path):
    """`echo 'python3 test_thing.py'` exits 0 but doesn't run the tests — must not pass."""
    a = _gate_agent(tmp_path)
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            _call("c2", "bash", {"command": "echo 'python3 test_thing.py'"}),
            LLMResponse(content="done"),
            LLMResponse(content="still done"),  # reprompt run 1
            LLMResponse(content="really done"),  # reprompt run 2
        ]
    )
    with patch.object(_agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        out = a.send("write foo.py")
    assert not a._observed_pass("python3 test_thing.py", 0)  # echo doesn't count
    assert "unverified" in out  # fail-closed, not a clean "done"


def test_gate_rejects_chained_command(tmp_path):
    a = _gate_agent(tmp_path)
    assert not a._is_test_run(
        '{"command": "python3 test_thing.py || true"}', "python3 test_thing.py"
    )
    assert not a._is_test_run(
        '{"command": "python3 test_thing.py; echo hi"}', "python3 test_thing.py"
    )
    assert a._is_test_run('{"command": "python3 test_thing.py -v"}', "python3 test_thing.py")


def test_gate_requires_run_after_last_mutation(tmp_path):
    """A pass BEFORE the final edit doesn't verify the edit."""
    a = _gate_agent(tmp_path)
    # run tests (pass), THEN edit code again, then narrate done without re-running.
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            _call("c2", "bash", {"command": "python3 test_thing.py"}),  # passes here
            _call(
                "c3", "write_file", {"path": "foo.py", "content": "x = 2  # changed after test\n"}
            ),
            LLMResponse(content="done"),
            LLMResponse(content="still done"),  # reprompt run 1
            LLMResponse(content="really done"),  # reprompt run 2
        ]
    )
    with patch.object(_agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        out = a.send("write foo.py")
    # the passing run predates the last mutation → not accepted as verification
    assert not a._observed_pass("python3 test_thing.py", 0)
    assert "unverified" in out


def test_gate_fails_closed_when_never_run(tmp_path):
    """Model changes code and never runs tests → reply is marked unverified."""
    a = _gate_agent(tmp_path)
    replies = iter(
        [
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            LLMResponse(content="all done, trust me"),
            LLMResponse(content="still trust me"),
            LLMResponse(content="really"),
        ]
    )
    with patch.object(_agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        out = a.send("write foo.py")
    assert "unverified" in out


def test_compaction_does_not_disable_the_gate(tmp_path):
    """Compaction mid-turn used to renumber messages and make the gate read an empty
    slice (fail-open). With compaction at turn-start, the gate still fires."""
    a = _gate_agent(tmp_path)
    a.context_limit = 1  # force compaction at the next turn-start
    # 8 messages (> keep_head 2 + keep_tail 4) so compaction actually summarizes.
    a.messages = [{"role": "user", "content": f"old {i}"} for i in range(8)]
    replies = iter(
        [
            LLMResponse(content="SUMMARY"),  # compaction summarizer call
            _call("c1", "write_file", {"path": "foo.py", "content": "x = 1\n"}),
            LLMResponse(content="done without running"),
            LLMResponse(content="still not running"),
            LLMResponse(content="nope"),
        ]
    )
    driver = lambda *a, **k: next(replies)  # noqa: E731
    with (
        patch.object(_agent_mod, "chat", side_effect=driver),
        patch.object(compaction, "chat", side_effect=driver),
    ):
        out = a.send("now change code")
    assert a.just_compacted  # compaction did fire this turn
    assert "unverified" in out  # and the gate still caught the un-run change


# --- sandbox: a timeout kills the whole process tree -------------------------
def test_sandbox_timeout_kills_process_group(tmp_path):
    import time as _time

    from harness.sandbox import Sandbox

    sb = Sandbox(trusted=True, timeout=1, prefer_docker=False)
    marker = tmp_path / "alive.txt"
    # background child that writes the marker after the parent's timeout would fire
    r = sb.run(f"(sleep 3; touch {marker}) & echo started", workdir=str(tmp_path))
    assert r.exit_code == 124  # timed out
    _time.sleep(3.5)  # wait past when the orphaned child would have written
    assert not marker.exists(), "background descendant survived the timeout"


# --- verification: the success sentinel can't be forged by an early exit -----
def test_run_python_early_exit_cannot_forge_pass():
    from harness.verification import run_python

    # print the OLD fixed sentinel and exit before the assertion runs — must NOT pass.
    res = run_python("print('VERIFICATION_OK')\nimport sys; sys.exit(0)", "assert False")
    assert res.passed is False


def test_run_python_still_passes_a_real_check():
    from harness.verification import run_python

    res = run_python("x = 2 + 2", "assert x == 4")
    assert res.passed is True


# --- /plan renders in the trace pane: record_plan emits a flat event ----------
def test_record_plan_emits_event_for_trace_pane():
    from harness.observability import Tracer

    tr = Tracer()
    tr.turn_start()
    tr.record_plan(0.05)
    # the trace pane renders tracer.events (not spans) — a plan step must appear there
    assert any(e.kind == "plan" for e in tr.events)
    # and still produces the OTel plan span
    assert any(s.operation == "plan" for s in tr.spans)
