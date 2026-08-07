"""The embedding seam (v0.1) — the surface external code builds agents on.

These prove the generic mechanisms the real consumers (crikit-agent, its eval
suite, the harness-editor) each hand-built: a structured run result, a per-call
tool record, a permission policy, registry introspection, an event stream,
config-schema introspection, provenance, schema output, and the curated façade.
All offline: the model is scripted through the provider's ``responder`` seam.
"""

from __future__ import annotations

import pytest

import model.openai_compatible as oc
from harness.agent import Agent
from harness.harness_config import CONFIG, config_schema
from harness.policy import Policy
from harness.provenance import provenance
from harness.result import RunResult
from harness.tools import Tool, ToolRegistry, default_tools
from model import LLMResponse, Provider, chat, fake

_EMPTY_PARAMS = {"type": "object", "properties": {}}


def _tool_call(name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        content="", tool_calls=[{"id": "1", "function": {"name": name, "arguments": args}}]
    )


def _scripted(responses: list[LLMResponse]) -> Provider:
    """A provider that returns each scripted ``LLMResponse`` in order (no network)."""
    it = iter(responses)

    def responder(messages, **kwargs) -> LLMResponse:
        return next(it)

    return Provider(base_url="fake://x", model="fake", api_key="x", responder=responder)


def _calc_call(expr: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "1",
                "function": {"name": "calculator", "arguments": f'{{"expression": "{expr}"}}'},
            }
        ],
    )


def _tool_then_done() -> Provider:
    return _scripted([_calc_call("6*7"), LLMResponse(content="done", finish_reason="stop")])


def _tools_with_calculator() -> ToolRegistry:
    """calculator is opt-in (not a default tool) — tests exercising it via the
    scripted ``_calc_call`` provider register it explicitly."""
    from harness.tools import calculator_tool

    reg = default_tools()
    reg.register(calculator_tool())
    return reg


# --- T1.1 structured run result ----------------------------------------------
def test_run_returns_structured_result_and_events():
    a = Agent(provider=_tool_then_done(), tools=_tools_with_calculator())
    events: list[dict] = []
    a.subscribe(events.append)

    r = a.run("compute it")

    assert isinstance(r, RunResult)
    assert r.text == "done" and str(r) == "done"
    assert r.turns == 2 and r.stop_reason == "stop"
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc.name == "calculator" and tc.result == "42"
    assert tc.is_error is False and tc.attributes == {}  # carbon leaves the bag empty

    kinds = [e["type"] for e in events]
    assert kinds[0] == "turn_start" and "tool_call" in kinds and kinds[-1] == "turn_end"
    assert events[-1]["result"] is r


def test_send_still_returns_text():
    a = Agent(provider=fake(scripted=lambda m: "PONG"))
    assert a.send("hi") == "PONG"


def test_tool_budget_stop_reason():
    def always_tool(messages, **k) -> LLMResponse:
        return _calc_call("1+1")

    p = Provider(base_url="fake://x", model="fake", api_key="x", responder=always_tool)
    r = Agent(provider=p, tools=_tools_with_calculator()).run("loop forever")
    assert r.stop_reason == "tool_budget"


# --- v0.3 per-instance tool-step budget ---------------------------------------
def test_max_tool_steps_overrides_the_module_global():
    calls = {"n": 0}

    def always_tool(messages, **k) -> LLMResponse:
        calls["n"] += 1
        return _calc_call("1+1")

    p = Provider(base_url="fake://x", model="fake", api_key="x", responder=always_tool)
    r = Agent(provider=p, tools=_tools_with_calculator(), max_tool_steps=3).run("loop forever")

    assert r.stop_reason == "tool_budget"
    assert calls["n"] == 3


def test_max_tool_steps_default_none_leaves_behavior_unchanged():
    a = Agent(provider=fake(scripted=lambda m: "PONG"), max_tool_steps=None)
    r = a.run("hi")
    assert r.text == "PONG" and r.stop_reason == "stop"


# --- v0.2 tool metadata + per-tool truncation --------------------------------
def test_tool_attributes_seed_into_each_call():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="ping",
            description="",
            parameters=_EMPTY_PARAMS,
            func=lambda: "pong",
            attributes={"tier": "domain"},
        )
    )
    p = _scripted([_tool_call("ping"), LLMResponse(content="done", finish_reason="stop")])

    r = Agent(provider=p, tools=reg).run("go")

    assert r.tool_calls[0].attributes == {"tier": "domain"}
    # a fresh copy: mutating one call's bag must not touch the tool's static data
    r.tool_calls[0].attributes["extra"] = 1
    assert reg.get("ping").attributes == {"tier": "domain"}


def test_per_tool_truncation_budget():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="big",
            description="",
            parameters=_EMPTY_PARAMS,
            func=lambda: "X" * 500,
            max_result_chars=10,
        )
    )
    p = _scripted([_tool_call("big"), LLMResponse(content="done", finish_reason="stop")])

    r = Agent(provider=p, tools=reg).run("go")

    res = r.tool_calls[0].result
    assert res.startswith("X" * 4) and res.endswith("X" * 6)
    assert "truncated" in res and len(res) < 100


# --- T1.3 permission policy ---------------------------------------------------
def test_policy_decisions():
    assert Policy().decision("x", "")[0] is True
    assert Policy(deny=frozenset({"x"})).decision("x", "")[0] is False
    assert Policy(allow=frozenset({"y"})).decision("x", "")[0] is False
    assert Policy(read_only=True).decision("write_file", "")[0] is False
    assert Policy(read_only=True).decision("read_file", "")[0] is True

    ok, _ = Policy(require_approval=frozenset({"x"}), approve=lambda n, a: True).decision("x", "")
    assert ok is True
    denied, marker = Policy(require_approval=frozenset({"x"})).decision("x", "")  # no approver
    assert denied is False and "approval gate" in marker


def test_approvals_counted_via_backcompat_args():
    a = Agent(
        provider=_tool_then_done(),
        tools=_tools_with_calculator(),
        approval_required={"calculator"},
        approve=lambda n, a: True,
    )
    r = a.run("compute it")
    assert r.approvals == 1 and r.tool_calls[0].result == "42"


# --- T1.4 registry introspection ---------------------------------------------
def test_registry_get_names_wrap():
    reg = default_tools()
    assert "read_file" in reg.names()
    assert reg.get("read_file") is not None and reg.get("nope") is None

    reg.register(Tool(name="echo", description="", parameters=_EMPTY_PARAMS, func=lambda: "hi"))
    reg.wrap("echo", lambda fn: lambda **kw: "WRAPPED:" + fn(**kw))
    assert reg.call("echo", "{}") == "WRAPPED:hi"

    with pytest.raises(KeyError):
        reg.wrap("nope", lambda fn: fn)


def test_default_tools_are_read_only_exploration_and_reading():
    """calculator is a teaching artifact, not a coding-agent need — it must be
    opt-in via calculator_tool(), not silently present in the default registry."""
    from harness.tools import calculator_tool

    reg = default_tools()
    assert set(reg.names()) == {"read_file", "list_files", "search_text"}
    assert all(reg.get(name).mutates is False for name in reg.names())

    reg.register(calculator_tool())
    assert reg.call("calculator", '{"expression": "6 * 7"}') == "42"


# --- T1.6 config schema -------------------------------------------------------
def test_config_schema_describes_the_surface():
    by = {f["name"]: f for f in config_schema()}
    assert set(by) == {
        "version",
        "system_prompt",
        "max_tool_steps",
        "default_context_limit",
        "approval_tools",
        "code_extensions",
        "verify_attempts",
        "require_run",
        "max_item_chars",
        "file_injection",
        "tool_output",
        "compaction",
        "retry",
        "compaction_prompt",
        "memory_search_limit",
        "attach_pattern",
        "temperature",
        "max_tokens",
    }
    assert by["approval_tools"]["collection"] and by["approval_tools"]["type"] == "list[str]"
    assert by["max_tool_steps"]["positive_int"] and by["max_tool_steps"]["type"] == "int"
    assert by["require_run"]["type"] == "bool" and not by["require_run"]["positive_int"]
    assert by["tool_output"]["strategies"] == ["head_tail", "keep_head"]
    assert by["compaction"]["strategies"] == [
        "structured_checkpoint",
        "summarize_middle",
    ]
    assert by["retry"]["strategies"] == ["backoff", "fail_fast"]


# --- T1.5 provenance + schema output -----------------------------------------
def test_provenance_returns_identity_primitives():
    pv = provenance(model="gemma-x", root=".")
    assert pv["config_version"] == CONFIG.version
    assert pv["model"] == "gemma-x"
    assert pv["gemma_sha"] is None or isinstance(pv["gemma_sha"], str)


def test_chat_forwards_response_format_to_responder():
    seen: dict = {}

    def responder(messages, **kwargs) -> LLMResponse:
        seen.update(kwargs)
        return LLMResponse(content="ok")

    p = Provider(base_url="fake://x", model="m", api_key="x", responder=responder)
    chat([{"role": "user", "content": "hi"}], provider=p, response_format={"type": "json_object"})
    assert seen["response_format"] == {"type": "json_object"}


def test_response_format_reaches_the_http_payload(monkeypatch):
    captured: dict = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }

    def fake_post(url, json, headers, timeout):  # noqa: A002 — mirrors httpx.post kwarg
        captured.update(json)
        return _Resp()

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    oc.complete_openai(
        Provider("http://x/v1", "m", "k"),
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert captured["response_format"] == {"type": "json_object"}


# --- T1.7 curated façade ------------------------------------------------------
def test_carbon_facade_exports_the_surface():
    import carbon

    assert carbon.__version__ == "0.4.0"
    for name in (
        "Agent",
        "RunResult",
        "ToolCall",
        "Policy",
        "Tool",
        "ToolRegistry",
        "Provider",
        "chat",
        "load_config",
        "config_schema",
        "surface_manifest",
        "provenance",
        "load_env",
        "Tracer",
    ):
        assert hasattr(carbon, name), name
    # the façade re-exports the same objects, it does not fork them
    from harness.agent import Agent as HarnessAgent

    assert carbon.Agent is HarnessAgent
