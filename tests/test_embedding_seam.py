"""The embedding seam (v0.1) — the surface external code builds agents on.

These prove the generic mechanisms the real consumers (crikit-agent, its eval
suite, the harness-editor) each hand-built: a structured run result, a per-call
tool record, a permission policy, registry introspection, an event stream,
config-schema introspection, provenance, schema output, and the curated façade.
All offline: the model is scripted through the provider's ``responder`` seam.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

import harness.harness_config as agent_config
import model.openai_compatible as oc
from harness.agent import Agent
from harness.harness_config import CONFIG, TruncationPolicy, config_schema
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


def _double_call(n: int) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[{"id": "1", "function": {"name": "double", "arguments": f'{{"n": {n}}}'}}],
    )


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
    return _scripted([_double_call(21), LLMResponse(content="done", finish_reason="stop")])


def _tools_with_double() -> ToolRegistry:
    """A minimal scripted tool for exercising generic Agent tool-loop mechanics
    (budget, approvals, structured results) — unrelated to any specific
    default tool, so it has no dependency on what default_tools() contains."""
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="double",
            description="",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            func=lambda n: str(n * 2),
            mutates=False,
        )
    )
    return reg


# --- T1.1 structured run result ----------------------------------------------
def test_run_returns_structured_result_and_events():
    a = Agent(provider=_tool_then_done(), tools=_tools_with_double())
    events: list[dict] = []
    a.subscribe(events.append)

    r = a.run("compute it")

    assert isinstance(r, RunResult)
    assert r.text == "done" and str(r) == "done"
    assert r.turns == 2 and r.stop_reason == "stop"
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc.name == "double" and tc.result == "42"
    assert tc.is_error is False and tc.attributes == {}  # carbon leaves the bag empty

    kinds = [e["type"] for e in events]
    assert kinds[0] == "turn_start" and "tool_call" in kinds and kinds[-1] == "turn_end"
    assert events[-1]["result"] is r


def test_send_still_returns_text():
    a = Agent(provider=fake(scripted=lambda m: "PONG"))
    assert a.send("hi") == "PONG"


def test_tool_budget_stop_reason():
    def always_tool(messages, **k) -> LLMResponse:
        return _double_call(1)

    p = Provider(base_url="fake://x", model="fake", api_key="x", responder=always_tool)
    r = Agent(provider=p, tools=_tools_with_double()).run("loop forever")
    assert r.stop_reason == "tool_budget"


# --- v0.3 per-instance tool-step budget ---------------------------------------
def test_max_tool_steps_overrides_the_module_global():
    calls = {"n": 0}

    def always_tool(messages, **k) -> LLMResponse:
        calls["n"] += 1
        return _double_call(1)

    p = Provider(base_url="fake://x", model="fake", api_key="x", responder=always_tool)
    r = Agent(provider=p, tools=_tools_with_double(), max_tool_steps=3).run("loop forever")

    assert r.stop_reason == "tool_budget"
    assert calls["n"] == 3


def test_max_tool_steps_default_none_leaves_behavior_unchanged():
    a = Agent(provider=fake(scripted=lambda m: "PONG"), max_tool_steps=None)
    r = a.run("hi")
    assert r.text == "PONG" and r.stop_reason == "stop"


# --- deadline_s: wall-clock bound, parallel to max_tool_steps -----------------
def test_deadline_stops_the_loop_before_starting_a_new_turn(monkeypatch):
    """Found needed live: an eval suite comparing this loop against a harness that
    DOES have a wall-clock bound (pi's 600s) needs the same bound here, or a raised
    tool-step budget just moves the ceiling from "turns" to "however long turns
    take" instead of adding a real one. The check runs BEFORE each new turn, not
    mid-turn — a blocking model call already in flight finishes rather than being
    torn down, mirroring how the tool-step budget has always worked."""
    import harness.agent as agent_mod

    calls = {"n": 0}

    def always_tool(messages, **k) -> LLMResponse:
        calls["n"] += 1
        return _calc_call("1+1")

    # Call 1 computes the deadline (t=0, deadline=0+5=5). Call 2 is the loop-top
    # check before turn 1 (t=0, under deadline — turn 1 runs). Call 3 is the
    # loop-top check before what would be turn 2 (t=10, past deadline — stop).
    seen = {"n": 0}

    def fake_monotonic():
        seen["n"] += 1
        return 0.0 if seen["n"] <= 2 else 10.0

    monkeypatch.setattr(agent_mod.time, "monotonic", fake_monotonic)

    p = Provider(base_url="fake://x", model="fake", api_key="x", responder=always_tool)
    r = Agent(provider=p, tools=default_tools(), deadline_s=5.0).run("loop forever")

    assert r.stop_reason == "deadline"
    assert calls["n"] == 1  # the one turn already in flight completed; no second


def test_deadline_default_none_leaves_behavior_unchanged():
    a = Agent(provider=fake(scripted=lambda m: "PONG"), deadline_s=None)
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

    # The subject is the per-tool BUDGET overriding the global one, so the strategy
    # that decides the SHAPE is named here rather than inherited. Read off the live
    # config, the head/tail assertion pins whatever ships: it fails under `keep_head`
    # (no tail), under `offload_to_file` (a footer lands last), and at either end of
    # the legal `tail_fraction` interval — four legal values, nothing broken.
    with patch(
        "harness.agent.CONFIG",
        replace(agent_config.CONFIG, tool_output=TruncationPolicy("head_tail", 4000, 0.6)),
    ):
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
        tools=_tools_with_double(),
        approval_required={"double"},
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
    """The default registry is exactly the three read-only exploration/reading
    tools — no arithmetic, nothing a coding agent doesn't need."""
    reg = default_tools()
    assert set(reg.names()) == {"read_file", "list_files", "search_text"}
    assert all(reg.get(name).mutates is False for name in reg.names())

    assert "harness/tools.py" in reg.call("list_files", '{"pattern": "harness/tools.py"}')
    assert "def default_tools" in reg.call("search_text", '{"query": "def default_tools"}')


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
    assert by["tool_output"]["strategies"] == ["head_tail", "keep_head", "offload_to_file"]
    assert by["compaction"]["strategies"] == [
        "structured_checkpoint",
        "summarize_middle",
        "token_budget_checkpoint",
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


def test_reasoning_effort_reaches_the_http_payload_as_a_nested_object(monkeypatch):
    """OpenRouter's unified reasoning control is `{"reasoning": {"effort": ...}}`,
    not a flat `reasoning_effort` field — confirmed against OpenRouter's own docs
    before wiring this up, not guessed."""
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
        Provider("http://x/v1", "m", "k", reasoning_effort="high"),
        [{"role": "user", "content": "hi"}],
    )
    assert captured["reasoning"] == {"effort": "high"}


def test_temperature_none_omits_the_field_entirely(monkeypatch):
    """temperature=None must not serialize as `"temperature": null` — the key
    itself is absent, so the provider applies its own default."""
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
        temperature=None,
    )
    assert "temperature" not in captured


def test_reasoning_effort_omitted_from_payload_when_not_set(monkeypatch):
    """Most models/providers neither support nor need this field — sending it
    unconditionally would risk a rejected/ignored parameter on every other call."""
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
    oc.complete_openai(Provider("http://x/v1", "m", "k"), [{"role": "user", "content": "hi"}])
    assert "reasoning" not in captured


def test_provider_from_env_reads_llm_reasoning_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    from model.provider import Provider as ProviderCls

    p = ProviderCls.from_env(root=tmp_path)
    assert p.reasoning_effort == "high"


def test_provider_from_env_defaults_reasoning_effort_to_none(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    from model.provider import Provider as ProviderCls

    p = ProviderCls.from_env(root=tmp_path)
    assert p.reasoning_effort is None


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
