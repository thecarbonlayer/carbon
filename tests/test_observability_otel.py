"""OTel GenAI span model (ch-13) — additive observability.

Asserts:
1. a recorded run produces an `invoke_agent` parent with `chat`/`execute_tool`
   children carrying the exact `gen_ai.*` attribute keys;
2. the exporter seam works (JsonlExporter to a tmp path + a capturing fake);
3. content capture is OFF by default and ON when `capture_content=True`;
4. the existing `timeline()`/`totals()`/Event-field API is unchanged (regression).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import harness.agent as agent_mod
from harness import events
from harness.events import JsonlExporter, NullExporter, Span
from harness.observability import Event, Tracer
from harness.tools import Tool, ToolRegistry
from model import LLMResponse


def _run_with_tool(tracer: Tracer) -> None:
    replies = iter(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        # Shape matches what an OpenAI-compatible server actually
                        # returns, `type` included — verified live against gemma.
                        "type": "function",
                        "id": "call-1",
                        "function": {"name": "double", "arguments": '{"n": 21}'},
                    }
                ],
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="42",
                usage={"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
                finish_reason="stop",
            ),
        ]
    )
    tools = ToolRegistry()
    tools.register(
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
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        agent_mod.Agent(tools=tools, tracer=tracer).send("compute it")


def test_run_produces_otel_spans_with_operation_names_and_attrs():
    tr = Tracer(model="google/gemma-4-26b-a4b")
    _run_with_tool(tr)
    spans = tr.get_spans()

    parents = [s for s in spans if s.operation == events.INVOKE_AGENT]
    chats = [s for s in spans if s.operation == events.CHAT]
    tools = [s for s in spans if s.operation == events.EXECUTE_TOOL]

    assert len(parents) == 1
    assert len(chats) == 2
    assert len(tools) == 1

    parent = parents[0]
    assert parent.parent_id is None
    assert parent.kind == events.CLIENT
    assert parent.attributes[events.OPERATION_NAME] == events.INVOKE_AGENT
    assert parent.attributes[events.AGENT_NAME] == "google/gemma-4-26b-a4b"
    # invoke_agent duration = sum of children's durations
    children = [s for s in spans if s.parent_id == parent.span_id]
    assert parent.duration_s == sum(c.duration_s for c in children)

    chat = chats[0]
    assert chat.parent_id == parent.span_id
    assert chat.kind == events.CLIENT
    assert chat.name.startswith("chat ")
    for key in (
        events.OPERATION_NAME,
        events.PROVIDER_NAME,
        events.REQUEST_MODEL,
        events.USAGE_INPUT_TOKENS,
        events.USAGE_OUTPUT_TOKENS,
        events.USAGE_COST,
    ):
        assert key in chat.attributes
    assert chat.attributes[events.OPERATION_NAME] == events.CHAT
    assert chat.attributes[events.USAGE_INPUT_TOKENS] == 5
    assert chat.attributes[events.USAGE_OUTPUT_TOKENS] == 2
    assert chat.attributes[events.RESPONSE_FINISH_REASONS] == ["tool_calls"]

    tool = tools[0]
    assert tool.parent_id == parent.span_id
    assert tool.kind == events.INTERNAL
    assert tool.attributes[events.OPERATION_NAME] == events.EXECUTE_TOOL
    assert tool.attributes[events.TOOL_NAME] == "double"
    assert tool.attributes[events.TOOL_TYPE] == "function"


def test_exporter_seam_jsonl(tmp_path):
    path = tmp_path / "spans.jsonl"
    tr = Tracer(model="gpt-4o-mini", exporter=JsonlExporter(str(path)))
    _run_with_tool(tr)
    tr.export()

    lines = path.read_text().strip().splitlines()
    assert lines
    rows = [json.loads(line) for line in lines]
    ops = {r["operation"] for r in rows}
    assert events.INVOKE_AGENT in ops and events.CHAT in ops and events.EXECUTE_TOOL in ops


def test_exporter_seam_capturing_fake():
    captured: list[Span] = []

    class FakeExporter:
        def export(self, spans):
            captured.extend(spans)

    tr = Tracer(model="m", exporter=FakeExporter())
    _run_with_tool(tr)
    tr.export()
    assert captured
    assert any(s.operation == events.INVOKE_AGENT for s in captured)


def test_content_off_by_default():
    tr = Tracer(model="m")
    assert tr.capture_content is False
    _run_with_tool(tr)
    chat = next(s for s in tr.get_spans() if s.operation == events.CHAT)
    assert events.INPUT_MESSAGES not in chat.attributes
    assert events.OUTPUT_MESSAGES not in chat.attributes
    tool = next(s for s in tr.get_spans() if s.operation == events.EXECUTE_TOOL)
    assert events.INPUT_MESSAGES not in tool.attributes


def test_content_on_when_enabled():
    tr = Tracer(model="m", capture_content=True)
    _run_with_tool(tr)
    chat = next(s for s in tr.get_spans() if s.operation == events.CHAT)
    assert events.INPUT_MESSAGES in chat.attributes
    assert events.OUTPUT_MESSAGES in chat.attributes
    tool = next(s for s in tr.get_spans() if s.operation == events.EXECUTE_TOOL)
    assert tool.attributes[events.INPUT_MESSAGES]
    assert tool.attributes[events.OUTPUT_MESSAGES]


def test_captured_content_keeps_the_tail_and_is_bounded_by_the_door_that_ran():
    """Two things the retention copies owe the person reading them.

    The tail, because a 24KB failing build ends in ``FATAL: undefined reference to
    main`` and content capture is opt-in precisely for debugging — head-only kept the
    compiler's banner and dropped the error. And the door's own bound, because a bare
    per-item clamp is faithful to nothing: it ignored both the configured tool_output
    budget and the tool's declared ``max_result_chars``, so the span could hold less
    than the model was given (a wider tool budget) or more (a narrower one).

    These are documented seams — ``subscribe()`` and the ``Tracer`` (adr/0002) — so this
    is what consumers are promised, not trace-size housekeeping.
    """
    from harness.harness_config import TruncationPolicy
    from harness.limits import MAX_ITEM_CHARS

    big = "BANNER\n" + "x" * 24_000 + "\nFATAL: undefined reference to main"
    replies = iter(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "c1", "function": {"name": "build", "arguments": '{"n": 1}'}}],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="build",
            description="",
            parameters={"type": "object", "properties": {"n": {"type": "integer"}}},
            func=lambda n: big,
            mutates=False,
        )
    )
    tr = Tracer(model="m", capture_content=True)
    seen: list[dict] = []
    door = TruncationPolicy("head_tail", 6_000, 0.5)  # deliberately not MAX_ITEM_CHARS
    with patch.object(agent_mod, "chat", side_effect=lambda *a, **k: next(replies)):
        a = agent_mod.Agent(tools=tools, tracer=tr, tool_output=door)
        a.subscribe(seen.append)
        a.send("build it")

    event = next(e for e in seen if e["type"] == "tool_call")["result"]
    span = next(s for s in tr.get_spans() if s.operation == events.EXECUTE_TOOL)
    captured = span.attributes[events.OUTPUT_MESSAGES]
    model_saw = next(m["content"] for m in a.messages if m.get("role") == "tool")

    for copy in (event, captured):
        assert copy.startswith("BANNER")
        assert copy.endswith("FATAL: undefined reference to main")  # the line that matters
        assert len(copy) <= door.budget + 100  # the door's budget, not a constant
        assert len(copy) > MAX_ITEM_CHARS  # …and this door was wider than that constant
    assert model_saw.endswith("FATAL: undefined reference to main")


def test_captured_args_are_bounded_the_same_way_as_the_result():
    """The other half of the same span. Bounding only the result left a 33MB write_file
    sitting in ``gen_ai.input.messages`` whole while its two-word result was trimmed —
    asymmetric, and the wrong way round."""
    tr = Tracer(model="m", capture_content=True)

    tr.record_tool("write_file", 0.1, args="A" * 50_000, result="wrote it", max_chars=1_000)

    span = next(s for s in tr.get_spans() if s.operation == events.EXECUTE_TOOL)
    assert len(span.attributes[events.INPUT_MESSAGES]) <= 1_100
    assert span.attributes[events.OUTPUT_MESSAGES] == "wrote it"


def test_tool_definitions_captured_once_per_turn():
    # The tool schemas handed to the model are part of the request. Capture them
    # on the turn's first chat span only — they don't change within a turn, and
    # repeating them on every tool-loop iteration just bloats the trace.
    tr = Tracer(model="m", capture_content=True)
    _run_with_tool(tr)
    chats = [s for s in tr.get_spans() if s.operation == events.CHAT]

    assert len(chats) == 2
    defs = chats[0].attributes[events.TOOL_DEFINITIONS]
    assert {d["function"]["name"] for d in defs} == {"double"}
    assert events.TOOL_DEFINITIONS not in chats[1].attributes


def test_tool_definitions_on_every_call_when_once_per_turn_is_off():
    # The once-per-turn default is a preference, not a constraint: a consumer that
    # wants each span self-contained for replay flips one flag.
    tr = Tracer(model="m", capture_content=True, tool_definitions_once_per_turn=False)
    _run_with_tool(tr)
    chats = [s for s in tr.get_spans() if s.operation == events.CHAT]

    assert len(chats) == 2
    assert all(events.TOOL_DEFINITIONS in c.attributes for c in chats)


def test_tool_definitions_absent_when_content_capture_off():
    tr = Tracer(model="m")
    _run_with_tool(tr)
    for chat in (s for s in tr.get_spans() if s.operation == events.CHAT):
        assert events.TOOL_DEFINITIONS not in chat.attributes


def test_captured_messages_preserve_tool_calls():
    # A turn that used tools has to round-trip: an assistant message carrying
    # tool_calls kept them, and its tool result kept the id that pairs them.
    tr = Tracer(model="m", capture_content=True)
    _run_with_tool(tr)
    chats = [s for s in tr.get_spans() if s.operation == events.CHAT]
    messages = chats[1].attributes[events.INPUT_MESSAGES]

    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call-1"
    result = next(m for m in messages if m["role"] == "tool")
    assert result["tool_call_id"] == "call-1"


def test_orchestrator_emits_plan_span_nested_under_turn():
    # ch-10 + ch-13: with a tracer, _plan records a `plan` span via the public
    # record_plan() (no reach into Tracer internals), nested under invoke_agent.
    from harness import orchestrator as orch_mod
    from harness.orchestrator import Orchestrator

    tr = Tracer(model="m")
    tr.turn_start()  # open the invoke_agent parent
    with patch.object(orch_mod, "chat", return_value=LLMResponse(content='["step a", "step b"]')):
        steps = Orchestrator(tracer=tr)._plan("do a thing")

    assert steps == ["step a", "step b"]
    plans = [s for s in tr.get_spans() if s.operation == events.PLAN]
    assert len(plans) == 1
    plan = plans[0]
    assert plan.kind == events.INTERNAL
    assert plan.attributes[events.OPERATION_NAME] == events.PLAN
    assert plan.duration_s >= 0.0
    parent = next(s for s in tr.get_spans() if s.operation == events.INVOKE_AGENT)
    assert plan.parent_id == parent.span_id  # nested, not orphaned


def test_orchestrator_without_tracer_emits_no_span_and_no_error():
    # tracer defaults to None => no plan span, no behavior change (accept ch-10 path).
    from harness import orchestrator as orch_mod
    from harness.orchestrator import Orchestrator

    with patch.object(orch_mod, "chat", return_value=LLMResponse(content='["x"]')):
        assert Orchestrator()._plan("t") == ["x"]


def test_existing_event_api_unchanged():
    # Regression guard: the flat Event/Tracer API is byte-for-byte preserved.
    tr = Tracer()
    assert isinstance(tr.exporter, NullExporter)
    tr.record_llm({"total_tokens": 10}, 0.5)
    tr.record_tool("calculator", 0.001, args='{"x": 1}', result="4")
    tr.record_verify(True, 0.002, "ok")

    field_names = {f for f in Event.__dataclass_fields__}
    assert field_names == {
        "kind",
        "label",
        "seconds",
        "tokens",
        "args",
        "result",
        "cost",
        "status",
        "turn",
    }

    totals = tr.totals()
    assert totals["llm_calls"] == 1 and totals["tool_calls"] == 1 and totals["tokens"] == 10
    assert "calculator" in tr.timeline()

    e = tr.events[1]
    assert e.kind == "tool" and e.args and e.result == "4"

    rows = tr.dump_events()
    tr2 = Tracer()
    tr2.load_events(rows)
    assert len(tr2.events) == len(tr.events)
