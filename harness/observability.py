"""Observability (ch-13).

A trace of every step — model calls (tokens, latency, cost) and tool calls (args,
result, status) — so a multi-step run is replayable. The interesting bug is usually
a few tool calls before the failure; you can't tune what you can't see.

The ``Tracer`` carries the bits a UI needs to *show* a run rather than print it:
- ``cost`` per model call (priced from usage), and ``status`` per step (ok /
  denied / error / pass / fail) so the trace can be color-coded;
- verify steps are recorded (``record_verify``) so the self-verify loop is visible;
- a ``turn`` index (bumped by ``turn_start``) so events nest under their turn;
- an ``on_event`` hook so a live UI can refresh as each event lands.
All of it is additive — pass no tracer and the loop is unchanged.

Alongside the flat ``Event`` list, the tracer builds an **OTel GenAI span tree** —
same single emit, two shapes. ``turn_start`` opens an ``invoke_agent`` parent span;
``record_llm`` adds a child ``chat`` span; ``record_tool`` adds a child
``execute_tool`` span — each carrying the exact ``gen_ai.*`` attribute names from
``harness.events``. An exporter seam (``SpanExporter``) mirrors the provider seam:
the default is a no-op, but the same spans can graduate to OTLP.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields

from harness import events
from harness.events import NullExporter, Span, SpanExporter
from harness.limits import MAX_ITEM_CHARS, clamp, recut
from model.pricing import cost_from_usage

# The head/tail split for captured content. Half and half: a captured result is read to
# find out what happened, and what happened is as often in the last line (`FATAL: …`,
# an exit summary) as in the first.
_CAPTURE_TAIL_FRACTION = 0.5


@dataclass
class Event:
    kind: str  # "llm" | "tool" | "verify"
    label: str
    seconds: float
    tokens: int = 0
    args: str = ""  # tool input
    result: str = ""  # tool output
    cost: float = 0.0  # USD for this step
    status: str = ""  # ok | denied | error | pass | fail
    turn: int = 0  # which user turn this step belongs to


# Message fields beyond role/content that link a tool call to the result it produced.
_TOOL_LINK_KEYS = ("tool_calls", "tool_call_id", "name")


def _capture_content_default() -> bool:
    """OTel makes message content opt-in and off by default (privacy)."""
    flag = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "")
    return flag.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Tracer:
    events: list[Event] = field(default_factory=list)
    model: str | None = None  # used to price llm calls
    on_event: Callable[[Event], None] | None = None  # live-UI hook
    # --- OTel span model (additive) ------------------------------------------
    spans: list[Span] = field(default_factory=list)
    exporter: SpanExporter = field(default_factory=NullExporter)
    provider_name: str = "openai"  # gen_ai.provider.name (OpenAI-compatible flavor)
    server_address: str | None = None
    server_port: int | None = None
    conversation_id: str | None = None  # our session id → gen_ai.conversation.id
    capture_content: bool = field(default_factory=_capture_content_default)
    # Tool defs don't change within a turn, so the default records them once. Set
    # False to repeat them on every chat span, making each span replayable alone.
    tool_definitions_once_per_turn: bool = True
    _turn: int = 0
    _span_seq: int = 0
    _turn_span_id: str | None = None  # current invoke_agent parent
    _turn_tool_defs_emitted: bool = False  # tool defs are captured once per turn
    # Phase 1 telemetry slice 1 (contract §2): the per-call input/output split
    # already computed in ``_add_chat_span``, accumulated for ``totals()``.
    _input_tokens: int = 0
    _output_tokens: int = 0

    def turn_start(self) -> None:
        """Begin a new user turn; subsequent events nest under it.

        Also opens a per-turn ``invoke_agent`` parent span. Its ``duration_s`` is
        filled in lazily from the sum of its children when spans are assembled.
        """
        self._turn += 1
        span_id = self._next_span_id()
        attrs: dict = {
            events.OPERATION_NAME: events.INVOKE_AGENT,
            events.AGENT_NAME: self.model or "agent",
        }
        if self.conversation_id is not None:
            attrs[events.CONVERSATION_ID] = self.conversation_id
        self.spans.append(
            Span(
                span_id=span_id,
                parent_id=None,
                name=f"invoke_agent {self.model or 'agent'}",
                kind=events.CLIENT,
                operation=events.INVOKE_AGENT,
                attributes=attrs,
            )
        )
        self._turn_span_id = span_id
        self._turn_tool_defs_emitted = False

    def _next_span_id(self) -> str:
        self._span_seq += 1
        return f"span-{self._span_seq}"

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def record_llm(
        self,
        usage: dict,
        seconds: float,
        *,
        finish_reason: str | None = None,
        request_model: str | None = None,
        response_id: str | None = None,
        messages: list[dict] | None = None,
        output: str | None = None,
        tool_definitions: list[dict] | None = None,
    ) -> None:
        # Price with the model actually called; ``self.model`` is the fallback when
        # the caller doesn't pass one. (A model-less tracer would otherwise cost $0.)
        cost = cost_from_usage(request_model or self.model, usage)
        self._emit(
            Event(
                "llm",
                "model call",
                seconds,
                tokens=int(usage.get("total_tokens", 0)),
                cost=cost,
                status="ok",
                turn=self._turn,
            )
        )
        self._add_chat_span(
            usage,
            seconds,
            cost,
            finish_reason=finish_reason,
            request_model=request_model,
            response_id=response_id,
            messages=messages,
            output=output,
            tool_definitions=tool_definitions,
        )

    def _add_chat_span(
        self,
        usage: dict,
        seconds: float,
        cost: float,
        *,
        finish_reason: str | None,
        request_model: str | None,
        response_id: str | None,
        messages: list[dict] | None,
        output: str | None,
        tool_definitions: list[dict] | None = None,
    ) -> None:
        model = request_model or self.model
        in_tokens = int(usage.get("prompt_tokens", 0) or 0)
        out_tokens = int(usage.get("completion_tokens", 0) or 0)
        if not in_tokens and not out_tokens:
            in_tokens = int(usage.get("total_tokens", 0) or 0)
        self._input_tokens += in_tokens
        self._output_tokens += out_tokens
        attrs: dict = {
            events.OPERATION_NAME: events.CHAT,
            events.PROVIDER_NAME: self.provider_name,
            events.REQUEST_MODEL: model,
            events.USAGE_INPUT_TOKENS: in_tokens,
            events.USAGE_OUTPUT_TOKENS: out_tokens,
            events.USAGE_COST: cost,  # our extension
        }
        if finish_reason is not None:
            attrs[events.RESPONSE_FINISH_REASONS] = [finish_reason]
        if response_id is not None:
            attrs[events.RESPONSE_ID] = response_id
        if self.server_address is not None:
            attrs[events.SERVER_ADDRESS] = self.server_address
        if self.server_port is not None:
            attrs[events.SERVER_PORT] = self.server_port
        if self.conversation_id is not None:
            attrs[events.CONVERSATION_ID] = self.conversation_id
        if self.capture_content:
            if messages is not None:
                attrs[events.INPUT_MESSAGES] = self._content_messages(messages)
                sys_text = next(
                    (m.get("content", "") for m in messages if m.get("role") == "system"), ""
                )
                if sys_text:
                    attrs[events.SYSTEM_INSTRUCTIONS] = sys_text
            if output is not None:
                attrs[events.OUTPUT_MESSAGES] = output
            # The tool menu is identical for every call in a turn, so by default
            # record it on the turn's first chat span rather than once per
            # tool-loop iteration. Opt out for span-level self-containment.
            once = self.tool_definitions_once_per_turn
            if tool_definitions and not (once and self._turn_tool_defs_emitted):
                attrs[events.TOOL_DEFINITIONS] = deepcopy(tool_definitions)
                self._turn_tool_defs_emitted = True
        self.spans.append(
            Span(
                span_id=self._next_span_id(),
                parent_id=self._turn_span_id,
                name=f"chat {model}" if model else "chat",
                kind=events.CLIENT,
                operation=events.CHAT,
                attributes=attrs,
                status="ok",
                duration_s=seconds,
            )
        )

    @staticmethod
    def _content_messages(messages: list[dict]) -> list[dict]:
        """Role and content, plus the fields that pair a tool call to its result.

        Dropping ``tool_calls``/``tool_call_id`` would flatten every tool-using turn
        into an assistant message with empty content, so the captured request would
        not replay. Copied rather than aliased: the live history stays out of reach
        of whatever the exporter does with the span.
        """
        rows = []
        for m in messages:
            row = {"role": m.get("role", ""), "content": m.get("content", "")}
            for key in _TOOL_LINK_KEYS:
                if key in m:
                    row[key] = deepcopy(m[key])
            rows.append(row)
        return rows

    def record_tool(
        self,
        name: str,
        seconds: float,
        args: str = "",
        result: str = "",
        status: str = "ok",
        *,
        call_id: str | None = None,
        description: str | None = None,
        max_chars: int = MAX_ITEM_CHARS,
    ) -> None:
        """Record one tool call. ``max_chars`` is how much of the captured content this
        keeps: the caller's own door budget for that result, so the span shows what the
        model was actually handed rather than a constant that may be smaller or larger.
        The default is the per-item ceiling, for callers with no door of their own."""
        # Keep the trace small but replayable — clamp the captured I/O.
        self._emit(
            Event(
                "tool",
                name,
                seconds,
                args=clamp(args, 120),
                result=clamp(result, 120),
                status=status,
                turn=self._turn,
            )
        )
        attrs: dict = {
            events.OPERATION_NAME: events.EXECUTE_TOOL,
            events.TOOL_NAME: name,
            events.TOOL_TYPE: "function",
        }
        if call_id is not None:
            attrs[events.TOOL_CALL_ID] = call_id
        if description is not None:
            attrs[events.TOOL_DESCRIPTION] = description
        if status == "error":
            attrs[events.ERROR_TYPE] = "error"
        if self.capture_content:
            # Both sides bounded, and bounded the same way: head AND tail, at the size
            # the caller's door allowed this result. Head-only lost the last line, which
            # on a failing build is the one line worth capturing; leaving the args
            # uncapped left a 33MB write_file in the span whole while its result was
            # trimmed to 4,000 chars. Spans are in-memory only — ``dump_events`` persists
            # the flat Event list and nothing else — so this bounds a live process's
            # exporter payload and its resident trace, not a file on disk.
            attrs[events.INPUT_MESSAGES] = recut(args, max_chars, _CAPTURE_TAIL_FRACTION)
            attrs[events.OUTPUT_MESSAGES] = recut(result, max_chars, _CAPTURE_TAIL_FRACTION)
        self.spans.append(
            Span(
                span_id=self._next_span_id(),
                parent_id=self._turn_span_id,
                name=f"execute_tool {name}",
                kind=events.INTERNAL,
                operation=events.EXECUTE_TOOL,
                attributes=attrs,
                status=status,
                duration_s=seconds,
            )
        )

    def record_verify(self, passed: bool, seconds: float, detail: str = "") -> None:
        """Record one self-verify attempt — makes the ch-12 verify loop visible.

        Also adds an INTERNAL span with a custom ``verify`` operation — this is a
        NON-STANDARD extension; OTel GenAI has no verify operation.
        """
        self._emit(
            Event(
                "verify",
                "verify",
                seconds,
                result=clamp(detail, 120),
                status="pass" if passed else "fail",
                turn=self._turn,
            )
        )
        self.spans.append(
            Span(
                span_id=self._next_span_id(),
                parent_id=self._turn_span_id,
                name="verify",
                kind=events.INTERNAL,
                operation="verify",  # custom, non-standard extension
                attributes={events.OPERATION_NAME: "verify"},
                status="ok" if passed else "error",
                duration_s=seconds,
            )
        )

    def record_plan(self, seconds: float, *, status: str = "ok") -> None:
        """Record a planning step as a ``plan`` span (ch-10 orchestrator).

        Emits both a flat ``Event`` (so a UI trace pane that renders events shows the
        plan step) and an OTel ``plan`` span. Like every other ``record_*`` method,
        the span's ``duration_s`` is the caller-measured ``seconds`` — never a fresh
        clock read — and it nests under the current turn.
        """
        self._emit(Event("plan", "plan", seconds, status=status, turn=self._turn))
        self.spans.append(
            Span(
                span_id=self._next_span_id(),
                parent_id=self._turn_span_id,
                name="plan",
                kind=events.INTERNAL,
                operation=events.PLAN,
                attributes={events.OPERATION_NAME: events.PLAN},
                status=status,
                duration_s=seconds,
            )
        )

    def get_spans(self) -> list[Span]:
        """Return the full OTel span list, with invoke_agent durations assembled.

        An ``invoke_agent`` span's ``duration_s`` is the sum of its children's
        durations — derived, never read from a fresh clock (offline-deterministic).
        """
        child_totals: dict[str, float] = {}
        for s in self.spans:
            if s.parent_id is not None:
                child_totals[s.parent_id] = child_totals.get(s.parent_id, 0.0) + s.duration_s
        for s in self.spans:
            if s.operation == events.INVOKE_AGENT:
                s.duration_s = child_totals.get(s.span_id, 0.0)
        return self.spans

    def export(self) -> None:
        """Hand the assembled spans to the exporter seam (default: no-op)."""
        self.exporter.export(self.get_spans())

    def dump_events(self) -> list[dict]:
        """Serialize events for persistence — so a trace survives a restart."""
        return [asdict(e) for e in self.events]

    def load_events(self, rows: list[dict]) -> None:
        """Restore persisted events and continue turn numbering from where they left off."""
        known = {f.name for f in fields(Event)}
        self.events = [Event(**{k: v for k, v in r.items() if k in known}) for r in rows]
        self._turn = max((e.turn for e in self.events), default=0)

    def totals(self) -> dict:
        return {
            "llm_calls": sum(e.kind == "llm" for e in self.events),
            "tool_calls": sum(e.kind == "tool" for e in self.events),
            "tokens": sum(e.tokens for e in self.events),
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cost": round(sum(e.cost for e in self.events), 6),
            "seconds": round(sum(e.seconds for e in self.events), 3),
        }

    def timeline(self) -> str:
        lines = []
        for i, e in enumerate(self.events):
            if e.kind == "llm":
                extra = f"{e.tokens} tok ${e.cost:.4f}"
            elif e.kind == "verify":
                extra = e.status
            else:
                extra = f"{e.args} -> {e.result}".strip()
            lines.append(f"{i:>2} {e.kind:<6} {e.label:<16} {e.seconds * 1000:6.0f} ms {extra}")
        t = self.totals()
        lines.append("   " + "-" * 42)
        lines.append(
            f"   {t['llm_calls']} llm · {t['tool_calls']} tool · "
            f"{t['tokens']} tok · ${t['cost']:.4f} · {t['seconds']} s"
        )
        return "\n".join(lines)
