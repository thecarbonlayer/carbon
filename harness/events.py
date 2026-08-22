"""The OTel GenAI span/event contract (ch-13, observability) — dependency-free.

These ARE the OpenTelemetry GenAI semantic-convention names. The constants below
(`gen_ai.operation.name`, `gen_ai.usage.input_tokens`, the `chat`/`execute_tool`/
`invoke_agent` operation names, the CLIENT/INTERNAL span kinds) are copied verbatim
from the spec:

    https://github.com/open-telemetry/semantic-conventions-genai

We hand-roll the span model instead of pulling in `opentelemetry-sdk` for three
reasons, all of which double as teaching points:

  1. **You SEE the conventions.** The attribute names aren't buried inside an SDK;
     they're right here, so the chapter can point at each one and explain it.
  2. **Deterministic and offline.** `verify` must run with no network, no exporter
     daemon, no wall-clock dependence. A plain dataclass span tree gives us that.
  3. **Optional OTLP graduation via the seam.** `SpanExporter` mirrors the provider
     seam: the default `NullExporter` is a no-op, but dropping in an OTLP-backed
     exporter ("graduation") sends the very same spans to Jaeger/Honeycomb/etc.
     The core never hard-depends on the OTel SDK.

This module is additive: it sits alongside the flat `Event` model in
`observability.py`, it does not replace it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

# --- gen_ai.operation.name values (span operations) ---------------------------
CHAT = "chat"
EXECUTE_TOOL = "execute_tool"
INVOKE_AGENT = "invoke_agent"
PLAN = "plan"
CREATE_AGENT = "create_agent"
COMPACT = "compact"  # NOTE: ours — OTel GenAI has no compaction operation

# --- span kinds ---------------------------------------------------------------
CLIENT = "client"
INTERNAL = "internal"

# --- gen_ai.* attribute names (exact spec strings) ----------------------------
OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
# NOTE: gen_ai.usage.cost is OUR custom extension — OTel has no dollar attribute.
USAGE_COST = "gen_ai.usage.cost"
TOOL_NAME = "gen_ai.tool.name"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_DESCRIPTION = "gen_ai.tool.description"
TOOL_CALL_ID = "gen_ai.tool.call.id"
AGENT_NAME = "gen_ai.agent.name"
AGENT_ID = "gen_ai.agent.id"
CONVERSATION_ID = "gen_ai.conversation.id"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"
ERROR_TYPE = "error.type"

# --- opt-in content attributes (off by default, see capture_content) ----------
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"

# --- carbon.compaction.* — OUR custom extension (Phase 1 §3, no OTel equivalent) ---
COMPACTION_STRATEGY = "carbon.compaction.strategy"
COMPACTION_PRE_TOKENS = "carbon.compaction.pre_tokens"
COMPACTION_POST_TOKENS = "carbon.compaction.post_tokens"
COMPACTION_SUMMARY_CHARS = "carbon.compaction.summary_chars"
COMPACTION_MIDDLE_COUNT = "carbon.compaction.middle_count"
COMPACTION_FILES_READ = "carbon.compaction.files_read"
COMPACTION_FILES_MODIFIED = "carbon.compaction.files_modified"
COMPACTION_TRUNCATED = "carbon.compaction.truncated"


@dataclass
class Span:
    """One OTel-shaped span. Serializable via `asdict`.

    `name` is the spec span name, e.g. "chat google/gemma-4-26b-a4b" or
    "execute_tool calculator". `parent_id` is the span_id of the enclosing span
    (None for a root). `duration_s` is derived from the seconds already measured
    by the tracer, never from a fresh wall-clock read — so spans are deterministic.
    """

    span_id: str
    parent_id: str | None
    name: str
    kind: str
    operation: str
    attributes: dict = field(default_factory=dict)
    status: str = "ok"  # "ok" | "error" | ...
    duration_s: float = 0.0


@runtime_checkable
class SpanExporter(Protocol):
    """The export seam. A pure consumer of spans — the mirror of the provider seam."""

    def export(self, spans: list[Span]) -> None: ...


class NullExporter:
    """Default exporter: do nothing. Keeps `verify` offline and side-effect free."""

    def export(self, spans: list[Span]) -> None:  # noqa: D401 — no-op
        return None


class JsonlExporter:
    """Write one JSON span per line — the simplest durable, greppable sink."""

    def __init__(self, path: str) -> None:
        self.path = path

    def export(self, spans: list[Span]) -> None:
        with open(self.path, "w") as fh:
            for span in spans:
                fh.write(json.dumps(asdict(span)) + "\n")


class ConsoleExporter:
    """Print a one-line summary per span — handy for live demos."""

    def export(self, spans: list[Span]) -> None:
        for s in spans:
            print(
                f"[span] {s.operation:<13} {s.name:<32} "
                f"{s.duration_s * 1000:6.0f} ms {s.status} (parent={s.parent_id})"
            )
