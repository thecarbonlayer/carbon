"""RunResult — the structured outcome of one agent turn (v0.1, the embedding seam).

For fourteen chapters ``Agent.send`` returned a bare string: the final text, with
everything else — the tool calls, the token totals, whether a code change was
verified — discarded into ``self.messages`` for a consumer to reconstruct. Real
consumers did exactly that reconstruction by hand. This makes it first-class.

``Agent.run`` returns a ``RunResult``; ``Agent.send`` stays returning the final
text (``RunResult.text``), so existing callers are untouched. carbon reports what
happened, never what it means: the ``attributes`` bag on each tool call is left
empty for a consumer to fill with its own taxonomy (see dev-notes/adr/0002).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """One tool invocation this turn: what the model asked, what it got back.

    ``attributes`` is a consumer-populated bag (a tier, a domain status, a cost);
    carbon leaves it empty. ``is_error`` is the generic signal that the tool
    returned an error string.
    """

    name: str
    args: str
    result: str
    is_error: bool = False
    attributes: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """The structured outcome of one turn.

    ``str(result)`` is the final text, so a caller that expected ``send``'s string
    keeps working. ``totals`` is the tracer's totals when a tracer is attached
    (cumulative if the tracer persists across turns); empty otherwise.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0  # model calls this turn, including verify re-prompts
    approvals: int = 0  # gated tool calls that were approved and ran
    stop_reason: str = "stop"  # "stop" | "tool_budget" | "incomplete_response" | "deadline"
    totals: dict = field(default_factory=dict)
    # v0.4 (Phase 1 telemetry slice 1, contract §1): the public alternative to
    # reaching into agent._observed_pass / agent._last_tokens.
    verified: bool | None = None  # run-verification verdict; None = not requested
    # {"total_tokens": int} always when a tracer is attached; "input_tokens"/
    # "output_tokens" join it only when every call this run reported a real
    # provider split (amendment 2026-08-19, audit finding 2) — never a fabricated
    # one built from the total-only fallback.
    usage: dict = field(default_factory=dict)
    compactions: int = 0  # agent.compaction_count at run end

    def __str__(self) -> str:
        return self.text
