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

    def __str__(self) -> str:
        return self.text
