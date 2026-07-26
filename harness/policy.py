"""Policy — the permission gate as a bound object (v0.1, the embedding seam).

Through ch-14 the gate was a global set of tool names plus a yes/no callback. That
is enough for a human at a REPL, but a consumer driving the agent wants to say, per
agent, exactly which tools may run and under what rule — an allow-list, a
deny-list, read-only, or an approval prompt — without reaching into the loop.

``Policy`` is that rule as one object. The interactive prompt becomes one backend
(``approve``); always-allow and always-deny are others. carbon supplies the
enforcement; the values and any predicate are the consumer's (dev-notes/adr/0002).

Back-compat: ``Agent`` still accepts ``approve=`` and ``approval_required=`` and
builds a ``Policy`` from them when none is passed, so every ch-05 caller is
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# carbon's own mutating built-ins. ``read_only`` denies these outright rather than
# gating them. A consumer whose tools all read (e.g. a query agent) sees no effect.
DEFAULT_MUTATORS = frozenset({"write_file", "edit_file", "bash"})


@dataclass(frozen=True)
class Policy:
    """A per-agent permission rule the loop consults before running any tool."""

    require_approval: frozenset[str] = frozenset()  # gated tools (prompt via ``approve``)
    allow: frozenset[str] | None = None  # if set, only these tools may run
    deny: frozenset[str] = frozenset()  # these tools never run
    read_only: bool = False  # deny the mutating built-ins outright
    approve: Callable[[str, str], bool] | None = None  # approval backend
    mutators: frozenset[str] = DEFAULT_MUTATORS

    def decision(self, name: str, args: str) -> tuple[bool, str]:
        """Return ``(allowed, marker)``. ``marker`` is the tool-result string to
        record when a call is refused, so a denial reads clearly in the transcript
        (and empty when the call is allowed). Fail closed: a gated tool with no
        approver is denied."""
        if self.allow is not None and name not in self.allow:
            return False, "[denied: not permitted by policy]"
        if name in self.deny:
            return False, "[denied: not permitted by policy]"
        if self.read_only and name in self.mutators:
            return False, "[denied: read-only policy]"
        if name in self.require_approval:
            ok = self.approve(name, args) if self.approve else False
            return ok, "" if ok else "[denied by approval gate]"
        return True, ""
