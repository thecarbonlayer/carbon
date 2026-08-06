"""Door control (ch-06).

Hard per-item size limits, applied before anything enters the prompt. A single
huge file or tool output can drown the window (distraction / confusion /
poisoning); clamping each item at the door is the cheapest defense.

Two strategies, registry-shaped to match ``compaction.py``'s ``_STRATEGIES`` even
though each is a single expression today: every strategy-shaped config knob uses
the same shape, so an external improver (or a person) reads one pattern once,
not one pattern per knob plus a note that this one is "simple enough" to skip it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harness.harness_config import CONFIG, TruncationPolicy

MAX_ITEM_CHARS = CONFIG.max_item_chars  # re-export; the value lives in the editable surface


def clamp(text: str, max_chars: int = MAX_ITEM_CHARS) -> str:
    """Truncate an item to ``max_chars``, with a marker noting what was dropped."""
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}\n…[truncated {dropped} chars]"


def _keep_head(text: str, content_budget: int, _tail_fraction: float, marker: str) -> str:
    return text[:content_budget] + marker


def _head_tail(text: str, content_budget: int, tail_fraction: float, marker: str) -> str:
    tail_chars = max(1, int(content_budget * tail_fraction))
    head_chars = max(1, content_budget - tail_chars)
    return text[:head_chars] + marker + "\n" + text[-tail_chars:]


@dataclass(frozen=True)
class _TruncationStrategy:
    # The marker's POSITION differs per strategy — trailing for keep_head, inserted
    # between head and tail for head_tail — so each strategy places it, rather than
    # the caller gluing one fixed shape onto every strategy's output.
    apply: Callable[[str, int, float, str], str]


_TRUNCATION_STRATEGIES: dict[str, _TruncationStrategy] = {
    "keep_head": _TruncationStrategy(_keep_head),
    "head_tail": _TruncationStrategy(_head_tail),
}


def truncate(
    text: str,
    policy: TruncationPolicy,
    *,
    budget: int | None = None,
    continuation_hint: str | None = None,
) -> str:
    """Apply one vetted truncation strategy.

    ``budget`` lets a tool declare a smaller result limit without changing the
    selected strategy. The marker is deliberately actionable: losing bytes is
    unavoidable, losing the fact that bytes were lost is not.
    """
    max_chars = budget or policy.budget
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    hint = f" {continuation_hint}" if continuation_hint else ""
    marker = f"\n…[truncated {dropped} chars using {policy.strategy}.{hint}]"
    strat = _TRUNCATION_STRATEGIES.get(policy.strategy)
    if strat is None:
        raise ValueError(f"unsupported truncation strategy: {policy.strategy}")
    return strat.apply(text, max_chars, policy.tail_fraction, marker)
