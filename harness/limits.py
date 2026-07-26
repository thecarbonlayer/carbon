"""Door control (ch-06).

Hard per-item size limits, applied before anything enters the prompt. A single
huge file or tool output can drown the window (distraction / confusion /
poisoning); clamping each item at the door is the cheapest defense.
"""

from __future__ import annotations

from harness.harness_config import CONFIG, TruncationPolicy

MAX_ITEM_CHARS = CONFIG.max_item_chars  # re-export; the value lives in the editable surface


def clamp(text: str, max_chars: int = MAX_ITEM_CHARS) -> str:
    """Truncate an item to ``max_chars``, with a marker noting what was dropped."""
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}\n…[truncated {dropped} chars]"


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
    content_budget = max_chars
    if policy.strategy == "keep_head":
        return text[:content_budget] + marker
    if policy.strategy == "head_tail":
        tail_chars = max(1, int(content_budget * policy.tail_fraction))
        head_chars = max(1, content_budget - tail_chars)
        return text[:head_chars] + marker + "\n" + text[-tail_chars:]
    raise ValueError(f"unsupported truncation strategy: {policy.strategy}")
