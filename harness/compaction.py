"""Context management — compaction (ch-06).

When the conversation outgrows a budget, summarize the middle into one note and
keep the head and tail intact (models read the start and end most reliably —
"present is not the same as used"). A good summary preserves what the *next*
turn needs, not merely fewer words.
"""

from __future__ import annotations

import json

from harness.harness_config import CONFIG
from model import Provider, chat

# re-export; the summarizer's instructions live in the editable surface
COMPACTION_PROMPT = CONFIG.compaction_prompt


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap ~4-chars-per-token estimate over message contents and tool-call args.

    Tool-call arguments (e.g. a whole file body in a ``write_file`` call) are part
    of the window too — count them, or a tool-heavy turn under-measures and the
    compaction door fires late, exactly when the window is fullest.
    """
    total = 0
    for m in messages:
        total += len(str(m.get("content", "") or ""))
        for tc in m.get("tool_calls") or []:
            total += len(json.dumps(tc))
    return total // 4


def _is_tool_call_assistant(m: dict) -> bool:
    return m.get("role") == "assistant" and bool(m.get("tool_calls"))


def _clean_cut(messages: list[dict], i: int) -> bool:
    """True if splitting ``messages`` at index ``i`` keeps every tool-call group whole.

    An OpenAI-compatible payload requires each assistant ``tool_calls`` message to
    be immediately followed by its ``tool`` results. A cut is unsafe if it would
    put a ``tool`` result on the right without its assistant (``messages[i]`` is a
    tool), or leave an assistant with dangling ``tool_calls`` on the left
    (``messages[i-1]`` still expects results at ``i``).
    """
    if i <= 0 or i >= len(messages):
        return True
    if messages[i].get("role") == "tool":
        return False
    if _is_tool_call_assistant(messages[i - 1]):
        return False
    return True


def compact(
    messages: list[dict],
    *,
    keep_head: int | None = None,
    keep_tail: int | None = None,
    strategy: str | None = None,
    summary_max_tokens: int | None = None,
    model: str | None = None,
    provider: Provider | None = None,
) -> list[dict]:
    """Summarize the middle of ``messages`` into a single note; keep head + tail.

    Head/tail boundaries are snapped to whole-turn cuts so compaction never
    orphans a tool result from its assistant ``tool_calls`` (which the API rejects
    with a 400). If snapping leaves nothing safe to summarize, the history is
    returned unchanged — better a large window this turn than a corrupt one.
    """
    policy = CONFIG.compaction
    keep_head = policy.keep_head if keep_head is None else keep_head
    keep_tail = policy.keep_tail if keep_tail is None else keep_tail
    strategy = strategy or policy.strategy
    summary_max_tokens = summary_max_tokens or policy.summary_max_tokens
    if len(messages) <= keep_head + keep_tail:
        return messages

    head_end = keep_head
    while not _clean_cut(messages, head_end):
        head_end -= 1
    tail_start = len(messages) - keep_tail
    while not _clean_cut(messages, tail_start):
        tail_start -= 1

    if head_end >= tail_start:  # snapping erased the middle — nothing safe to compact
        return messages

    head = messages[:head_end]
    tail = messages[tail_start:]
    middle = messages[head_end:tail_start]

    transcript = "\n".join(_serialize_message(m) for m in middle)
    prompt = COMPACTION_PROMPT
    if strategy == "structured_checkpoint":
        prompt += (
            "\n\nReturn a cumulative structured checkpoint with these headings: Goal, "
            "Constraints, Decisions, Completed work, Files read or changed, Exact commands "
            "and tool calls, Failures and rejected approaches, Current state, Next steps. "
            "Merge any earlier [summary of earlier conversation] into this checkpoint. "
            "Preserve identifiers, paths, arguments, receipts, and error tails verbatim."
        )
    elif strategy != "summarize_middle":
        raise ValueError(f"unsupported compaction strategy: {strategy}")
    summary = chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        model=model,
        provider=provider,  # summarize through the same endpoint the turn uses
        max_tokens=summary_max_tokens,
    ).content

    note = {
        "role": "system",
        "content": f"[summary of earlier conversation; strategy={strategy}]\n{summary}",
    }
    return head + [note] + tail


def _serialize_message(message: dict) -> str:
    """Loss-aware JSONL for the summarizer, including tool names and arguments."""
    kept = {
        key: message[key]
        for key in ("role", "content", "tool_calls", "tool_call_id")
        if key in message
    }
    return json.dumps(kept, ensure_ascii=False, sort_keys=True)
