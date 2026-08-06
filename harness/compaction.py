"""Context management — compaction (ch-06).

When the conversation outgrows a budget, summarize the middle into one note and
keep the head and tail intact (models read the start and end most reliably —
"present is not the same as used"). A good summary preserves what the *next*
turn needs, not merely fewer words.
"""

from __future__ import annotations

import json

from harness import checkpoint
from harness.harness_config import CONFIG
from model import Provider, chat

# re-export; the summarizer's instructions live in the editable surface
COMPACTION_PROMPT = CONFIG.compaction_prompt

# Every checkpoint note starts with this, which is how a later compaction finds the
# previous one to update instead of re-summarizing raw history it has already seen.
_NOTE_PREFIX = "[summary of earlier conversation"

_CHECKPOINT_HEADINGS = (
    "Return a cumulative structured checkpoint with these headings: Goal, "
    "Constraints, Decisions, Completed work, Files read or changed, Exact commands "
    "and tool calls, Failures and rejected approaches, Current state, Next steps. "
)
_PRESERVE_VERBATIM = "Preserve identifiers, paths, arguments, receipts, and error tails verbatim."

# The strategies this module implements. Mirrored by the config validator's registry,
# which is what an external improver selects from; this one is what actually branches.
_STRATEGIES = frozenset({"summarize_middle", "structured_checkpoint", "token_budget_checkpoint"})


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


def _message_tokens(m: dict) -> int:
    """``estimate_tokens`` for a single message — same accounting, one element."""
    return estimate_tokens([m])


def _token_budget_cut(messages: list[dict], floor: int, recent_reserve: int) -> int:
    """Walk back from the newest message until ``recent_reserve`` tokens are kept.

    The message-count cut this replaces is a proxy that fails in both directions: four
    tail messages are nearly nothing after a turn of one-line acknowledgements, and far
    over budget after a turn that read a file. Budgeting in tokens makes what is kept
    the same size regardless of the shape of the turn that produced it.

    Never returns below ``floor`` (the head boundary), so a single message larger than
    the whole reserve cannot swallow the head — in that case the caller finds an empty
    middle and leaves the history alone.
    """
    total = 0
    i = len(messages)
    while i > floor:
        total += _message_tokens(messages[i - 1])
        if total > recent_reserve:
            break
        i -= 1
    # Clamped below the end: a newest message that alone exceeds the reserve must still
    # be kept — dropping the message the model just produced is never the right answer,
    # and an unclamped index would run off the end of the list in the turn snap.
    return max(min(i, len(messages) - 1), floor)


def _turn_start_at_or_before(messages: list[dict], i: int) -> int:
    """Snap ``i`` back to the start of the turn it lands in.

    A turn starts at a user message and runs through the assistant and tool traffic it
    caused. Cutting inside one hands the summarizer half a turn and the model the other
    half, so the two disagree about what happened. Snapping to the boundary keeps each
    turn whole on exactly one side of the cut.

    Returns ``i`` unchanged when no user message precedes it — the caller's clean-cut
    check still applies, and a split turn is handled by the budget, not by this.
    """
    j = i
    while j > 0 and messages[j].get("role") != "user":
        j -= 1
    return j if j > 0 and messages[j].get("role") == "user" else i


def _strip_note_prefix(text: str) -> str:
    """The body of a checkpoint note, without its ``[summary ...]`` header line.

    Used when a checkpoint is carried forward verbatim: the caller re-adds the header,
    so keeping the old one would nest a header inside the note on every pass.
    """
    if text.startswith(_NOTE_PREFIX):
        _, _, body = text.partition("\n")
        return body
    return text


def _previous_checkpoint(messages: list[dict]) -> tuple[str, checkpoint.FileOps]:
    """The most recent checkpoint note's text and its tracked file state.

    Scanned newest-first so repeated compaction reads the CURRENT checkpoint rather
    than the first one ever written. Returns empty values when there is none, which is
    the ordinary case on the first compaction.
    """
    for m in reversed(messages):
        content = str(m.get("content") or "")
        if m.get("role") == "system" and content.startswith(_NOTE_PREFIX):
            return content, checkpoint.parse(content)
    return "", checkpoint.FileOps()


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
    recent_token_reserve: int | None = None,
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
    if strategy not in _STRATEGIES:
        raise ValueError(f"unsupported compaction strategy: {strategy}")
    token_budgeted = strategy == "token_budget_checkpoint"
    recent_reserve = (
        policy.recent_token_reserve if recent_token_reserve is None else recent_token_reserve
    )
    # The floor differs by strategy because the tail is chosen differently. Under a
    # token budget `keep_tail` is not the mechanism, so gating on it would refuse to
    # compact a short-but-enormous history — exactly the case the budget exists for.
    floor = keep_head + 1 if token_budgeted and recent_reserve > 0 else keep_head + keep_tail
    if len(messages) <= floor:
        return messages

    head_end = keep_head
    while not _clean_cut(messages, head_end):
        head_end -= 1

    if token_budgeted and recent_reserve > 0:
        # Budget first, then snap to a turn boundary, then to a legal payload cut. The
        # order matters: turn-snapping only ever moves the cut EARLIER, which keeps
        # more than the reserve rather than less, so the reserve is a floor.
        tail_start = _token_budget_cut(messages, head_end, recent_reserve)
        tail_start = _turn_start_at_or_before(messages, tail_start)
    else:
        tail_start = len(messages) - keep_tail
    while not _clean_cut(messages, tail_start):
        tail_start -= 1

    if head_end >= tail_start:  # snapping erased the middle — nothing safe to compact
        return messages

    head = messages[:head_end]
    tail = messages[tail_start:]
    middle = messages[head_end:tail_start]

    prior_text, prior_ops = _previous_checkpoint(messages[:tail_start])
    # The previous checkpoint is passed to the summarizer as settled state, so it must
    # NOT also appear in the transcript: a model shown the same content twice, once as
    # "carry this forward" and once as "compress this", compresses it. That double
    # presentation is precisely how a fact erodes across repeated compactions.
    summarizable = (
        [m for m in middle if str(m.get("content") or "") != prior_text]
        if token_budgeted and prior_text
        else middle
    )
    transcript = "\n".join(_serialize_message(m) for m in summarizable)
    prompt = COMPACTION_PROMPT
    if strategy == "structured_checkpoint":
        prompt += (
            "\n\n"
            + _CHECKPOINT_HEADINGS
            + "Merge any earlier [summary of earlier conversation] into this checkpoint. "
            + _PRESERVE_VERBATIM
        )
    elif token_budgeted:
        prompt += (
            "\n\n"
            + _CHECKPOINT_HEADINGS
            + "You are UPDATING an existing checkpoint, not writing a new one: carry "
            "every still-true fact from it forward unchanged, revise what the new "
            "messages have changed, and add what they introduced. Never drop a fact "
            "merely because it is old. " + _PRESERVE_VERBATIM
        )

    if not transcript.strip():
        # Nothing new to fold in — the middle held only the previous checkpoint, which
        # happens on a repeat compaction that advanced by less than one turn. Re-asking
        # the summarizer here is worse than useless: the provider rejects an empty
        # transcript outright (a real 400 from the local endpoint), and even when it
        # does not, re-summarizing a checkpoint against nothing is the erosion this
        # strategy exists to prevent. Carry the prior text forward verbatim instead.
        summary = _strip_note_prefix(prior_text)
    else:
        summary = _summarize(
            prompt,
            transcript,
            prior_text if token_budgeted else "",
            model=model,
            provider=provider,
            max_tokens=summary_max_tokens,
        )

    if token_budgeted:
        # The checkpoint is state, not prose, so it gets a hard ceiling of its own: an
        # oversized one crowds out the very window compaction just freed, and it grows
        # every pass if nothing bounds it. Truncation is a bounded, already-vetted door.
        summary = _bound_checkpoint(summary, policy, summary_max_tokens)
        # Re-attached AFTER any bounding, so the state that must not be paraphrased
        # also cannot be truncated away by the fallback.
        state = checkpoint.render(checkpoint.merge(prior_ops, checkpoint.file_ops(middle)))
        if state:
            summary = f"{summary}\n\n{state}"

    note = {
        "role": "system",
        "content": f"{_NOTE_PREFIX}; strategy={strategy}]\n{summary}",
    }
    return head + [note] + tail


def _summarize(
    prompt: str,
    transcript: str,
    prior: str,
    *,
    model: str | None,
    provider: Provider | None,
    max_tokens: int,
) -> str:
    """One summarizer call, with the previous checkpoint passed as its own message.

    Kept separate from the transcript rather than concatenated into it: the previous
    checkpoint is settled state to carry forward, the transcript is new material to
    fold in, and a model given both in one blob treats the older half as more history
    to compress — which is how facts erode across repeated compactions.
    """
    conversation = [{"role": "system", "content": prompt}]
    if prior:
        conversation.append({"role": "user", "content": f"Existing checkpoint to update:\n{prior}"})
    conversation.append({"role": "user", "content": f"New messages to fold in:\n{transcript}"})
    return chat(
        conversation,
        model=model,
        provider=provider,  # summarize through the same endpoint the turn uses
        max_tokens=max_tokens,
    ).content


def _bound_checkpoint(summary: str, policy: object, max_tokens: int) -> str:
    """Clamp an oversized checkpoint through the shipped truncation door.

    Reuses ``limits.truncate`` rather than inventing a second clamp, so the fallback
    obeys the same vetted strategies as every other door and cannot become a private
    escape hatch. The budget is in characters, matching ``TruncationPolicy``; the
    token ceiling is converted with the same ~4-chars-per-token ratio the estimator
    uses, so the two agree about what "too big" means.
    """
    from harness.harness_config import TruncationPolicy
    from harness.limits import truncate

    budget = max_tokens * 4
    if len(summary) <= budget:
        return summary
    fallback = getattr(policy, "checkpoint_fallback", "head_tail")
    return truncate(summary, TruncationPolicy(fallback, budget, 0.5))


def _serialize_message(message: dict) -> str:
    """Loss-aware JSONL for the summarizer, including tool names and arguments."""
    kept = {
        key: message[key]
        for key in ("role", "content", "tool_calls", "tool_call_id")
        if key in message
    }
    return json.dumps(kept, ensure_ascii=False, sort_keys=True)
