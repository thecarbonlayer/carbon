"""Context management — compaction (ch-06).

When the conversation outgrows a budget, summarize the middle into one note and
keep the head and tail intact (models read the start and end most reliably —
"present is not the same as used"). A good summary preserves what the *next*
turn needs, not merely fewer words.

Three strategies, one shipped default (``structured_checkpoint``) and one bounded,
Refinery-selectable addition (``token_budget_checkpoint``, ch-v4). Each is a
``_Strategy`` record in ``_STRATEGIES`` rather than a branch inline in ``compact()``
— adding a fourth strategy means adding a registry entry, not re-reading the whole
function to find every place ``token_budgeted`` was checked.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from harness import checkpoint
from harness.harness_config import CONFIG
from model import LLMResponse, Provider, chat

# re-export; the summarizer's instructions live in the editable surface
COMPACTION_PROMPT = CONFIG.compaction_prompt

# The caller's retry wrapper around the summarizer's one model call: it takes the
# zero-argument call and runs it under the caller's retry policy. The Agent passes
# its own (``Agent._summarizer_retry``) from both of its ``compact()`` call sites,
# so the summarizer is retried exactly like every other model call in a turn —
# it used to call ``chat()`` bare, and a transient serving fault here crashed the
# turn that every other call site would have absorbed. ``None`` means call
# directly, which keeps direct callers (tests) on the old single-try behavior.
SummarizerRetry = Callable[[Callable[[], LLMResponse]], LLMResponse]

# Every checkpoint note starts with this, which is how a later compaction finds the
# previous one to update instead of re-summarizing raw history it has already seen.
_NOTE_PREFIX = "[summary of earlier conversation"

_CHECKPOINT_HEADINGS = (
    "Return a cumulative structured checkpoint with these headings: Goal, "
    "Constraints, Decisions, Completed work, Files read or changed, Exact commands "
    "and tool calls, Failures and rejected approaches, Current state, Next steps. "
)
_PRESERVE_VERBATIM = "Preserve identifiers, paths, arguments, receipts, and error tails verbatim."


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


# --- cut-point mechanisms, one per strategy family ---------------------------------


def _count_cut(messages: list[dict], _head_end: int, keep_tail: int, _recent_reserve: int) -> int:
    """The two shipped strategies: keep the last ``keep_tail`` messages, uncut."""
    return len(messages) - keep_tail


def _token_budget_cut(
    messages: list[dict], head_end: int, keep_tail: int, recent_reserve: int
) -> int:
    """Walk back from the newest message until ``recent_reserve`` tokens are kept,
    then snap to the turn boundary that cut lands in.

    The message-count cut this replaces is a proxy that fails in both directions: four
    tail messages are nearly nothing after a turn of one-line acknowledgements, and far
    over budget after a turn that read a file. Budgeting in tokens makes what is kept
    the same size regardless of the shape of the turn that produced it.

    A non-positive reserve means "no budget configured" — fall back to the plain
    message-count cut, the same one the other two strategies use, rather than every
    caller having to know to skip this function when the reserve is off.

    Never returns below ``head_end``, so a single message larger than the whole
    reserve cannot swallow the head — in that case the caller finds an empty middle
    and leaves the history alone. Snapping only ever moves the cut EARLIER, so the
    reserve is a floor, not a target.
    """
    if recent_reserve <= 0:
        return _count_cut(messages, head_end, keep_tail, recent_reserve)
    total = 0
    i = len(messages)
    while i > head_end:
        total += _message_tokens(messages[i - 1])
        if total > recent_reserve:
            break
        i -= 1
    # Clamped below the end: the newest message must always be kept even if it alone
    # exceeds the reserve — dropping what the model just produced is never right, and
    # an unclamped index would run off the end of the list in the turn snap below.
    i = max(min(i, len(messages) - 1), head_end)
    j = i
    while j > 0 and messages[j].get("role") != "user":
        j -= 1
    return j if j > 0 and messages[j].get("role") == "user" else i


# --- prompt assembly, one per strategy family ---------------------------------------


def _plain_prompt(base: str) -> str:
    return base


def _structured_prompt(base: str) -> str:
    return (
        base
        + "\n\n"
        + _CHECKPOINT_HEADINGS
        + "Merge any earlier [summary of earlier conversation] into this checkpoint. "
        + _PRESERVE_VERBATIM
    )


def _incremental_prompt(base: str) -> str:
    return (
        base
        + "\n\n"
        + _CHECKPOINT_HEADINGS
        + "You are UPDATING an existing checkpoint, not writing a new one: carry "
        "every still-true fact from it forward unchanged, revise what the new "
        "messages have changed, and add what they introduced. Never drop a fact "
        "merely because it is old. " + _PRESERVE_VERBATIM
    )


# --- post-processing, one per strategy family ---------------------------------------


def _identity_finalize(summary: str, **_kwargs: object) -> tuple[str, bool]:
    return summary, False


def _stateful_finalize(
    summary: str,
    *,
    prior_ops: checkpoint.FileOps,
    middle: list[dict],
    policy: object,
    summary_max_tokens: int,
) -> tuple[str, bool]:
    """Bound an oversized checkpoint, then attach the deterministic state block.

    The checkpoint is state, not prose, so it gets a hard ceiling of its own: an
    oversized one crowds out the very window compaction just freed, and it grows every
    pass if nothing bounds it. Truncation is a bounded, already-vetted door — reusing
    ``limits.truncate`` rather than inventing a second clamp.

    The state block is attached AFTER bounding, so the state that must not be
    paraphrased also cannot be truncated away by the fallback.

    Returns the finalized text alongside whether the truncation fallback fired —
    the caller folds that into ``CompactionFacts.truncated`` (Phase 1 §3).
    """
    from harness.harness_config import TruncationPolicy
    from harness.limits import truncate

    budget = summary_max_tokens * 4  # chars, matching the ~4-chars-per-token estimator
    truncated = len(summary) > budget
    if truncated:
        fallback = getattr(policy, "checkpoint_fallback", "head_tail")
        summary = truncate(summary, TruncationPolicy(fallback, budget, 0.5))
    state = checkpoint.render(checkpoint.merge(prior_ops, checkpoint.file_ops(middle)))
    return (f"{summary}\n\n{state}" if state else summary), truncated


# --- the registry --------------------------------------------------------------------


@dataclass(frozen=True)
class _Strategy:
    """One compaction strategy's four behaviors. ``compact()`` dispatches through
    this record and never branches on a strategy name directly — adding a strategy
    means adding an entry here, not re-reading ``compact()`` for every place an
    old strategy name was checked."""

    cut: Callable[[list[dict], int, int, int], int]
    prompt: Callable[[str], str]
    finalize: Callable[..., tuple[str, bool]]
    # Whether the previous checkpoint is pulled out of the transcript and passed to
    # the summarizer as its own message, carried forward verbatim when there is
    # nothing new to fold in, and bounded/re-attached after summarizing. False for
    # the two strategies that predate incremental checkpointing — they fold the
    # prior note into the transcript like any other message, exactly as before this
    # strategy existed, so their behavior is untouched by its addition.
    incremental: bool
    # Below this reserve there is no budget to enforce, so the strategy falls back to
    # a message count — only meaningful for the token-budgeted strategy.
    floor_needs_reserve: bool = False


_STRATEGIES: dict[str, _Strategy] = {
    "summarize_middle": _Strategy(_count_cut, _plain_prompt, _identity_finalize, incremental=False),
    "structured_checkpoint": _Strategy(
        _count_cut, _structured_prompt, _identity_finalize, incremental=False
    ),
    "token_budget_checkpoint": _Strategy(
        _token_budget_cut,
        _incremental_prompt,
        _stateful_finalize,
        incremental=True,
        floor_needs_reserve=True,
    ),
}


@dataclass(frozen=True)
class CompactionFacts:
    """What actually happened during one compaction pass (Phase 1 §3).

    The telemetry half of ``compact()``'s return value: pre/post token counts and
    elapsed seconds live with the caller (it already has ``window`` and a clock), so
    this carries only what ``compact()`` itself knows — the strategy that ran, the
    shape of what was cut, and whether the checkpoint had to be truncated. Consumed
    by ``Tracer.record_compaction`` to build an Event and a span.
    """

    strategy: str
    middle_count: int
    summary_chars: int
    # Amendment (2026-08-20): supersedes the Codex finding 3 note this replaced,
    # which documented these two fields as counting a tool call whether or not it
    # actually succeeded. That was a decided-wrong behavior, not a permanent
    # contract: both fields now count only SUCCESSFUL operations —
    # ``checkpoint.file_ops`` checks each tool call's paired result and drops a
    # call whose result errored (or has no paired result at all) before either
    # count sees its path. A denied or failed write no longer inflates
    # ``files_modified``, and a failed read no longer inflates ``files_read``.
    files_read: int
    files_modified: int
    truncated: bool
    # Amendment (2026-08-19, Codex finding 1): the summarizer's own model-call
    # usage, when a summarizer call actually ran this pass — None for every path
    # that never calls the model (the no-op early returns, and an incremental
    # strategy's "nothing new to fold in" carry-forward). The caller feeds this
    # to ``Tracer.record_llm`` exactly once so the summarizer's cost is measured
    # like any other model call, never fabricated as a phantom event when no
    # call happened.
    summarizer_usage: dict | None = None
    summarizer_seconds: float = 0.0


def _no_compaction(strategy: str) -> CompactionFacts:
    """Facts for a call that found nothing safe to compact.

    Both early returns in ``compact()`` hand the messages list back unchanged — the
    caller detects that with an identity check exactly as before this pair existed —
    so every count here is honestly zero rather than left over from a prior pass.
    """
    return CompactionFacts(
        strategy=strategy,
        middle_count=0,
        summary_chars=0,
        files_read=0,
        files_modified=0,
        truncated=False,
    )


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
    with_retry: SummarizerRetry | None = None,
) -> tuple[list[dict], CompactionFacts]:
    """Summarize the middle of ``messages`` into a single note; keep head + tail.

    Head/tail boundaries are snapped to whole-turn cuts so compaction never
    orphans a tool result from its assistant ``tool_calls`` (which the API rejects
    with a 400). If snapping leaves nothing safe to summarize, the history is
    returned unchanged — better a large window this turn than a corrupt one.

    Returns the (possibly unchanged) message list paired with a ``CompactionFacts``
    describing what happened. A caller still checks ``result is messages`` for the
    no-op case exactly as before this pair was added; the facts on that path are
    all zero rather than meaningful.
    """
    policy = CONFIG.compaction
    keep_head = policy.keep_head if keep_head is None else keep_head
    keep_tail = policy.keep_tail if keep_tail is None else keep_tail
    strategy = strategy or policy.strategy
    summary_max_tokens = summary_max_tokens or policy.summary_max_tokens
    strat = _STRATEGIES.get(strategy)
    if strat is None:
        raise ValueError(f"unsupported compaction strategy: {strategy}")
    recent_reserve = (
        policy.recent_token_reserve if recent_token_reserve is None else recent_token_reserve
    )
    budgeted = strat.floor_needs_reserve and recent_reserve > 0

    # The floor differs by strategy because the tail is chosen differently. Under a
    # token budget `keep_tail` is not the mechanism, so gating on it would refuse to
    # compact a short-but-enormous history — exactly the case the budget exists for.
    floor = keep_head + 1 if budgeted else keep_head + keep_tail
    if len(messages) <= floor:
        return messages, _no_compaction(strategy)

    head_end = keep_head
    while not _clean_cut(messages, head_end):
        head_end -= 1

    tail_start = strat.cut(messages, head_end, keep_tail, recent_reserve)
    while not _clean_cut(messages, tail_start):
        tail_start -= 1

    if head_end >= tail_start:  # snapping erased the middle — nothing safe to compact
        return messages, _no_compaction(strategy)

    head = messages[:head_end]
    tail = messages[tail_start:]
    middle = messages[head_end:tail_start]

    prior_text, prior_ops = (
        _previous_checkpoint(messages[:tail_start])
        if strat.incremental
        else ("", checkpoint.FileOps())
    )
    # The previous checkpoint is passed to the summarizer as settled state, so it must
    # NOT also appear in the transcript: a model shown the same content twice, once as
    # "carry this forward" and once as "compress this", compresses it. That double
    # presentation is precisely how a fact erodes across repeated compactions.
    summarizable = (
        [m for m in middle if str(m.get("content") or "") != prior_text]
        if strat.incremental and prior_text
        else middle
    )
    transcript = "\n".join(_serialize_message(m) for m in summarizable)
    prompt = strat.prompt(COMPACTION_PROMPT)

    if not transcript.strip():
        # Nothing new to fold in — the middle held only the previous checkpoint, which
        # happens on a repeat compaction that advanced by less than one turn. Re-asking
        # the summarizer here is worse than useless: the provider rejects an empty
        # transcript outright (a real 400 from the local endpoint), and even when it
        # does not, re-summarizing a checkpoint against nothing is the erosion this
        # strategy exists to prevent. Carry the prior text forward verbatim instead.
        summary = _strip_note_prefix(prior_text)
        summarizer_usage: dict | None = None
        summarizer_seconds = 0.0
    else:
        # Codex finding 1: time and keep the summarizer's own usage so the caller
        # can feed it to Tracer.record_llm — this model call must be measured like
        # any other, not discarded, so a config that compacts more pays its cost.
        summarize_start = time.perf_counter()
        summarize_resp = _summarize(
            prompt,
            transcript,
            prior_text if strat.incremental else "",
            model=model,
            provider=provider,
            max_tokens=summary_max_tokens,
            with_retry=with_retry,
        )
        summarizer_seconds = time.perf_counter() - summarize_start
        summary = summarize_resp.content
        summarizer_usage = summarize_resp.usage

    summary, truncated = strat.finalize(
        summary,
        prior_ops=prior_ops,
        middle=middle,
        policy=policy,
        summary_max_tokens=summary_max_tokens,
    )

    note = {
        "role": "system",
        "content": f"{_NOTE_PREFIX}; strategy={strategy}]\n{summary}",
    }
    ops = checkpoint.file_ops(middle)
    facts = CompactionFacts(
        strategy=strategy,
        middle_count=len(middle),
        summary_chars=len(summary),
        files_read=len(ops.read),
        files_modified=len(ops.modified),
        truncated=truncated,
        summarizer_usage=summarizer_usage,
        summarizer_seconds=summarizer_seconds,
    )
    return head + [note] + tail, facts


def _summarize(
    prompt: str,
    transcript: str,
    prior: str,
    *,
    model: str | None,
    provider: Provider | None,
    max_tokens: int,
    with_retry: SummarizerRetry | None = None,
) -> LLMResponse:
    """One summarizer call, with the previous checkpoint passed as its own message.

    Kept separate from the transcript rather than concatenated into it: the previous
    checkpoint is settled state to carry forward, the transcript is new material to
    fold in, and a model given both in one blob treats the older half as more history
    to compress — which is how facts erode across repeated compactions.

    Returns the full ``LLMResponse`` rather than just ``.content`` — the caller needs
    ``.usage`` too, to feed ``Tracer.record_llm`` (Codex finding 1) so this call is
    measured like any other model call instead of discarded.

    ``with_retry`` wraps the call itself, not ``chat`` — the ``chat`` name stays
    resolved here at call time, so the module seam tests patch keeps working.
    """
    conversation = [{"role": "system", "content": prompt}]
    if prior:
        conversation.append({"role": "user", "content": f"Existing checkpoint to update:\n{prior}"})
    conversation.append({"role": "user", "content": f"New messages to fold in:\n{transcript}"})

    def call() -> LLMResponse:
        return chat(
            conversation,
            model=model,
            provider=provider,  # summarize through the same endpoint the turn uses
            max_tokens=max_tokens,
        )

    return call() if with_retry is None else with_retry(call)


def _serialize_message(message: dict) -> str:
    """Loss-aware JSONL for the summarizer, including tool names and arguments."""
    kept = {
        key: message[key]
        for key in ("role", "content", "tool_calls", "tool_call_id")
        if key in message
    }
    return json.dumps(kept, ensure_ascii=False, sort_keys=True)
