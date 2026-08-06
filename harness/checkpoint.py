"""Deterministic harness state carried across compaction (ch-06).

A summarizer is a lossy channel, and the things it drops first are the things the
next turn most needs: which files were touched, what was already tried and rejected,
what the pending action is. Asking a model to "preserve everything important" and
then measuring whether it did is a bet; extracting the mechanical part of that state
from the transcript and re-attaching it verbatim is not.

So this module owns the part of a checkpoint that must never be paraphrased. It reads
tool calls, never prose, and it is pure: no model call, no I/O. ``compaction.py`` owns
where to cut and what to ask the summarizer for; this owns what survives regardless of
what the summarizer says.

The state block is rendered into the checkpoint note and parsed back out of the
previous one, which is what makes tracking cumulative — a file touched before the
first compaction is still listed after the third.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Tool name -> whether calling it READ or MODIFIED the path in its ``path`` argument.
# Deliberately a closed table rather than a heuristic on the name: a tool added later
# is tracked when someone declares what it does, not silently mis-filed because it
# happened to contain "write".
_READ_TOOLS = frozenset({"read_file"})
_MODIFY_TOOLS = frozenset({"write_file", "edit_file"})

_READ_OPEN, _READ_CLOSE = "<read-files>", "</read-files>"
_MOD_OPEN, _MOD_CLOSE = "<modified-files>", "</modified-files>"

_BLOCK_RE = re.compile(
    rf"{_READ_OPEN}\n?(.*?){_READ_CLOSE}|{_MOD_OPEN}\n?(.*?){_MOD_CLOSE}",
    re.DOTALL,
)


@dataclass(frozen=True)
class FileOps:
    """Paths read and paths modified, each de-duplicated and in first-seen order.

    Order is first-seen rather than sorted so the earliest-touched file stays first
    across repeated compactions — a stable list is easier to diff between checkpoints,
    and a re-sort on every pass would make an unchanged state look changed.
    """

    read: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.read or self.modified)


def _dedup(paths: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(p for p in paths if p))


def merge(*ops: FileOps) -> FileOps:
    """Union of several ``FileOps``, preserving first-seen order across all of them.

    A path that was read and later modified appears in both lists. That is deliberate:
    "we read this" and "we changed this" are different facts for the next turn, and
    collapsing them would lose the one that matters for a diff.
    """
    return FileOps(
        read=_dedup([p for o in ops for p in o.read]),
        modified=_dedup([p for o in ops for p in o.modified]),
    )


def file_ops(messages: list[dict]) -> FileOps:
    """Extract read/modified paths from the tool calls in ``messages``.

    Reads the ASSISTANT's tool-call arguments, not the tool results: a result can be
    an error string, and a call that failed still tells us what the agent was working
    on. Malformed arguments are skipped rather than raising — a broken tool call is
    already the tool layer's problem, and losing the whole checkpoint over one is a
    worse outcome than an incomplete file list.
    """
    read: list[str] = []
    modified: list[str] = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name not in _READ_TOOLS and name not in _MODIFY_TOOLS:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            path = args.get("path") if isinstance(args, dict) else None
            if not isinstance(path, str) or not path:
                continue
            (read if name in _READ_TOOLS else modified).append(path)
    return FileOps(read=_dedup(read), modified=_dedup(modified))


def render(ops: FileOps) -> str:
    """The state block appended verbatim to a checkpoint note.

    Returns "" for empty state so a checkpoint with nothing tracked does not carry two
    empty tags — the parser treats absent and empty identically, but a reader should
    not have to know that.
    """
    if not ops:
        return ""
    parts = []
    if ops.read:
        parts.append(f"{_READ_OPEN}\n" + "\n".join(ops.read) + f"\n{_READ_CLOSE}")
    if ops.modified:
        parts.append(f"{_MOD_OPEN}\n" + "\n".join(ops.modified) + f"\n{_MOD_CLOSE}")
    return "\n".join(parts)


def parse(text: str) -> FileOps:
    """Recover a state block from a previous checkpoint note.

    The inverse of ``render``, and the reason tracking accumulates rather than
    restarting at each compaction. Tolerant by design: text the summarizer wrapped,
    re-ordered, or duplicated still parses, because the alternative is silently
    dropping the file list on the third compaction.
    """
    read: list[str] = []
    modified: list[str] = []
    for match in _BLOCK_RE.finditer(text or ""):
        body, target = (
            (match.group(1), read) if match.group(1) is not None else (match.group(2), modified)
        )
        target.extend(line.strip() for line in (body or "").splitlines() if line.strip())
    return FileOps(read=_dedup(read), modified=_dedup(modified))
