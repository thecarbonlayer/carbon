"""The editable surface — every behavioral knob, in one versioned data file.

Until now the harness's behavior was smeared across the code as module-level
constants: the system prompt in ``agent.py``, the clamp size in ``limits.py``,
the compaction prompt in ``compaction.py``, ctor defaults buried in signatures.
Changing behavior meant editing *code*, and knowing which of five files to edit.

This module declares those knobs as a first-class primitive: one JSON file
(``harness_config.json``, next to this module) is the *entire* editable surface.
Want a different prompt, a bigger window budget, another gated tool? Edit the
data file — never the code. That boundary is the point: an editor (human or
model) that may touch only this file can retune the whole harness and nothing
else, and the diff of what changed is the diff of one small file.

Three properties make the surface safe to hand over:

- **Versioned.** The file carries an integer ``version``; bump it on every
  change. Rollback is reverting the file; comparing two behaviors is diffing
  two versions of it. Cheap, because the surface is pure data.
- **Frozen.** ``HarnessConfig`` is a frozen dataclass and every consumer binds
  its values at import, so the real lifecycle is: edit the file, restart the
  process, ``git revert`` the file to roll back — never in-place mutation, and
  no hot-reload.
- **Validated at the door.** ``load_config`` rejects unknown keys, missing
  fields, wrong types, non-positive counts, and a malformed ``@path`` regex —
  loudly, at import. A malformed surface must fail the run, not silently fall
  back to defaults the editor thought it replaced.

Sets travel as JSON arrays and land as ``frozenset``; the ``@path`` attach
pattern travels as a regex *string* and is compiled at its use site
(``context.py``). ``CONFIG`` is loaded once at import and is the single source
of truth — the old module-level names still exist, but only as re-exports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("harness_config.json")


@dataclass(frozen=True)
class HarnessConfig:
    """The harness's behavioral knobs, loaded from ``harness_config.json``."""

    version: int  # bump on every edit; rollback = revert the file
    system_prompt: str  # the agent's default system prompt
    max_tool_steps: int  # tool-call rounds per turn before the loop gives up
    default_context_limit: int  # ~token budget before compaction fires
    approval_tools: frozenset[str]  # tools the approval gate guards
    code_extensions: frozenset[str]  # a write/edit of one of these arms the test gate
    verify_attempts: int  # re-prompts before the gate marks a turn unverified
    require_run: bool  # enforce an observed passing test run after code changes
    max_item_chars: int  # per-item clamp applied at the door (limits.py)
    compaction_prompt: str  # the summarizer's instructions (compaction.py)
    memory_search_limit: int  # max hits returned by cross-session recall
    attach_pattern: str  # regex (as a string) for @path references; compiled in context.py


# name -> expected JSON type; list-typed fields are validated as arrays of
# strings and converted to frozenset below.
_SCHEMA: dict[str, type] = {
    "version": int,
    "system_prompt": str,
    "max_tool_steps": int,
    "default_context_limit": int,
    "approval_tools": list,
    "code_extensions": list,
    "verify_attempts": int,
    "require_run": bool,
    "max_item_chars": int,
    "compaction_prompt": str,
    "memory_search_limit": int,
    "attach_pattern": str,
}
_SET_FIELDS = {"approval_tools", "code_extensions"}
# integer knobs that are counts/budgets — zero or negative would wedge the loop
_POSITIVE_INT_FIELDS = {
    "max_tool_steps",
    "default_context_limit",
    "verify_attempts",
    "max_item_chars",
    "memory_search_limit",
}


def _short(value: object) -> str:
    """A repr truncated to ~80 chars, so a wrong-typed prompt doesn't dump its
    whole body into the exception."""
    r = repr(value)
    return r if len(r) <= 80 else f"{r[:80]}…"


def _check_field(key: str, value: object, expected: type) -> None:
    """Reject a malformed value loudly. ``bool`` is a subclass of ``int`` in
    Python, so integer knobs must explicitly refuse booleans (and vice versa —
    ``bool`` fields accept only real booleans, which isinstance already ensures).
    Count/budget knobs must be positive, and the attach pattern must be a regex
    that compiles with a capture group — well-formedness checks, not value pins."""
    if expected is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif expected is list:
        ok = isinstance(value, list) and all(isinstance(x, str) for x in value)
    else:
        ok = isinstance(value, expected)
    if not ok:
        raise ValueError(
            f"harness config: field {key!r} must be {expected.__name__}"
            f"{' of str' if expected is list else ''}, got {_short(value)}"
        )
    if key in _POSITIVE_INT_FIELDS and isinstance(value, int) and value <= 0:
        raise ValueError(
            f"harness config: field {key!r} must be a positive integer, got {_short(value)}"
        )
    if key == "attach_pattern" and isinstance(value, str):
        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(
                f"harness config: field {key!r} must be a valid regex, got {_short(value)} ({exc})"
            ) from exc
        if compiled.groups < 1:
            raise ValueError(
                f"harness config: field {key!r} must have at least one capture group "
                f"(the use site extracts the path via group(1)), got {_short(value)}"
            )


def config_schema() -> list[dict]:
    """Describe the editable surface: one entry per field with its type and bounds.

    Derived from the same tables ``load_config`` validates against, so it cannot
    drift from the door. An external editor reads this to know which knobs exist,
    their types, which are collections, and which must stay positive — instead of
    hardcoding that knowledge (the mechanism carbon owns; which knob to turn stays
    the editor's, see dev-notes/adr/0002).
    """
    out: list[dict] = []
    for name, typ in _SCHEMA.items():
        out.append(
            {
                "name": name,
                "type": "list[str]" if typ is list else typ.__name__,
                "collection": name in _SET_FIELDS,
                "positive_int": name in _POSITIVE_INT_FIELDS,
            }
        )
    return out


def load_config(path: str | Path = CONFIG_PATH) -> HarnessConfig:
    """Read and structurally validate the editable surface; fail loudly.

    Unknown keys, missing fields, wrong types, non-positive counts, and a
    malformed attach regex are all errors — the loader never silently defaults,
    because a silent default would mean the file on disk and the behavior in
    memory disagree, which is exactly the drift the surface exists to prevent."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"harness config: expected a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_SCHEMA))
    if unknown:
        raise ValueError(f"harness config: unknown keys {unknown}")
    missing = sorted(set(_SCHEMA) - set(raw))
    if missing:
        raise ValueError(f"harness config: missing fields {missing}")
    kwargs: dict[str, Any] = {}
    for key, expected in _SCHEMA.items():
        _check_field(key, raw[key], expected)
        kwargs[key] = frozenset(raw[key]) if key in _SET_FIELDS else raw[key]
    return HarnessConfig(**kwargs)


CONFIG = load_config()
