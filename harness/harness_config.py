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
class TruncationPolicy:
    """A bounded, Carbon-implemented strategy for one information door."""

    strategy: str
    budget: int
    tail_fraction: float


@dataclass(frozen=True)
class CompactionPolicy:
    """When and how conversation history is compacted.

    The last three fields are the ``token_budget_checkpoint`` surface and default to
    values that reproduce the previous behavior exactly, so adding them is additive and
    default-neutral: an existing config file omits them, gets these defaults, and
    computes the same ``config_version``. That matters because an external improver
    pins its recorded baselines to that version — a default that shifted behavior would
    silently invalidate every one of them.
    """

    strategy: str
    keep_head: int
    keep_tail: int
    trigger_fraction: float
    summary_max_tokens: int
    # 0 disables token budgeting and falls back to the ``keep_tail`` message count.
    recent_token_reserve: int = 0
    # 0 disables the reserve; the trigger is then ``trigger_fraction`` alone.
    completion_reserve: int = 0
    # Bounded door for a checkpoint that outgrows ``summary_max_tokens``.
    checkpoint_fallback: str = "head_tail"


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded recovery for transient provider failures."""

    strategy: str
    max_attempts: int
    base_delay_ms: int


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
    file_injection: TruncationPolicy  # policy for @path context blocks
    tool_output: TruncationPolicy  # policy for tool results entering history
    compaction: CompactionPolicy  # strategy, shape, trigger, and summary budget
    retry: RetryPolicy  # transient provider retry strategy and hard bound
    compaction_prompt: str  # the summarizer's instructions (compaction.py)
    memory_search_limit: int  # max hits returned by cross-session recall
    attach_pattern: str  # regex (as a string) for @path references; compiled in context.py
    temperature: float  # default sampling temperature
    max_tokens: int  # default completion budget


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
    "file_injection": dict,
    "tool_output": dict,
    "compaction": dict,
    "retry": dict,
    "compaction_prompt": str,
    "memory_search_limit": int,
    "attach_pattern": str,
    "temperature": float,
    "max_tokens": int,
}
_SET_FIELDS = {"approval_tools", "code_extensions"}
# integer knobs that are counts/budgets — zero or negative would wedge the loop
_POSITIVE_INT_FIELDS = {
    "max_tool_steps",
    "default_context_limit",
    "verify_attempts",
    "max_item_chars",
    "memory_search_limit",
    "max_tokens",
}

_TRUNCATION_STRATEGIES = frozenset({"keep_head", "head_tail"})
_COMPACTION_STRATEGIES = frozenset(
    {"summarize_middle", "structured_checkpoint", "token_budget_checkpoint"}
)
# Additive `token_budget_checkpoint` knobs. Optional so an existing config file stays
# valid and its ``config_version`` — which external baselines pin to — does not move.
_COMPACTION_OPTIONAL = frozenset(
    {"recent_token_reserve", "completion_reserve", "checkpoint_fallback"}
)
_RETRY_STRATEGIES = frozenset({"fail_fast", "backoff"})

# These are correctness and trust properties, not optimization choices. They
# are published beside the editable schema so an external improver can avoid
# wasting proposals on behavior it is intentionally forbidden to weaken.
_IMMUTABLE_INVARIANTS = (
    {
        "name": "tool_argument_validation",
        "reason": "Malformed or incomplete tool calls must never execute.",
    },
    {
        "name": "unique_atomic_edits",
        "reason": "An edit must match exactly once and report the resulting diff.",
    },
    {
        "name": "workspace_and_secret_boundaries",
        "reason": "No strategy may expand file access or weaken secret-file refusal.",
    },
    {
        "name": "subagent_workspace_identity",
        "reason": (
            "Delegated workers use the workspace explicitly bound by the parent, "
            "read-only: a worker runs under its own Policy, so a mutating tool "
            "would execute outside the parent's approval gate."
        ),
    },
    {
        "name": "verification_integrity",
        "reason": "Fresh real test receipts and fail-closed verification are not editable.",
    },
    {
        "name": "strategy_registry_and_config_validation",
        "reason": (
            "Refinery may select vetted strategies; it may not install executable code hooks."
        ),
    },
)
_NON_EDITABLE_FIELDS = frozenset(
    {
        "version",
        "approval_tools",
        "code_extensions",
        "require_run",
        "attach_pattern",
        "max_item_chars",
        "memory_search_limit",
    }
)


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
    elif expected is float:
        # JSON has one number type: an improver proposing `1` means 1.0, and
        # bouncing it on a type technicality wastes a legitimate proposal.
        ok = isinstance(value, int | float) and not isinstance(value, bool)
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
    if key == "temperature" and not 0 <= float(value) <= 2:  # type: ignore[arg-type]
        raise ValueError(
            f"harness config: field {key!r} must be a number from 0 to 2, got {_short(value)}"
        )


def _object_keys(
    name: str, value: dict, expected: set[str], optional: frozenset[str] = frozenset()
) -> None:
    """Exact key check, with an allowance for additive fields.

    ``optional`` exists so a new bounded strategy can ship its knobs without forcing
    every existing config file — and every baseline pinned to its version — through a
    rewrite. Optional keys are still type-validated when present; they are simply not
    required to be there.
    """
    unknown = sorted(set(value) - expected - optional)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            f"harness config: field {name!r} has "
            f"unknown keys {unknown or 'none'} and missing keys {missing or 'none'}"
        )


def _truncation_policy(name: str, value: dict) -> TruncationPolicy:
    _object_keys(name, value, {"strategy", "budget", "tail_fraction"})
    strategy = value["strategy"]
    budget = value["budget"]
    tail_fraction = value["tail_fraction"]
    if strategy not in _TRUNCATION_STRATEGIES:
        raise ValueError(
            f"harness config: field {name!r} strategy must be one of "
            f"{sorted(_TRUNCATION_STRATEGIES)}, got {_short(strategy)}"
        )
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"harness config: field {name!r}.budget must be a positive integer")
    if (
        not isinstance(tail_fraction, int | float)
        or isinstance(tail_fraction, bool)
        or not 0 < float(tail_fraction) < 1
    ):
        raise ValueError(f"harness config: field {name!r}.tail_fraction must be between 0 and 1")
    return TruncationPolicy(strategy, budget, float(tail_fraction))


def _compaction_policy(value: dict) -> CompactionPolicy:
    name = "compaction"
    _object_keys(
        name,
        value,
        {"strategy", "keep_head", "keep_tail", "trigger_fraction", "summary_max_tokens"},
        _COMPACTION_OPTIONAL,
    )
    strategy = value["strategy"]
    if strategy not in _COMPACTION_STRATEGIES:
        raise ValueError(
            f"harness config: field {name!r} strategy must be one of "
            f"{sorted(_COMPACTION_STRATEGIES)}, got {_short(strategy)}"
        )
    for key in ("keep_head", "keep_tail", "summary_max_tokens"):
        item = value[key]
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"harness config: field {name!r}.{key} must be a positive integer")
    # Reserves may be 0 — that is how they are switched off — so they are validated as
    # non-negative, unlike the required fields above.
    for key in ("recent_token_reserve", "completion_reserve"):
        if key in value:
            item = value[key]
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise ValueError(
                    f"harness config: field {name!r}.{key} must be a non-negative integer"
                )
    fallback = value.get("checkpoint_fallback", "head_tail")
    if fallback not in _TRUNCATION_STRATEGIES:
        raise ValueError(
            f"harness config: field {name!r}.checkpoint_fallback must be one of "
            f"{sorted(_TRUNCATION_STRATEGIES)}, got {_short(fallback)}"
        )
    fraction = value["trigger_fraction"]
    if (
        not isinstance(fraction, int | float)
        or isinstance(fraction, bool)
        or not 0 < float(fraction) <= 1
    ):
        raise ValueError(
            f"harness config: field {name!r}.trigger_fraction must be greater than 0 and at most 1"
        )
    return CompactionPolicy(
        strategy,
        value["keep_head"],
        value["keep_tail"],
        float(fraction),
        value["summary_max_tokens"],
        value.get("recent_token_reserve", 0),
        value.get("completion_reserve", 0),
        fallback,
    )


def _retry_policy(value: dict) -> RetryPolicy:
    name = "retry"
    _object_keys(name, value, {"strategy", "max_attempts", "base_delay_ms"})
    strategy = value["strategy"]
    if strategy not in _RETRY_STRATEGIES:
        raise ValueError(
            f"harness config: field {name!r} strategy must be one of "
            f"{sorted(_RETRY_STRATEGIES)}, got {_short(strategy)}"
        )
    attempts = value["max_attempts"]
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 5:
        raise ValueError(
            f"harness config: field {name!r}.max_attempts must be an integer from 1 to 5"
        )
    delay = value["base_delay_ms"]
    if not isinstance(delay, int) or isinstance(delay, bool) or not 0 <= delay <= 10_000:
        raise ValueError(
            f"harness config: field {name!r}.base_delay_ms must be an integer from 0 to 10000"
        )
    return RetryPolicy(strategy, attempts, delay)


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
        item = {
            "name": name,
            "type": "list[str]" if typ is list else typ.__name__,
            "collection": name in _SET_FIELDS,
            "positive_int": name in _POSITIVE_INT_FIELDS,
            "editable": name not in _NON_EDITABLE_FIELDS,
        }
        if name == "max_item_chars":
            item["deprecated"] = (
                "Compatibility re-export only; use file_injection.budget or "
                "tool_output.budget through their policy objects."
            )
        elif name in {"approval_tools", "code_extensions", "require_run"}:
            item["locked_reason"] = "Part of permission or verification integrity."
        elif name == "attach_pattern":
            item["locked_reason"] = "Context-delivery parser contract, not an optimization knob."
        elif name == "memory_search_limit":
            item["locked_reason"] = "Locked until Refinery has memory-recall miners and guards."
        if name in {"file_injection", "tool_output"}:
            item["strategies"] = sorted(_TRUNCATION_STRATEGIES)
            item["parameters"] = {
                "budget": {"type": "int", "positive": True},
                "tail_fraction": {"type": "float", "exclusive_min": 0, "exclusive_max": 1},
            }
        elif name == "compaction":
            item["strategies"] = sorted(_COMPACTION_STRATEGIES)
            item["parameters"] = {
                "keep_head": {"type": "int", "positive": True},
                "keep_tail": {"type": "int", "positive": True},
                "trigger_fraction": {"type": "float", "exclusive_min": 0, "max": 1},
                "summary_max_tokens": {"type": "int", "positive": True},
            }
        elif name == "retry":
            item["strategies"] = sorted(_RETRY_STRATEGIES)
            item["parameters"] = {
                "max_attempts": {"type": "int", "min": 1, "max": 5},
                "base_delay_ms": {"type": "int", "min": 0, "max": 10_000},
            }
        out.append(item)
    return out


def surface_manifest() -> dict:
    """Machine-readable editable choices and permanent non-editable invariants."""
    fields = config_schema()
    return {
        "editable": [item for item in fields if item["editable"]],
        "locked_fields": [item for item in fields if not item["editable"]],
        "immutable": [dict(item) for item in _IMMUTABLE_INVARIANTS],
    }


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
        if key in _SET_FIELDS:
            kwargs[key] = frozenset(raw[key])
        elif key in {"file_injection", "tool_output"}:
            kwargs[key] = _truncation_policy(key, raw[key])
        elif key == "compaction":
            kwargs[key] = _compaction_policy(raw[key])
        elif key == "retry":
            kwargs[key] = _retry_policy(raw[key])
        elif expected is float:
            kwargs[key] = float(raw[key])  # a JSON `1` is a valid temperature
        else:
            kwargs[key] = raw[key]
    return HarnessConfig(**kwargs)


CONFIG = load_config()
