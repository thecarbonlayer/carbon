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
import math
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

    def __post_init__(self) -> None:
        # ``load_config`` validates the file's copy of these at import; a policy built
        # in *code* (the ``Agent(tool_output=...)`` seam) never passes that door. A
        # budget of 0 or a tail_fraction of 1.0 would surface only as a bizarre excerpt
        # mid-turn, so the type validates itself wherever it is constructed.
        #
        # The types are checked, not just the ranges, because both wrong types survive
        # the range check and fail later, deep inside the door: a float budget reaches
        # a string slice as a TypeError mid-turn (in the very fallback that exists so
        # truncation can never be fatal), and ``budget=True`` is a perfectly positive
        # integer that silently cuts every result to one character.
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget <= 0:
            raise ValueError(f"truncation budget must be a positive integer, got {self.budget!r}")
        if (
            isinstance(self.tail_fraction, bool)
            or not isinstance(self.tail_fraction, int | float)
            or not math.isfinite(self.tail_fraction)
            or not 0 < self.tail_fraction < 1
        ):
            raise ValueError(
                f"truncation tail_fraction must be between 0 and 1, got {self.tail_fraction!r}"
            )


@dataclass(frozen=True)
class CompactionPolicy:
    """When and how conversation history is compacted.

    Every field with a default is additive and default-neutral: it defaults to a
    value that reproduces the previous behavior exactly, so an existing config file
    omits it, gets that default, and computes the same ``config_version``. That
    matters because an external improver pins its recorded baselines to that
    version — a default that shifted behavior would silently invalidate every one
    of them. The three reserve/fallback fields are the ``token_budget_checkpoint``
    surface; ``prompt_suffix`` applies to every strategy.
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
    # The strategy-specific tail appended to ``compaction_prompt`` (the headings and
    # update instruction, compaction.py). None = the strategy's built-in suffix,
    # byte-identical prompts; a string replaces that suffix; "" strips it, leaving
    # the base prompt alone.
    prompt_suffix: str | None = None


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
    temperature: float | None  # sampling temperature; None sends no field (provider default)
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

# Single-source int bounds: consulted by BOTH `_check_field` (top-level HarnessConfig
# fields) and `_retry_policy` (retry's nested fields), and published verbatim by
# `config_schema()` — a bound changed here changes what the loader enforces and what
# an external editor is told in the same edit, which is the whole point of collecting
# them in one table instead of the hand-written literals this replaces (retry's
# `max_attempts`/`base_delay_ms` used to be checked directly against 1/5/0/10_000 in
# `_retry_policy`, duplicating what `config_schema()` published about them). `None` as
# the high bound means no ceiling.
_INT_BOUNDS: dict[str, tuple[int, int | None]] = {
    "max_tool_steps": (1, 200),
    "default_context_limit": (256, 1_000_000),
    "verify_attempts": (1, 10),
    "max_tokens": (256, 200_000),
    "memory_search_limit": (1, 100),
    "max_item_chars": (1, None),
    "max_attempts": (1, 5),
    "base_delay_ms": (0, 10_000),
}
# Top-level HarnessConfig fields that must be positive ints — derived from the bounds
# table rather than hand-picked a second time, so a field can't be bounded in one place
# and "positive" in another. Retry's nested fields (max_attempts, base_delay_ms) are
# naturally excluded: they aren't top-level `_SCHEMA` keys.
_POSITIVE_INT_FIELDS = frozenset(k for k in _INT_BOUNDS if k in _SCHEMA)

_TRUNCATION_STRATEGIES = frozenset({"keep_head", "head_tail"})
# tool_output's menu additionally offers offload_to_file: the complete result goes
# to a workspace file and the inline excerpt carries the path, so an over-budget
# tool result is recoverable instead of gone. Only the tool-result door gets it —
# file_injection already reads from a file the model can re-open, and the
# compaction fallback runs where no workspace is in hand to write into.
_TOOL_OUTPUT_STRATEGIES = _TRUNCATION_STRATEGIES | {"offload_to_file"}
_COMPACTION_STRATEGIES = frozenset(
    {"summarize_middle", "structured_checkpoint", "token_budget_checkpoint"}
)
# Additive compaction knobs: the three `token_budget_checkpoint` fields, plus the
# strategy-agnostic prompt suffix. Optional so an existing config file stays
# valid and its ``config_version`` — which external baselines pin to — does not move.
_COMPACTION_OPTIONAL = frozenset(
    {"recent_token_reserve", "completion_reserve", "checkpoint_fallback", "prompt_suffix"}
)
# Bounds for the two reserve knobs above — not part of `_INT_BOUNDS` because that table
# is keyed by top-level `_SCHEMA` field names and retry's nested fields; these two live
# one level deeper, inside the `compaction` object, validated by `_compaction_policy`
# and published in `config_schema()["compaction"]["parameters"]` from this same pair.
_COMPACTION_RESERVE_BOUNDS: tuple[int, int] = (0, 100_000)
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


def _bound_message(label: str, value: object, low: int, high: int | None) -> str:
    """The out-of-range error for one int-bounded field, phrased from the bound
    itself rather than hand-written per field — so a field with the same shape of
    bound (``low >= 1``, or a hard ceiling) reads the same way everywhere.

    ``label`` is the caller's already-formatted field reference (``repr(key)`` for a
    top-level field, ``f"{name!r}.{key}"`` for one nested a level down, e.g. inside
    ``retry`` or ``compaction``).
    """
    if low >= 1 and high is None:
        bound = "a positive integer"
    elif low >= 1:
        bound = f"a positive integer from {low} to {high}"
    elif high is None:
        bound = f"an integer >= {low}"
    else:
        bound = f"an integer from {low} to {high}"
    return f"harness config: field {label} must be {bound}, got {_short(value)}"


def _check_field(key: str, value: object, expected: type) -> None:
    """Reject a malformed value loudly. ``bool`` is a subclass of ``int`` in
    Python, so integer knobs must explicitly refuse booleans (and vice versa —
    ``bool`` fields accept only real booleans, which isinstance already ensures).
    Count/budget knobs must be positive, and the attach pattern must be a regex
    that compiles with a capture group — well-formedness checks, not value pins."""
    if key == "temperature" and value is None:
        # null is "send no temperature at all", deferring to the provider's own
        # default — the only way to match a harness that never pins the knob,
        # since every number (including 0.0) is a pin.
        return
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
    if key in _INT_BOUNDS and isinstance(value, int):
        low, high = _INT_BOUNDS[key]
        if value < low or (high is not None and value > high):
            raise ValueError(_bound_message(repr(key), value, low, high))
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


def _truncation_menu(name: str) -> frozenset[str]:
    """The strategy menu for one truncation-policy field — tool_output's is wider."""
    return _TOOL_OUTPUT_STRATEGIES if name == "tool_output" else _TRUNCATION_STRATEGIES


def _truncation_policy(name: str, value: dict) -> TruncationPolicy:
    _object_keys(name, value, {"strategy", "budget", "tail_fraction"})
    strategy = value["strategy"]
    budget = value["budget"]
    tail_fraction = value["tail_fraction"]
    if strategy not in _truncation_menu(name):
        raise ValueError(
            f"harness config: field {name!r} strategy must be one of "
            f"{sorted(_truncation_menu(name))}, got {_short(strategy)}"
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
    # bounded rather than strictly positive, unlike the required fields above.
    low, high = _COMPACTION_RESERVE_BOUNDS
    for key in ("recent_token_reserve", "completion_reserve"):
        if key in value:
            item = value[key]
            if not isinstance(item, int) or isinstance(item, bool) or item < low or item > high:
                raise ValueError(_bound_message(f"{name!r}.{key}", item, low, high))
    fallback = value.get("checkpoint_fallback", "head_tail")
    if fallback not in _TRUNCATION_STRATEGIES:
        raise ValueError(
            f"harness config: field {name!r}.checkpoint_fallback must be one of "
            f"{sorted(_TRUNCATION_STRATEGIES)}, got {_short(fallback)}"
        )
    # None — absent or an explicit null, the temperature precedent — means "use the
    # strategy's built-in suffix"; only a real string may replace it.
    suffix = value.get("prompt_suffix")
    if suffix is not None and not isinstance(suffix, str):
        raise ValueError(
            f"harness config: field {name!r}.prompt_suffix must be a string, got {_short(suffix)}"
        )
    fraction = value["trigger_fraction"]
    if (
        # Both ends EXCLUSIVE. At exactly 1 the pre-turn door fires only once the window
        # has already passed the whole limit — for a real provider, after the request
        # overflowed. The agent still recovers through the overflow door, but it pays a
        # wasted provider call and spends its one recovery attempt to reach a state it
        # could have reached before calling. That is not a strategy anything measures,
        # so the surface stops offering it rather than a test forbidding what the door
        # allows: the door and the published menu have to say the same thing.
        not isinstance(fraction, int | float)
        or isinstance(fraction, bool)
        or not 0 < float(fraction) < 1
    ):
        raise ValueError(
            f"harness config: field {name!r}.trigger_fraction must be greater than 0 "
            f"and less than 1"
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
        suffix,
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
    for key in ("max_attempts", "base_delay_ms"):
        item = value[key]
        low, high = _INT_BOUNDS[key]
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < low
            or (high is not None and item > high)
        ):
            raise ValueError(_bound_message(f"{name!r}.{key}", item, low, high))
    return RetryPolicy(strategy, value["max_attempts"], value["base_delay_ms"])


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
        if name in _INT_BOUNDS:
            item["min"], item["max"] = _INT_BOUNDS[name]
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
            item["strategies"] = sorted(_truncation_menu(name))
            item["parameters"] = {
                "budget": {"type": "int", "positive": True},
                "tail_fraction": {"type": "float", "exclusive_min": 0, "exclusive_max": 1},
            }
        elif name == "compaction":
            item["strategies"] = sorted(_COMPACTION_STRATEGIES)
            reserve_min, reserve_max = _COMPACTION_RESERVE_BOUNDS
            item["parameters"] = {
                "keep_head": {"type": "int", "positive": True},
                "keep_tail": {"type": "int", "positive": True},
                "trigger_fraction": {"type": "float", "exclusive_min": 0, "exclusive_max": 1},
                "summary_max_tokens": {"type": "int", "positive": True},
                "recent_token_reserve": {"type": "int", "min": reserve_min, "max": reserve_max},
                "completion_reserve": {"type": "int", "min": reserve_min, "max": reserve_max},
                "checkpoint_fallback": {"type": "str", "enum": sorted(_TRUNCATION_STRATEGIES)},
                "prompt_suffix": {"type": "str"},
            }
        elif name == "retry":
            item["strategies"] = sorted(_RETRY_STRATEGIES)
            item["parameters"] = {
                key: {"type": "int", "min": _INT_BOUNDS[key][0], "max": _INT_BOUNDS[key][1]}
                for key in ("max_attempts", "base_delay_ms")
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
            # a JSON `1` is a valid temperature; null stays None (no field sent)
            kwargs[key] = None if raw[key] is None else float(raw[key])
        else:
            kwargs[key] = raw[key]
    return HarnessConfig(**kwargs)


CONFIG = load_config()
