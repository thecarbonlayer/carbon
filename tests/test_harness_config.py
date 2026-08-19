"""The editable surface has a shape, not a content (harness_config).

These tests pin the *structure* of the config primitive — it loads, it's frozen,
the loader rejects malformed files loudly, and the legacy module-level names are
pure re-exports of the config values. They deliberately do NOT pin any knob's
*value*: the whole point of the surface is that values change (an editor bumps
the version and rewrites the file); the verifier must not entangle with them.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from harness import harness_config
from harness.harness_config import (
    CONFIG,
    CONFIG_PATH,
    CompactionPolicy,
    HarnessConfig,
    RetryPolicy,
    TruncationPolicy,
    config_schema,
    load_config,
    surface_manifest,
)


def _valid_raw() -> dict:
    """The checked-in config as a plain dict — the base for mutation tests."""
    return json.loads(CONFIG_PATH.read_text())


def _write(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "harness_config.json"
    p.write_text(json.dumps(raw))
    return p


def test_config_loads_and_is_typed():
    assert isinstance(CONFIG, HarnessConfig)
    assert isinstance(CONFIG.version, int)
    assert isinstance(CONFIG.system_prompt, str)
    assert isinstance(CONFIG.max_tool_steps, int)
    assert isinstance(CONFIG.default_context_limit, int)
    assert isinstance(CONFIG.verify_attempts, int)
    assert isinstance(CONFIG.require_run, bool)
    assert isinstance(CONFIG.max_item_chars, int)
    assert isinstance(CONFIG.file_injection, TruncationPolicy)
    assert isinstance(CONFIG.tool_output, TruncationPolicy)
    assert isinstance(CONFIG.compaction, CompactionPolicy)
    assert isinstance(CONFIG.retry, RetryPolicy)
    assert isinstance(CONFIG.compaction_prompt, str)
    assert isinstance(CONFIG.memory_search_limit, int)
    assert isinstance(CONFIG.attach_pattern, str)
    assert CONFIG.temperature is None or isinstance(CONFIG.temperature, float)
    assert isinstance(CONFIG.max_tokens, int)


def test_set_fields_are_frozensets_of_str():
    for value in (CONFIG.approval_tools, CONFIG.code_extensions):
        assert isinstance(value, frozenset)
        assert all(isinstance(x, str) for x in value)


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.max_tool_steps = 99  # type: ignore[misc]


def test_load_config_roundtrips_the_checked_in_file():
    assert load_config(CONFIG_PATH) == CONFIG


def test_unknown_key_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["mystery_knob"] = 7
    with pytest.raises(ValueError, match="unknown"):
        load_config(_write(tmp_path, raw))


def test_missing_field_is_rejected(tmp_path):
    raw = _valid_raw()
    del raw["max_tool_steps"]
    with pytest.raises(ValueError, match="missing"):
        load_config(_write(tmp_path, raw))


def test_wrong_type_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["max_tool_steps"] = "six"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_bool_is_not_an_int(tmp_path):
    # bool is a subclass of int in Python; the door must not let True through
    # where an integer knob is expected.
    raw = _valid_raw()
    raw["version"] = True
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_non_positive_int_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["max_tool_steps"] = 0
    with pytest.raises(ValueError, match="positive"):
        load_config(_write(tmp_path, raw))


def test_non_compiling_attach_pattern_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["attach_pattern"] = "@(unclosed"
    with pytest.raises(ValueError, match="attach_pattern"):
        load_config(_write(tmp_path, raw))


def test_groupless_attach_pattern_is_rejected(tmp_path):
    # The use site extracts the path via group(1); a pattern with no capture
    # group compiles fine but would break every @path delivery.
    raw = _valid_raw()
    raw["attach_pattern"] = "@\\S+"
    with pytest.raises(ValueError, match="attach_pattern"):
        load_config(_write(tmp_path, raw))


def test_non_string_in_set_field_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["approval_tools"] = ["bash", 3]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_unknown_strategy_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["tool_output"]["strategy"] = "run_whatever_code_refinery_sent"
    with pytest.raises(ValueError, match="strategy must be one of"):
        load_config(_write(tmp_path, raw))


def test_invalid_strategy_parameter_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["compaction"]["trigger_fraction"] = 1.5
    with pytest.raises(ValueError, match="trigger_fraction"):
        load_config(_write(tmp_path, raw))


def test_retry_attempts_are_hard_bounded(tmp_path):
    raw = _valid_raw()
    raw["retry"]["max_attempts"] = 6
    with pytest.raises(ValueError, match="from 1 to 5"):
        load_config(_write(tmp_path, raw))


def test_non_object_document_is_rejected(tmp_path):
    p = tmp_path / "harness_config.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_config(p)


def test_reexports_equal_config_values():
    """Legacy names stay importable, but are pure views of the config."""
    import harness.agent as agent
    import harness.compaction as compaction
    import harness.context as context
    import harness.limits as limits

    assert agent.DEFAULT_SYSTEM == CONFIG.system_prompt
    assert agent.MAX_TOOL_STEPS == CONFIG.max_tool_steps
    assert agent.DEFAULT_CONTEXT_LIMIT == CONFIG.default_context_limit
    assert agent.APPROVAL_TOOLS == CONFIG.approval_tools
    assert agent.CODE_EXTENSIONS == CONFIG.code_extensions
    assert limits.MAX_ITEM_CHARS == CONFIG.max_item_chars
    assert compaction.COMPACTION_PROMPT == CONFIG.compaction_prompt
    assert context._ATTACH.pattern == CONFIG.attach_pattern


def test_surface_manifest_separates_choices_from_invariants():
    manifest = surface_manifest()
    editable = {item["name"]: item for item in manifest["editable"]}
    locked = {item["name"]: item for item in manifest["locked_fields"]}
    immutable = {item["name"] for item in manifest["immutable"]}
    assert editable["tool_output"]["strategies"] == ["head_tail", "keep_head", "offload_to_file"]
    assert editable["file_injection"]["strategies"] == ["head_tail", "keep_head"]
    assert locked["version"]["editable"] is False
    assert "verification integrity" in locked["require_run"]["locked_reason"]
    assert {
        "tool_argument_validation",
        "unique_atomic_edits",
        "verification_integrity",
        "workspace_and_secret_boundaries",
    } <= immutable


def test_ctor_defaults_come_from_config():
    from harness.agent import Agent

    a = Agent(agents_dir=str(Path(__file__).parent))  # dodge the ambient AGENTS.md
    assert a.context_limit == CONFIG.default_context_limit
    assert a.verify_attempts == CONFIG.verify_attempts
    assert a.require_run == CONFIG.require_run
    assert a.max_tokens == CONFIG.max_tokens
    assert a.temperature == CONFIG.temperature


def test_tui_approval_tools_read_the_config():
    import ui.tui as tui

    assert tui.APPROVAL_TOOLS == CONFIG.approval_tools


# --- telemetry slice 2: single-source int bounds (Phase 1 §4) ----------------


def test_compaction_optional_keys_published_with_contract_bounds():
    """The three ``token_budget_checkpoint`` knobs are on the published surface with
    exactly the bounds/enum the frozen contract (§4) specifies."""
    params = {f["name"]: f for f in config_schema()}["compaction"]["parameters"]
    assert params["recent_token_reserve"] == {"type": "int", "min": 0, "max": 100_000}
    assert params["completion_reserve"] == {"type": "int", "min": 0, "max": 100_000}
    assert params["checkpoint_fallback"] == {
        "type": "str",
        "enum": ["head_tail", "keep_head"],
    }


def test_every_int_bounds_field_publishes_what_the_loader_enforces(tmp_path):
    """``_INT_BOUNDS`` is the single source for both the loader and the published
    schema — read both sides from that same table rather than pinning numbers here,
    so this test cannot silently pass while the two drift apart.

    Covers both shapes: the six top-level ``HarnessConfig`` fields (published on
    their own schema item) and retry's two nested fields (published inside the
    ``retry`` item's ``parameters``).
    """
    schema = {f["name"]: f for f in config_schema()}
    retry_params = schema["retry"]["parameters"]
    for key, (low, high) in harness_config._INT_BOUNDS.items():
        published = (
            (schema[key]["min"], schema[key]["max"])
            if key in schema
            else (
                retry_params[key]["min"],
                retry_params[key]["max"],
            )
        )
        assert published == (low, high), f"{key}: published {published} != enforced ({low}, {high})"


def test_int_above_its_published_max_is_rejected_naming_the_field(tmp_path):
    """A value above ``_INT_BOUNDS``'s max is rejected, and the message names the
    field — for every bounded field that actually has a ceiling."""
    for key, (_low, high) in harness_config._INT_BOUNDS.items():
        if high is None:  # unbounded above (max_item_chars) — nothing to violate
            continue
        raw = _valid_raw()
        if key in raw:
            raw[key] = high + 1
        else:
            raw["retry"][key] = high + 1
        with pytest.raises(ValueError, match=key):
            load_config(_write(tmp_path, raw))
