"""Tool exposure (strategy-surface seam 3, Select) — the ``tool_exposure`` knob.

``ToolRegistry.specs()`` has always returned every registered tool, every turn.
This knob makes WHICH tools are offered an editable, bounded choice:

- ``all`` (the default): today's exposure, byte for byte;
- ``allowlist``: a fixed subset, in the allowlist's own order;
- ``query_match``: rank tools against the current user turn, offer the top k.

Two properties carry the whole seam and both are pinned here:

1. **Default neutrality.** The checked-in config does not carry the field, the
   loaded default is ``all``, and the offered-tools payload an agent sends is
   byte-identical to ``registry.specs()`` — landing the seam changes nothing
   until an editor turns the knob (roadmap ground rule 3).
2. **Validation at the door.** Values validate the way the neighboring policy
   objects do: menu membership, per-strategy required params, and loud rejection
   of params that belong to a different strategy — an editor who sets ``k``
   under ``allowlist`` believes they changed something, and silence would let
   the file on disk and the behavior in memory disagree.
"""

from __future__ import annotations

import json

import pytest

from harness.agent import Agent
from harness.harness_config import (
    CONFIG,
    CONFIG_PATH,
    ToolExposurePolicy,
    config_schema,
    load_config,
    surface_manifest,
)
from harness.tools import Tool, ToolRegistry, exposed_specs
from model import LLMResponse, Provider


def _tool(name: str, description: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: "ok",
        mutates=False,
    )


def _registry() -> ToolRegistry:
    """Four tools with distinct vocabularies, in a known registration order."""
    reg = ToolRegistry()
    reg.register(_tool("read_file", "Read a workspace file by path."))
    reg.register(_tool("search_text", "Search workspace files for a literal string."))
    reg.register(_tool("calculator", "Evaluate an arithmetic expression."))
    reg.register(_tool("weather_report", "Current weather conditions for a city."))
    return reg


def _capturing_agent(tmp_path, reg: ToolRegistry, **kwargs) -> tuple[Agent, list]:
    """An offline agent whose provider records the ``tools`` payload it was sent."""
    offered: list = []

    def responder(messages, **kw) -> LLMResponse:
        offered.append(kw.get("tools"))
        return LLMResponse(content="done")

    provider = Provider(base_url="fake://exposure", model="fake", responder=responder)
    agent = Agent(
        provider=provider,
        model="fake",
        tools=reg,
        agents_dir=str(tmp_path),  # dodge the ambient AGENTS.md
        **kwargs,
    )
    return agent, offered


def _valid_raw() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _write(tmp_path, raw: dict):
    p = tmp_path / "harness_config.json"
    p.write_text(json.dumps(raw))
    return p


# --- default neutrality -------------------------------------------------------


def test_checked_in_config_omits_the_field_and_defaults_to_all():
    """The seam lands without touching the file: no field on disk, so the version
    external baselines pin to does not move, and the loaded default is ``all``."""
    assert "tool_exposure" not in _valid_raw()
    assert CONFIG.tool_exposure == ToolExposurePolicy("all")


def test_default_offered_tools_payload_is_byte_identical_to_the_registry(tmp_path):
    """THE neutrality pin: with the knob unset, the tools payload the provider
    receives is byte-for-byte what ``registry.specs()`` has always produced —
    same entries, same order, same serialization."""
    reg = _registry()
    agent, offered = _capturing_agent(tmp_path, reg)
    try:
        agent.send("hello")
    finally:
        agent.close()
    assert len(offered) == 1
    assert json.dumps(offered[0]) == json.dumps(reg.specs())


def test_all_strategy_returns_the_registry_specs_object_shape():
    reg = _registry()
    assert exposed_specs(reg, ToolExposurePolicy("all")) == reg.specs()
    assert exposed_specs(reg, None) == reg.specs()


# --- the selection mechanism --------------------------------------------------


def test_allowlist_selects_exactly_the_named_tools_in_allowlist_order():
    reg = _registry()
    policy = ToolExposurePolicy("allowlist", tools=("calculator", "read_file"))
    names = [s["function"]["name"] for s in exposed_specs(reg, policy)]
    assert names == ["calculator", "read_file"]


def test_allowlist_names_missing_from_the_registry_are_skipped():
    """The config cannot know a consumer's registry at load time, so an absent
    name is skipped at exposure time rather than refused at the door."""
    reg = _registry()
    policy = ToolExposurePolicy("allowlist", tools=("calculator", "no_such_tool"))
    names = [s["function"]["name"] for s in exposed_specs(reg, policy)]
    assert names == ["calculator"]


def test_query_match_ranks_by_token_overlap_and_takes_top_k():
    reg = _registry()
    policy = ToolExposurePolicy("query_match", k=2)
    query = "Search the workspace files for the string TODO"
    names = [s["function"]["name"] for s in exposed_specs(reg, policy, query=query)]
    assert names[0] == "search_text"
    assert len(names) == 2


def test_query_match_breaks_ties_by_registration_order():
    reg = _registry()
    policy = ToolExposurePolicy("query_match", k=4)
    # No token overlaps anything: every score is zero, so rank must fall back to
    # registration order rather than any incidental sort instability.
    names = [s["function"]["name"] for s in exposed_specs(reg, policy, query="zzz qqq")]
    assert names == ["read_file", "search_text", "calculator", "weather_report"]


def test_query_match_k_beyond_the_registry_offers_everything_ranked():
    reg = _registry()
    policy = ToolExposurePolicy("query_match", k=50)
    names = [s["function"]["name"] for s in exposed_specs(reg, policy, query="weather in Basel")]
    assert sorted(names) == sorted(reg.names())
    assert names[0] == "weather_report"


def test_query_match_ranking_is_case_insensitive():
    reg = _registry()
    policy = ToolExposurePolicy("query_match", k=1)
    names = [s["function"]["name"] for s in exposed_specs(reg, policy, query="WEATHER Basel")]
    assert names == ["weather_report"]


# --- the agent seam -----------------------------------------------------------


def test_agent_allowlist_offers_only_the_allowed_tools(tmp_path):
    reg = _registry()
    agent, offered = _capturing_agent(
        tmp_path, reg, tool_exposure=ToolExposurePolicy("allowlist", tools=("calculator",))
    )
    try:
        agent.send("what is 2+2")
    finally:
        agent.close()
    assert [s["function"]["name"] for s in offered[0]] == ["calculator"]


def test_agent_query_match_ranks_against_the_current_user_turn(tmp_path):
    reg = _registry()
    agent, offered = _capturing_agent(
        tmp_path, reg, tool_exposure=ToolExposurePolicy("query_match", k=1)
    )
    try:
        agent.send("Evaluate the arithmetic expression 17 * 3")
    finally:
        agent.close()
    assert [s["function"]["name"] for s in offered[0]] == ["calculator"]


def test_agent_empty_selection_sends_no_tools_field(tmp_path):
    """An allowlist that matches nothing offers nothing — the payload carries no
    tools field at all (None), never an empty list a provider may reject."""
    reg = _registry()
    agent, offered = _capturing_agent(
        tmp_path, reg, tool_exposure=ToolExposurePolicy("allowlist", tools=("absent",))
    )
    try:
        agent.send("hello")
    finally:
        agent.close()
    assert offered == [None]


def test_agent_default_exposure_comes_from_config(tmp_path):
    """No ctor value → the editable surface decides, same as ``tool_output``."""
    agent, offered = _capturing_agent(tmp_path, _registry())
    try:
        assert agent.tool_exposure is None
    finally:
        agent.close()


# --- validation at the door (policy object; the code seam) --------------------


def test_policy_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        ToolExposurePolicy("run_whatever_code_refinery_sent")


def test_policy_allowlist_requires_a_nonempty_tool_tuple():
    with pytest.raises(ValueError, match="tools"):
        ToolExposurePolicy("allowlist")
    with pytest.raises(ValueError, match="tools"):
        ToolExposurePolicy("allowlist", tools=())


def test_policy_allowlist_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        ToolExposurePolicy("allowlist", tools=("calculator", "calculator"))


def test_policy_query_match_requires_k_in_bounds():
    with pytest.raises(ValueError, match="k"):
        ToolExposurePolicy("query_match")
    with pytest.raises(ValueError, match="k"):
        ToolExposurePolicy("query_match", k=0)
    with pytest.raises(ValueError, match="k"):
        ToolExposurePolicy("query_match", k=51)
    with pytest.raises(ValueError, match="k"):
        ToolExposurePolicy("query_match", k=True)  # bool is not an int here


def test_policy_rejects_params_that_belong_to_another_strategy():
    """A param the chosen strategy never reads is a silent no-op the editor
    believes in — refuse it loudly instead."""
    with pytest.raises(ValueError, match="k"):
        ToolExposurePolicy("all", k=3)
    with pytest.raises(ValueError, match="tools"):
        ToolExposurePolicy("all", tools=("calculator",))
    with pytest.raises(ValueError, match="k"):
        ToolExposurePolicy("allowlist", tools=("calculator",), k=3)
    with pytest.raises(ValueError, match="tools"):
        ToolExposurePolicy("query_match", k=3, tools=("calculator",))


# --- validation at the door (the config file) ---------------------------------


def test_file_allowlist_value_loads(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "allowlist", "tools": ["read_file", "bash"]}
    cfg = load_config(_write(tmp_path, raw))
    assert cfg.tool_exposure == ToolExposurePolicy("allowlist", tools=("read_file", "bash"))


def test_file_query_match_value_loads(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "query_match", "k": 5}
    cfg = load_config(_write(tmp_path, raw))
    assert cfg.tool_exposure == ToolExposurePolicy("query_match", k=5)


def test_file_all_value_loads_and_matches_the_absent_default(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "all"}
    assert load_config(_write(tmp_path, raw)).tool_exposure == CONFIG.tool_exposure


def test_file_unknown_strategy_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "phase_gated"}
    with pytest.raises(ValueError, match="strategy must be one of"):
        load_config(_write(tmp_path, raw))


def test_file_unknown_key_inside_the_object_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "all", "mystery": 1}
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write(tmp_path, raw))


def test_file_allowlist_without_tools_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "allowlist"}
    with pytest.raises(ValueError, match="tools"):
        load_config(_write(tmp_path, raw))


def test_file_allowlist_with_non_string_entry_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["tool_exposure"] = {"strategy": "allowlist", "tools": ["bash", 3]}
    with pytest.raises(ValueError, match="tools"):
        load_config(_write(tmp_path, raw))


def test_file_query_match_k_out_of_bounds_is_rejected(tmp_path):
    for bad in (0, 51, "five", False):
        raw = _valid_raw()
        raw["tool_exposure"] = {"strategy": "query_match", "k": bad}
        with pytest.raises(ValueError, match="k"):
            load_config(_write(tmp_path, raw))


def test_file_cross_strategy_params_are_rejected(tmp_path):
    for value in (
        {"strategy": "all", "k": 3},
        {"strategy": "all", "tools": ["bash"]},
        {"strategy": "allowlist", "tools": ["bash"], "k": 3},
        {"strategy": "query_match", "k": 3, "tools": ["bash"]},
    ):
        raw = _valid_raw()
        raw["tool_exposure"] = value
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, raw))


# --- the published surface ----------------------------------------------------


def test_schema_publishes_the_menu_and_the_bounds_the_loader_enforces():
    item = {f["name"]: f for f in config_schema()}["tool_exposure"]
    assert item["editable"] is True
    assert item["strategies"] == ["all", "allowlist", "query_match"]
    assert item["optional"] is True  # absent = today's exposure
    assert item["parameters"]["k"]["min"] == 1
    assert item["parameters"]["k"]["max"] == 50
    assert item["parameters"]["k"]["strategy"] == "query_match"
    assert item["parameters"]["tools"]["strategy"] == "allowlist"


def test_surface_manifest_lists_tool_exposure_as_editable():
    editable = {f["name"] for f in surface_manifest()["editable"]}
    assert "tool_exposure" in editable
