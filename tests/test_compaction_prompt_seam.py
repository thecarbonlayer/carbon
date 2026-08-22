"""The summarizer's FULL instruction is config-reachable (``compaction.prompt_suffix``).

``compaction_prompt`` was only the BASE of what the summarizer sees: each strategy
appended its own hard-coded tail — the checkpoint headings, the carry-forward update
instruction, the preserve-verbatim line — in code, out of the editable surface's
reach. ``compaction.prompt_suffix`` exposes that tail as data.

Two properties are pinned here:

- **Default byte-identity.** With the knob unset, the assembled prompt for every
  strategy is byte-identical to the pre-seam text. The suffix literals below
  deliberately restate that text rather than importing the module's constants —
  a change to a default suffix changes live behavior under every config that does
  not set the knob (and invalidates every recorded baseline), so it must turn a
  test red and be made deliberately.
- **The knob reaches the model.** A suffix set in a config FILE arrives, through
  ``load_config`` and the policy, as part of the system message the summarizer is
  actually sent.

The base half of the prompt is read from the config, never pinned: that field is
the improvement loop's to change. Likewise every test that depends on the knob
being unset SETS it to None explicitly rather than trusting the checked-in file,
so a later legitimate config edit cannot read as a regression here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from harness import compaction
from harness.harness_config import CONFIG, CONFIG_PATH, load_config
from model import LLMResponse

# The pre-seam suffix of each suffix-bearing strategy, byte for byte. Restated as
# literals on purpose — asserting against the module's own constants would prove
# nothing (see the module docstring above).
_HEADINGS = (
    "Return a cumulative structured checkpoint with these headings: Goal, "
    "Constraints, Decisions, Completed work, Files read or changed, Exact commands "
    "and tool calls, Failures and rejected approaches, Current state, Next steps. "
)
_VERBATIM = "Preserve identifiers, paths, arguments, receipts, and error tails verbatim."
STRUCTURED_SUFFIX = (
    _HEADINGS
    + "Merge any earlier [summary of earlier conversation] into this checkpoint. "
    + _VERBATIM
)
INCREMENTAL_SUFFIX = (
    _HEADINGS + "You are UPDATING an existing checkpoint, not writing a new one: carry "
    "every still-true fact from it forward unchanged, revise what the new "
    "messages have changed, and add what they introduced. Never drop a fact "
    "merely because it is old. " + _VERBATIM
)


def _messages() -> list[dict]:
    """Enough complete turns that every strategy finds a non-empty middle."""
    head = [{"role": "user", "content": "head"}, {"role": "assistant", "content": "ack"}]
    filler = [
        m
        for i in range(6)
        for m in (
            {"role": "user", "content": f"filler turn {i}"},
            {"role": "assistant", "content": f"acknowledged {i}"},
        )
    ]
    return head + filler


def _config_with(suffix: str | None):
    """The live config with ``compaction.prompt_suffix`` pinned to ``suffix``."""
    return replace(CONFIG, compaction=replace(CONFIG.compaction, prompt_suffix=suffix))


def _rendered_prompt(strategy: str, cfg=None) -> str:
    """Run a real ``compact()`` and return the system message the summarizer saw."""
    captured: dict = {}

    def summarize(payload, **kwargs):
        captured["payload"] = payload
        return LLMResponse(content="CHECKPOINT")

    with (
        patch.object(compaction, "CONFIG", cfg if cfg is not None else _config_with(None)),
        patch.object(compaction, "chat", side_effect=summarize),
    ):
        compaction.compact(_messages(), strategy=strategy)
    payload = captured["payload"]
    assert payload[0]["role"] == "system"
    return payload[0]["content"]


# --- default byte-identity, one test per strategy ---------------------------------


def test_unset_suffix_renders_the_shipped_strategy_prompt_byte_identically():
    prompt = _rendered_prompt("token_budget_checkpoint")
    assert prompt == compaction.COMPACTION_PROMPT + "\n\n" + INCREMENTAL_SUFFIX


def test_unset_suffix_renders_the_structured_prompt_byte_identically():
    prompt = _rendered_prompt("structured_checkpoint")
    assert prompt == compaction.COMPACTION_PROMPT + "\n\n" + STRUCTURED_SUFFIX


def test_unset_suffix_renders_the_plain_prompt_as_the_bare_base():
    # summarize_middle never had a suffix: base only, no separator either.
    assert _rendered_prompt("summarize_middle") == compaction.COMPACTION_PROMPT


# --- the knob ---------------------------------------------------------------------


def test_a_configured_suffix_replaces_every_strategys_default():
    for strategy in ("summarize_middle", "structured_checkpoint", "token_budget_checkpoint"):
        prompt = _rendered_prompt(strategy, cfg=_config_with("CARRY THE OLDEST FACTS FIRST."))
        expected = compaction.COMPACTION_PROMPT + "\n\nCARRY THE OLDEST FACTS FIRST."
        assert prompt == expected, strategy


def test_an_empty_suffix_strips_the_tail_leaving_the_base_alone():
    for strategy in ("summarize_middle", "structured_checkpoint", "token_budget_checkpoint"):
        assert _rendered_prompt(strategy, cfg=_config_with("")) == compaction.COMPACTION_PROMPT, (
            strategy
        )


def test_a_suffix_set_in_a_config_file_reaches_the_summarizer(tmp_path: Path):
    """End to end through the editable surface: file -> load_config -> policy -> prompt.

    This is the loop's actual route — it edits the JSON, never the code — so the
    seam is only real if the file's value is what the summarizer is sent.
    """
    raw = json.loads(CONFIG_PATH.read_text())
    raw["compaction"]["prompt_suffix"] = "FROM THE FILE."
    p = tmp_path / "harness_config.json"
    p.write_text(json.dumps(raw))
    cfg = load_config(p)
    prompt = _rendered_prompt(raw["compaction"]["strategy"], cfg=cfg)
    assert prompt == compaction.COMPACTION_PROMPT + "\n\nFROM THE FILE."
