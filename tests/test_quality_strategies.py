"""Quality-critical strategy seams and locked correctness invariants."""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from unittest.mock import patch

import pytest

from harness import compaction, harness_config
from harness.agent import Agent
from harness.harness_config import (
    CONFIG,
    CONFIG_PATH,
    RetryPolicy,
    TruncationPolicy,
    load_config,
)
from harness.limits import truncate
from harness.tools import Tool, ToolRegistry, read_file, read_file_tool
from harness.workspace import Workspace
from model import LLMResponse, Provider


def _scripted(responses: list[LLMResponse]) -> Provider:
    items = iter(responses)
    return Provider("fake://quality", "fake", responder=lambda messages, **kwargs: next(items))


def test_head_tail_retains_both_failure_contexts():
    policy = TruncationPolicy("head_tail", 10, 0.6)
    out = truncate("ABCDEFGHIJ" * 5, policy)
    assert out.startswith("ABCD")
    assert out.endswith("EFGHIJ")
    assert "truncated" in out


def test_the_post_door_recut_validates_what_the_door_would_have():
    """``recut`` is the one truncation helper that takes raw numbers instead of a
    policy, which is exactly why it needs the policy's validation: it took a
    ``tail_fraction`` of 2.0 without a word and returned 61 chars for a budget of 10 —
    a cut bigger than the thing it was cutting to.

    It is also the one that skips the door, so its name and its docstring say what it is
    for (text already through a door) rather than reading as a general entry point."""
    from harness.limits import recut

    assert recut("ABCDEFGHIJ" * 5, 10, 0.6).endswith("EFGHIJ")
    assert recut("short", 10, 0.6) == "short"
    with pytest.raises(ValueError, match="tail_fraction"):
        recut("ABCDEFGHIJ" * 5, 10, 2.0)
    with pytest.raises(ValueError, match="tail_fraction"):
        recut("ABCDEFGHIJ" * 5, 10, 0.0)
    with pytest.raises(ValueError, match="budget"):
        recut("ABCDEFGHIJ" * 5, 0, 0.5)


def test_sandbox_ceiling_does_not_destroy_failure_tail():
    """The sandbox's ceiling is blunt and policy-free, but it is still head AND tail:
    the last thing a failing command writes is usually the reason it failed. It also
    sits above anything real now — the 100k it replaced was measured to protect nothing
    and cost a second truncation door on the same text."""
    from harness import sandbox

    assert "truncated" not in sandbox._cap("PASS\n" * 30_000)  # 150k is not "chatty"

    text = "PASS\n" * (sandbox._MAX_OUTPUT // 5 + 1000) + "FINAL-FAILURE-TAIL"
    out = sandbox._cap(text)
    assert out.startswith("PASS")
    assert out.endswith("FINAL-FAILURE-TAIL")
    assert "truncated" in out
    assert len(out) <= sandbox._MAX_OUTPUT + 100  # the ceiling holds, plus its marker


def test_read_file_supports_precise_line_ranges(tmp_path):
    (tmp_path / "large.txt").write_text("".join(f"line-{i}\n" for i in range(1, 501)))
    out = read_file("large.txt", root=tmp_path, start_line=201, end_line=203)
    assert "[large.txt: lines 201-203 of 500]" in out
    assert "line-201\nline-202\nline-203\n" in out
    assert "start_line=204" in out


def test_read_file_tool_advertises_and_accepts_ranges(tmp_path):
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n")
    tool = read_file_tool(tmp_path)
    assert {"start_line", "end_line"} <= set(tool.parameters["properties"])
    assert "lines 2-3 of 3" in tool.func(path="notes.txt", start_line=2, end_line=3)


def test_edit_rejects_ambiguous_match_without_mutating(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("x.py", "timeout = 5\n\ntimeout = 5\n")
    before = (tmp_path / "x.py").read_text()
    result = ws.edit("x.py", "timeout = 5", "timeout = 30")
    assert "occurs 2 times" in result
    assert (tmp_path / "x.py").read_text() == before


def test_unique_edit_is_atomic_and_returns_diff(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("x.py", "alpha = 1\nbeta = 2\n")
    result = ws.edit("x.py", "beta = 2", "beta = 3")
    assert (tmp_path / "x.py").read_text() == "alpha = 1\nbeta = 3\n"
    assert "--- a/x.py" in result and "+++ b/x.py" in result
    assert "-beta = 2" in result and "+beta = 3" in result
    assert not list(tmp_path.glob(".x.py.carbon-edit"))


def test_tool_arguments_are_validated_before_execution():
    calls = {"n": 0}

    def mutate(path: str) -> str:
        calls["n"] += 1
        return path

    reg = ToolRegistry()
    reg.register(
        Tool(
            "mutate",
            "",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            mutate,
        )
    )
    assert "missing required" in reg.call("mutate", "{}")
    assert "must be string" in reg.call("mutate", '{"path": 7}')
    assert "unknown fields" in reg.call("mutate", '{"path": "x", "oops": true}')
    assert calls["n"] == 0


def test_truncated_tool_call_never_executes():
    calls = {"n": 0}
    reg = ToolRegistry()
    reg.register(
        Tool(
            "mutate",
            "",
            {"type": "object", "properties": {}, "required": []},
            lambda: calls.__setitem__("n", calls["n"] + 1) or "changed",
        )
    )
    response = LLMResponse(
        content="",
        tool_calls=[{"id": "cut", "function": {"name": "mutate", "arguments": "{}"}}],
        finish_reason="length",
    )
    result = Agent(provider=_scripted([response]), tools=reg).run("go")
    assert result.stop_reason == "incomplete_response"
    assert calls["n"] == 0
    assert "no tool calls were executed" in result.text


def test_structured_compaction_serializes_tool_names_arguments_and_prior_summary():
    messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t1",
                    "function": {
                        "name": "edit_file",
                        "arguments": '{"path":"a.py","old":"x","new":"y"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "edited a.py"},
        {
            "role": "system",
            "content": "[summary of earlier conversation]\nEARLY-DECISION-7",
        },
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "done"},
    ]
    captured: dict = {}

    def summarize(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return LLMResponse(content="CHECKPOINT")

    with patch.object(compaction, "chat", side_effect=summarize):
        out = compaction.compact(
            messages,
            keep_head=1,
            keep_tail=2,
            strategy="structured_checkpoint",
            summary_max_tokens=777,
        )
    transcript = captured["payload"][1]["content"]
    assert "edit_file" in transcript
    assert '\\"path\\":\\"a.py\\"' in transcript
    assert "EARLY-DECISION-7" in transcript
    assert captured["kwargs"]["max_tokens"] == 777
    assert "strategy=structured_checkpoint" in out[1]["content"]


def test_checked_in_strategy_defaults_are_quality_oriented():
    assert CONFIG.tool_output.strategy == "head_tail"
    assert CONFIG.compaction.strategy == "structured_checkpoint"
    assert CONFIG.compaction.trigger_fraction < 1
    assert CONFIG.retry.strategy == "backoff"
    assert CONFIG.retry.max_attempts <= 5
    assert CONFIG.max_tokens >= 4096


def test_transient_provider_failure_retries_then_recovers():
    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return LLMResponse(content="RECOVERED")

    provider = Provider("fake://retry", "fake", responder=responder)
    with patch("harness.agent.time.sleep"):
        result = Agent(provider=provider).run("go")
    assert result.text == "RECOVERED"
    assert state["calls"] == 2


def test_transient_retry_is_bounded():
    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        raise RuntimeError("429 rate limit")

    provider = Provider("fake://retry", "fake", responder=responder)
    with (
        patch("harness.agent.time.sleep"),
        pytest.raises(RuntimeError, match="rate limit"),
    ):
        Agent(provider=provider).run("go")
    assert state["calls"] == CONFIG.retry.max_attempts


def test_context_overflow_compacts_active_history_and_retries():
    state = {"main_calls": 0, "summary_calls": 0}

    def responder(messages, **kwargs):
        is_summary = bool(
            messages
            and messages[0].get("role") == "system"
            and "context summarizer" in str(messages[0].get("content", ""))
        )
        if is_summary:
            state["summary_calls"] += 1
            return LLMResponse(content="STRUCTURED CHECKPOINT")
        state["main_calls"] += 1
        if state["main_calls"] == 1:
            raise RuntimeError("maximum context length exceeded")
        return LLMResponse(content="OVERFLOW-RECOVERED")

    provider = Provider("fake://overflow", "fake", responder=responder)
    agent = Agent(provider=provider)
    agent.messages = [{"role": "user", "content": f"old-{i}"} for i in range(10)]
    result = agent.run("continue")
    assert result.text == "OVERFLOW-RECOVERED"
    assert state == {"main_calls": 2, "summary_calls": 1}
    assert agent.compaction_count == 1
    assert agent.retry_count == 1


# --- regressions found reviewing the strategy surface -------------------------
def test_edit_preserves_file_mode_and_leaves_no_temp_file(tmp_path):
    """The atomic rename must not strip an executable bit off a script."""
    script = tmp_path / "run.sh"
    script.write_text("echo one\n")
    script.chmod(0o755)

    Workspace(root=tmp_path).edit("run.sh", "one", "two")

    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    assert script.read_text() == "echo two\n"
    assert [p.name for p in tmp_path.iterdir()] == ["run.sh"]


def test_temperature_accepts_a_json_integer(tmp_path):
    """JSON has one number type; `1` is a legitimate proposal, not a type error."""
    raw = json.loads(CONFIG_PATH.read_text())
    raw["temperature"] = 1
    path = tmp_path / "harness_config.json"
    path.write_text(json.dumps(raw))

    loaded = load_config(path)

    assert loaded.temperature == 1.0
    assert isinstance(loaded.temperature, float)


def test_temperature_out_of_range_is_still_rejected(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text())
    raw["temperature"] = 3
    path = tmp_path / "harness_config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="from 0 to 2"):
        load_config(path)


def test_temperature_null_defers_to_the_provider(tmp_path):
    """null is not a value pin — it means the request carries no temperature
    field, so the provider's own default applies (parity with harnesses that
    never send the knob)."""
    raw = json.loads(CONFIG_PATH.read_text())
    raw["temperature"] = None
    path = tmp_path / "harness_config.json"
    path.write_text(json.dumps(raw))

    loaded = load_config(path)

    assert loaded.temperature is None


def test_truncated_empty_response_does_not_leak_none():
    provider = _scripted([LLMResponse(content=None, finish_reason="length")])

    result = Agent(provider=provider).run("go")

    assert "None" not in result.text
    assert result.stop_reason == "incomplete_response"


def test_read_file_range_on_an_empty_file_is_not_an_error(tmp_path):
    (tmp_path / "empty.txt").write_text("")

    out = read_file("empty.txt", root=tmp_path, start_line=1)

    assert "error" not in out and "empty file" in out


def test_retries_do_not_inflate_the_reported_turn_count():
    """`turns` counts completed model calls — a flaky endpoint is not a chattier agent."""
    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return LLMResponse(content="RECOVERED")

    agent = Agent(provider=Provider("fake://flaky", "fake", responder=responder))
    result = agent.run("go")

    assert result.text == "RECOVERED"
    assert state["calls"] == 2  # one failed attempt, one success
    assert result.turns == 1  # but only one completed call
    assert agent.retry_count == 1


# --- every advertised strategy must be implemented ---------------------------
# The allowed-name frozensets live in harness_config.py; the code that acts on a
# name lives at each use site. Nothing couples them, so a name can be added to a
# menu with no implementation behind it. Truncation and compaction would crash
# mid-turn; retry would do something worse — an unimplemented name makes the
# `strategy == "backoff"` conjunct false, which is silently identical to
# fail_fast, so Refinery could measure a strategy that does not exist and record
# another one's numbers as its result.
#
# Each test below runs every name in a menu and asserts the observable behaviors
# are all distinct. A name with no implementation collides with an existing one
# (or raises) and fails here, at `verify` time, instead of mid-run.
def test_every_truncation_strategy_behaves_distinctly():
    text = "HEAD" + "-" * 200 + "TAIL"
    seen = {
        name: truncate(text, TruncationPolicy(name, 20, 0.5))
        for name in harness_config._TRUNCATION_STRATEGIES
    }
    assert len(set(seen.values())) == len(seen), f"indistinguishable truncation: {seen}"


def test_every_compaction_strategy_behaves_distinctly():
    history = [{"role": "user", "content": f"msg-{i}"} for i in range(12)]
    prompts: dict[str, str] = {}

    for name in harness_config._COMPACTION_STRATEGIES:
        captured: list[str] = []

        def responder(messages, _sink=captured, **kwargs):
            _sink.append(str(messages[0]["content"]))
            return LLMResponse(content="SUMMARY")

        compaction.compact(
            history,
            strategy=name,
            provider=Provider("fake://compact", "fake", responder=responder),
        )
        prompts[name] = captured[0]

    assert len(set(prompts.values())) == len(prompts), "indistinguishable compaction prompts"


def test_every_retry_strategy_behaves_distinctly():
    """A no-crash check would pass an unimplemented name here — it has to be
    behavior, because an unknown name is byte-identical to fail_fast."""
    attempts: dict[str, int] = {}

    for name in harness_config._RETRY_STRATEGIES:
        state = {"calls": 0}

        def responder(messages, _state=state, **kwargs):
            _state["calls"] += 1
            if _state["calls"] == 1:
                raise RuntimeError("503 temporarily unavailable")
            return LLMResponse(content="RECOVERED")

        provider = Provider("fake://retry", "fake", responder=responder)
        policy = replace(CONFIG, retry=RetryPolicy(name, 3, 0))
        with patch("harness.agent.CONFIG", policy), patch("harness.agent.time.sleep"):
            try:
                Agent(provider=provider).run("go")
            except RuntimeError:
                pass
        attempts[name] = state["calls"]

    assert len(set(attempts.values())) == len(attempts), f"indistinguishable retry: {attempts}"
