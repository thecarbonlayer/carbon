"""The shipped example extension (dev-notes/adr/0003) — proves register() and
wrap() together are enough for a real, useful extension, with no event-hook
system."""

from __future__ import annotations

from pathlib import Path

from harness.extensions import load_extensions
from harness.tools import Tool, ToolRegistry

_EXTENSIONS_DIR = Path(__file__).resolve().parents[1] / "extensions"


def test_audit_log_wraps_and_registers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # audit.log is written relative to cwd
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="add",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            func=lambda expression: "42",
            mutates=False,
        )
    )

    loaded = load_extensions(registry, _EXTENSIONS_DIR)

    assert "audit_log" in loaded
    assert "read_audit_log" in registry.names()

    result = registry.call("calculator", '{"expression": "6*7"}')
    assert result == "42"

    log = registry.call("read_audit_log", "{}")
    assert "calculator" in log
    assert "42" in log


def test_audit_log_records_a_failing_call_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="boom",
            description="always fails",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
            mutates=False,
        )
    )

    load_extensions(registry, _EXTENSIONS_DIR)

    # ToolRegistry.call catches the re-raised exception and reports it as a
    # string (tool errors are fed back to the model, not raised to the caller).
    result = registry.call("boom", "{}")
    assert result.startswith("error:")

    log = registry.call("read_audit_log", "{}")
    assert "boom" in log
    assert "kaboom" in log
