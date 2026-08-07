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
