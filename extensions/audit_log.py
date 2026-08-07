"""audit_log — wrap every tool with an audit trail, and add a tool to read it back.

The shipped example extension (mirrors ``skills/sign-off/SKILL.md`` as the one
shipped example skill). It proves the two ``ToolRegistry`` methods an
extension needs — ``register()`` and ``wrap()`` — are enough to build a real,
useful extension without any event-hook system (dev-notes/adr/0003).

The log is written relative to the current working directory (extensions only
receive a ``ToolRegistry``, not a workspace root) — fine for this example;
a workspace-aware audit log is a consumer's own extension to write.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from harness.tools import Tool, ToolRegistry

_LOG_PATH = Path(".carbon") / "audit.log"


def _audited(name: str, func: Callable[..., str]) -> Callable[..., str]:
    def wrapped(**kwargs: object) -> str:
        result = func(**kwargs)
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a") as f:
            f.write(f"{time.time():.0f} {name} {kwargs!r} -> {result[:200]!r}\n")
        return result

    return wrapped


def _read_audit_log() -> str:
    if not _LOG_PATH.is_file():
        return "no audit log yet"
    return _LOG_PATH.read_text()


def setup(registry: ToolRegistry) -> None:
    for name in registry.names():
        registry.wrap(name, lambda func, name=name: _audited(name, func))
    registry.register(
        Tool(
            name="read_audit_log",
            description="Read the audit log of every tool call made this session.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=_read_audit_log,
            mutates=False,
        )
    )
