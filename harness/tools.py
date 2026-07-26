"""Tools — the actions the model can ask the harness to run.

A tool is just a function plus a JSON-schema contract. The registry turns those
functions into OpenAI tool specs (so the model knows what it can call) and
dispatches the calls by name, parsing arguments and returning a string result —
or an error string the model can read and recover from.

Tools are an API surface you expose to a model: keep the list small, keep each
contract narrow, and validate arguments. ``calculator`` evaluates arithmetic
without ``eval``; ``read_file`` returns a file's contents — and as of ch-08 it
is confined to the workspace: a model-invoked tool must not wander the host
filesystem, so paths are resolved and must live under the working directory.
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no eval, just numbers + + - * / % **)."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("unsupported expression")

    result = ev(ast.parse(expression, mode="eval").body)
    return str(int(result) if result == int(result) else result)


def _is_secret_file(p: Path) -> bool:
    """A model-invoked read must never exfiltrate credentials. Refuse dotenv files,
    private keys, and PEM/key material even when they sit inside the workspace."""
    name = p.name
    return (
        name == ".env"
        or name.startswith(".env.")
        or name.startswith("id_")  # ssh private keys: id_rsa, id_ed25519, ...
        or p.suffix in (".pem", ".key")
    )


def read_file(
    path: str,
    root: str | Path | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Return a file's contents — confined to a root (ch-08 hardening).

    The model-invoked tool must not wander the host filesystem (no /etc/passwd) and
    must not read secrets (no ``.env`` API-key exfiltration). Paths are resolved and
    must live under ``root`` (the current working directory by default; the caller
    binds it to the agent's workspace so reads and writes share one root).
    """
    base = Path(root).resolve() if root else Path.cwd().resolve()
    p = (base / path).resolve()
    if p != base and base not in p.parents:
        return f"error: path outside workspace: {path}"
    if _is_secret_file(p):
        return f"error: refusing to read secret file: {path}"
    if not p.is_file():
        return f"error: no such file: {path}"
    body = p.read_text()
    if start_line is None and end_line is None:
        return body
    lines = body.splitlines(keepends=True)
    total = len(lines)
    first = start_line if start_line is not None else 1
    # Judge the range the model actually asked for before clamping it against the
    # file's length — deriving `last` from `total` first turns a valid start_line=1
    # on an empty file into a bogus "1 <= start_line <= end_line" complaint.
    if first < 1 or (end_line is not None and end_line < first):
        return "error: read_file line range must satisfy 1 <= start_line <= end_line"
    if not total:
        return f"[{path}: empty file]"
    if first > total:
        return f"error: start_line {first} is past end of file ({total} lines)"
    last = end_line if end_line is not None else min(total, first + 199)
    selected = "".join(lines[first - 1 : last])
    shown_last = min(last, total)
    marker = f"[{path}: lines {first}-{shown_last} of {total}]"
    if shown_last < total:
        marker += (
            f"\n[continue with read_file(path={path!r}, "
            f"start_line={shown_last + 1}, end_line={min(total, shown_last + 200)})]"
        )
    return f"{marker}\n{selected}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]
    # v0.2: static, consumer-defined metadata seeded into every ToolCall's
    # ``attributes`` bag (e.g. a tier, a category). carbon never reads it — the
    # values and their meaning are the consumer's (dev-notes/adr/0002).
    attributes: dict = field(default_factory=dict)
    # v0.2: this tool's own result budget. ``None`` uses the global door clamp
    # (CONFIG.max_item_chars); set it to truncate a chatty tool at its own size.
    max_result_chars: int | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """The registered tool by name, or ``None`` — a supported lookup so a
        consumer never has to reach into the private ``_tools`` dict."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """The registered tool names, in registration order."""
        return list(self._tools)

    def wrap(self, name: str, wrapper: Callable[[Callable[..., str]], Callable[..., str]]) -> None:
        """Replace tool ``name`` in place with ``wrapper(original_func)``, keeping
        its description and schema. The generic mechanism behind fault injection,
        logging, caching, and permission middleware — what the wrapper does is the
        consumer's (dev-notes/adr/0002). Raises ``KeyError`` if the tool is absent."""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"no tool {name!r} to wrap")
        self._tools[name] = replace(tool, func=wrapper(tool.func))

    def specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"error: could not parse arguments {arguments!r}"
        problem = _validate_arguments(args, tool.parameters)
        if problem:
            return f"error: invalid arguments for {name}: {problem}"
        try:
            return str(tool.func(**args))
        except Exception as exc:  # noqa: BLE001 — tool errors are fed back to the model
            return f"error: {exc}"

    def __len__(self) -> int:
        return len(self._tools)


def _matches_type(value: object, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _validate_arguments(arguments: object, schema: dict) -> str | None:
    """Small JSON-schema door for the subset Carbon tool contracts use."""
    if not isinstance(arguments, dict):
        return "top-level arguments must be an object"
    required = schema.get("required") or []
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"missing required fields {missing}"
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return f"unknown fields {unknown}"
    for name, value in arguments.items():
        rule = properties.get(name)
        if not isinstance(rule, dict):
            continue
        expected = rule.get("type")
        if isinstance(expected, str) and not _matches_type(value, expected):
            return f"field {name!r} must be {expected}"
        if "enum" in rule and value not in rule["enum"]:
            return f"field {name!r} must be one of {rule['enum']!r}"
        if expected == "array" and isinstance(value, list) and isinstance(rule.get("items"), dict):
            item_type = rule["items"].get("type")
            if isinstance(item_type, str) and any(
                not _matches_type(item, item_type) for item in value
            ):
                return f"every item in field {name!r} must be {item_type}"
    return None


def read_file_tool(root: str | Path | None = None) -> Tool:
    """A ``read_file`` tool confined to ``root`` (defaults to the process cwd).

    The mature agent binds this to its workspace so the model reads the same tree
    it writes to — and never the host cwd (where ``.env`` lives)."""

    def _read(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        return read_file(path, root=root, start_line=start_line, end_line=end_line)

    return Tool(
        name="read_file",
        description=(
            "Read a UTF-8 workspace file. For large files, use 1-based start_line/end_line "
            "ranges and follow the continuation hint instead of repeatedly requesting the "
            "whole file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        func=_read,
    )


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression like '47 * 89'.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    reg.register(read_file_tool())
    return reg
