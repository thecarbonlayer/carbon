"""Tools — the actions the model can ask the harness to run.

A tool is just a function plus a JSON-schema contract. The registry turns those
functions into OpenAI tool specs (so the model knows what it can call) and
dispatches the calls by name, parsing arguments and returning a string result —
or an error string the model can read and recover from.

Tools are an API surface you expose to a model: keep the list small, keep each
contract narrow, and validate arguments. ``read_file`` returns a file's
contents, and ``list_files``/``search_text`` explore the tree — and as of
ch-08 all three are confined to the workspace: a model-invoked tool must not
wander the host filesystem, so paths are resolved and must live under the
working directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path


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
    # v0.4: does this tool change anything outside the conversation? A read-only
    # policy consults it. The default is ``True`` — carbon cannot inspect a
    # consumer's callable to find out, and a permission boundary that guesses
    # "harmless" about code it has never seen is not a boundary. A tool named
    # ``save_report`` is not in ``DEFAULT_MUTATORS``, so name-matching alone
    # would have let it write. Declare ``mutates=False`` to let a read-only
    # agent use it.
    mutates: bool = True


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
        mutates=False,
    )


def _confined(root: str | Path | None) -> Path:
    return Path(root).resolve() if root else Path.cwd().resolve()


def _visible_files(base: Path, pattern: str) -> list[Path]:
    """Files under ``base`` matching ``pattern``, minus secrets and VCS/venv noise.

    Confinement is judged on the *resolved* target, not the apparent path. A
    symlink inside the workspace can name anything on the host — ``innocent.txt``
    pointing at a file outside the root, or at ``.env`` — so checking the link's
    own name would leak the content it points to. ``read_file`` already resolves
    before its containment check; these have to agree with it.
    """
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}
    out = []
    for p in sorted(base.glob(pattern)):
        target = p.resolve()
        if target != base and base not in target.parents:
            continue  # symlink (or traversal) out of the workspace
        if not target.is_file() or _is_secret_file(p) or _is_secret_file(target):
            continue
        if skip & set(p.relative_to(base).parts):
            continue
        out.append(p)
    return out


def list_files(pattern: str = "**/*", root: str | Path | None = None, limit: int = 200) -> str:
    """List workspace files matching a glob — the read-only way to find a path.

    Without this, a read-only agent can only open files whose exact paths it was
    already told, which is not enough to explore a repository.
    """
    base = _confined(root)
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return f"error: pattern must stay inside the workspace: {pattern}"
    files = _visible_files(base, pattern)
    if not files:
        return f"no files match {pattern}"
    shown = [str(p.relative_to(base)) for p in files[:limit]]
    more = f"\n…[{len(files) - limit} more; narrow the pattern]" if len(files) > limit else ""
    return "\n".join(shown) + more


def search_text(
    query: str,
    root: str | Path | None = None,
    pattern: str = "**/*",
    limit: int = 100,
) -> str:
    """Grep the workspace for a literal string, returning ``path:line: text`` hits."""
    base = _confined(root)
    if not query:
        return "error: query must be non-empty"
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return f"error: pattern must stay inside the workspace: {pattern}"
    hits: list[str] = []
    for p in _visible_files(base, pattern):
        try:
            body = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable — skip, don't fail the search
        for i, line in enumerate(body.splitlines(), 1):
            if query in line:
                hits.append(f"{p.relative_to(base)}:{i}: {line.strip()[:200]}")
                if len(hits) > limit:
                    return "\n".join(hits[:limit]) + f"\n…[more than {limit} hits; narrow it]"
    return "\n".join(hits) if hits else f"no matches for {query!r}"


def list_files_tool(root: str | Path | None = None) -> Tool:
    def _list(pattern: str = "**/*") -> str:
        return list_files(pattern, root=root)

    return Tool(
        name="list_files",
        description=(
            "List workspace files matching a glob (default '**/*'). Use this to find "
            "a path before read_file. Secrets and VCS/venv directories are excluded."
        ),
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        func=_list,
        mutates=False,
    )


def search_text_tool(root: str | Path | None = None) -> Tool:
    def _search(query: str, pattern: str = "**/*") -> str:
        return search_text(query, root=root, pattern=pattern)

    return Tool(
        name="search_text",
        description=(
            "Search workspace files for a literal string; returns 'path:line: text' hits. "
            "Optionally restrict to a glob via `pattern`."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "pattern": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        func=_search,
        mutates=False,
    )


def default_tools(root: str | Path | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(read_file_tool(root))
    reg.register(list_files_tool(root))
    reg.register(search_text_tool(root))
    return reg
