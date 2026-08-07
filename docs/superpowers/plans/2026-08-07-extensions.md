# Extension Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tools-only extension loader (`harness/extensions.py`) that discovers `*.py` files under a user directory and a project directory, calls each one's `setup(registry: ToolRegistry) -> None`, and wires it into the CLI — while keeping it completely outside `harness_config.json`/`surface_manifest()`, per [dev-notes/adr/0003](../../../dev-notes/adr/0003-extensions-tools-only-outside-the-editable-surface.md).

**Architecture:** One new module, `harness/extensions.py`, with two public functions (`discover_extensions`, `load_extensions`) that operate on an existing `ToolRegistry` — no new `Agent` parameter, no new config field. `harness/agent.py`'s CLI entrypoints (`run_once`, `_run_repl`) call `load_extensions` on a small pure helper's output (`_extension_dirs`) right after building the `ToolRegistry` and right before constructing `Agent`. One example extension ships at `extensions/audit_log.py`, exercising both `register()` and `wrap()`.

**Tech Stack:** Python 3.11, stdlib only (`importlib.util`, `pathlib`), pytest, ruff, mypy — matches the rest of the repo; no new dependency.

## Global Constraints

- `uv run verify` (ruff format + lint, mypy, pytest, smoke import) must stay green after every task — it is the floor per `AGENTS.md`.
- Extensions must never be reachable through `harness/harness_config.json`, `config_schema()`, or `surface_manifest()`. No task in this plan may add an extensions-related field to `HarnessConfig` or the config schema.
- No new `Agent.__init__` parameter. Extension loading happens at the CLI/consumer construction site, against the `ToolRegistry` `Agent` already accepts via `tools=`.
- Commit scope for every commit in this plan is `feat(sdk):` (ADR 0001's convention: grows the API external consumers import, not the loop's editable surface).
- Follow existing patterns: `harness/extensions.py` mirrors the shape of `harness/skills.py` (small dataclass-free module, a `load_*` function that tolerates a missing directory).
- Ruff line length is 100 (`pyproject.toml`); match existing docstring/comment style (see `harness/tools.py`, `harness/skills.py`) — plain, explains *why* not *what*.

---

### Task 1: Core extension loader

**Files:**
- Create: `harness/extensions.py`
- Test: `tests/test_extensions.py`

**Interfaces:**
- Produces: `discover_extensions(directory: str | Path) -> list[Path]`; `load_extensions(registry: ToolRegistry, *directories: str | Path) -> list[str]` (returns the loaded extensions' file stems, in load order). Both live in `harness/extensions.py`. `load_extensions` prints one line to `stderr` per skipped extension (no `setup`, or `setup` raised) and never raises itself.
- Consumes: `harness.tools.ToolRegistry`, `.register()`, `.wrap()`, `.call()`, `.names()` — all already public (`harness/tools.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extensions.py`:

```python
"""Extension loader (dev-notes/adr/0003): discovery, tools-only registration
through the existing ToolRegistry seam, per-extension failure isolation, and
proof that none of it touches the editable config surface."""

from __future__ import annotations

from pathlib import Path

from harness.extensions import discover_extensions, load_extensions
from harness.tools import Tool, ToolRegistry


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(body)
    return p


_REGISTER_TOOL = (
    "from harness.tools import Tool\n\n"
    "def setup(registry):\n"
    "    registry.register(Tool(\n"
    "        name={name!r}, description='d',\n"
    "        parameters={{'type': 'object', 'properties': {{}}, 'required': []}},\n"
    "        func=lambda: {value!r},\n"
    "        mutates=False,\n"
    "    ))\n"
)


def test_discover_extensions_finds_py_files(tmp_path):
    _write(tmp_path, "a.py", "")
    _write(tmp_path, "b.py", "")
    _write(tmp_path, "not_python.txt", "")
    assert [p.name for p in discover_extensions(tmp_path)] == ["a.py", "b.py"]


def test_discover_extensions_missing_dir_returns_empty():
    assert discover_extensions("/no/such/dir") == []


def test_load_extensions_registers_a_new_tool(tmp_path):
    _write(tmp_path, "hello.py", _REGISTER_TOOL.format(name="hello", value="hi"))
    registry = ToolRegistry()
    loaded = load_extensions(registry, tmp_path)
    assert loaded == ["hello"]
    assert registry.call("hello", "{}") == "hi"


def test_load_extensions_wrap_changes_existing_tool_behavior(tmp_path):
    _write(
        tmp_path,
        "shout.py",
        "def setup(registry):\n"
        "    registry.wrap('echo', lambda f: lambda **kw: f(**kw).upper())\n",
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            func=lambda text: text,
            mutates=False,
        )
    )
    load_extensions(registry, tmp_path)
    assert registry.call("echo", '{"text": "hi"}') == "HI"


def test_a_broken_extension_does_not_block_the_others(tmp_path, capsys):
    _write(tmp_path, "broken.py", "def setup(registry):\n    raise RuntimeError('boom')\n")
    _write(tmp_path, "good.py", _REGISTER_TOOL.format(name="good", value="ok"))
    registry = ToolRegistry()
    loaded = load_extensions(registry, tmp_path)
    assert loaded == ["good"]
    assert registry.call("good", "{}") == "ok"
    assert "boom" in capsys.readouterr().err


def test_a_file_with_no_setup_is_skipped(tmp_path, capsys):
    _write(tmp_path, "not_an_extension.py", "x = 1\n")
    registry = ToolRegistry()
    assert load_extensions(registry, tmp_path) == []
    assert "no setup" in capsys.readouterr().err


def test_later_directory_overrides_earlier_same_named_tool(tmp_path):
    user_dir, project_dir = tmp_path / "user", tmp_path / "project"
    _write(user_dir, "greet.py", _REGISTER_TOOL.format(name="greet", value="user"))
    _write(project_dir, "greet.py", _REGISTER_TOOL.format(name="greet", value="project"))
    registry = ToolRegistry()
    load_extensions(registry, user_dir, project_dir)
    assert registry.call("greet", "{}") == "project"


def test_extensions_module_never_imports_harness_config():
    source = Path("harness/extensions.py").read_text()
    assert "harness_config" not in source
    assert "CONFIG" not in source


def test_surface_manifest_has_no_extension_field():
    from harness.harness_config import surface_manifest

    manifest = surface_manifest()
    names = {item["name"] for item in manifest["editable"] + manifest["locked_fields"]}
    assert not any("extension" in n for n in names)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extensions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.extensions'` (or `ImportError`) on collection.

- [ ] **Step 3: Write the implementation**

Create `harness/extensions.py`:

```python
"""Extensions — load tools and tool middleware from outside the source tree.

An extension is a Python file exposing ``def setup(registry: ToolRegistry) ->
None``. ``setup`` calls the ToolRegistry methods that already exist and are
already public — ``register()`` for a new tool, ``wrap()`` to layer behavior
onto an existing one (logging, caching, fault injection) — so nothing new is
added to the tool-call path itself (dev-notes/adr/0003). Extension
directories are always given explicitly by the caller (the CLI's ``main()``,
or an SDK consumer building its own ``ToolRegistry``); nothing here reads
``harness/harness_config.json`` or ``CONFIG``, and nothing about extensions
is reachable from ``surface_manifest()`` — there is no field for the
self-improving loop to discover, create, or point somewhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from harness.tools import ToolRegistry

ENTRYPOINT = "setup"


def discover_extensions(directory: str | Path) -> list[Path]:
    """Every ``*.py`` file directly under ``directory``, sorted for a stable
    load order. A missing directory yields an empty list (same handling as
    ``load_skills`` for a missing ``skills/`` dir) — extensions are opt-in,
    not a required project layout."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.py"))


def _import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"carbon_extension_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_extensions(registry: ToolRegistry, *directories: str | Path) -> list[str]:
    """Discover and run every extension under each of ``directories``, in the
    order given — a later directory's extension overrides an earlier one's
    same-named tool, since both call the same ``registry.register()``.

    A single broken extension (a file with no ``setup``, or whose ``setup``
    raises) is skipped and reported to stderr; it does not stop the rest of
    that directory, or any later directory, from loading (matches how Tau and
    Pi both isolate one broken extension from the others). Returns the file
    stems that loaded successfully, in load order."""
    loaded: list[str] = []
    for directory in directories:
        for path in discover_extensions(directory):
            try:
                module = _import_module(path)
                entry = getattr(module, ENTRYPOINT, None)
                if entry is None:
                    print(f"extension {path}: no {ENTRYPOINT}() — skipped", file=sys.stderr)
                    continue
                entry(registry)
            except Exception as exc:  # noqa: BLE001 — one broken extension must not block the rest
                print(f"extension {path}: {exc} — skipped", file=sys.stderr)
                continue
            loaded.append(path.stem)
    return loaded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extensions.py -v`
Expected: PASS — all 8 tests green.

- [ ] **Step 5: Run the full verify gate**

Run: `uv run verify`
Expected: PASS (ruff format + lint, mypy, pytest, smoke import all green).

- [ ] **Step 6: Commit**

```bash
git add harness/extensions.py tests/test_extensions.py
git commit -m "feat(sdk): add a tools-only extension loader (harness/extensions.py)"
```

---

### Task 2: Shipped example extension — `audit_log`

**Files:**
- Create: `extensions/audit_log.py`
- Test: `tests/test_audit_log_extension.py`

**Interfaces:**
- Consumes: `harness.extensions.load_extensions` (Task 1), `harness.tools.{Tool, ToolRegistry}`.
- Produces: nothing new consumed by later tasks — this is the standalone example, analogous to `skills/sign-off/SKILL.md` being the one shipped example skill.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_log_extension.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audit_log_extension.py -v`
Expected: FAIL — `assert "audit_log" in loaded` fails (empty list; `extensions/audit_log.py` doesn't exist yet, so nothing loads and nothing is logged).

- [ ] **Step 3: Write the implementation**

Create `extensions/audit_log.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_audit_log_extension.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full verify gate**

Run: `uv run verify`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add extensions/audit_log.py tests/test_audit_log_extension.py
git commit -m "feat(sdk): ship audit_log, the example extension"
```

---

### Task 3: Wire the loader into the CLI

**Files:**
- Modify: `harness/agent.py` (add `_extension_dirs`; call it from `run_once` and `_run_repl`)
- Test: `tests/test_extensions.py` (append)

**Interfaces:**
- Consumes: `harness.extensions.load_extensions` (Task 1).
- Produces: `_extension_dirs(workspace_root: str | Path) -> tuple[Path, Path]` in `harness/agent.py` — `(user_dir, project_dir)`, `user_dir = Path.home() / ".carbon" / "extensions"`, `project_dir = Path(workspace_root) / ".carbon" / "extensions"`.

- [ ] **Step 1: Write the failing tests**

First, add two import lines to the top of `tests/test_extensions.py`, alongside the existing imports (ruff's `E402` requires module-level imports stay at the top of the file, so these cannot be added next to the new tests further down):

```python
from harness.agent import _extension_dirs, run_once
from harness.extensions import discover_extensions, load_extensions
from harness.tools import Tool, ToolRegistry
from model import LLMResponse, Provider
```

(This replaces the existing `from harness.extensions import discover_extensions, load_extensions` / `from harness.tools import Tool, ToolRegistry` pair with all four import lines, alphabetically ordered by module.)

Then append the new tests at the bottom of the file:

```python
# --- CLI wiring ----------------------------------------------------------


def test_extension_dirs_are_user_then_project(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    user_dir, project_dir = _extension_dirs(tmp_path / "project")
    assert user_dir == tmp_path / "home" / ".carbon" / "extensions"
    assert project_dir == tmp_path / "project" / ".carbon" / "extensions"


def _tool_then_text(name: str, args_json: str, text: str) -> Provider:
    responses = iter(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "function": {"name": name, "arguments": args_json}}],
            ),
            LLMResponse(content=text, finish_reason="stop"),
        ]
    )

    def responder(messages, **kwargs):
        return next(responses)

    return Provider(base_url="fake://x", model="fake", api_key="x", responder=responder)


def test_run_once_loads_project_local_extensions(tmp_path, monkeypatch):
    # Keep the real ~/.carbon out of this test regardless of the host machine.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    ext_dir = tmp_path / ".carbon" / "extensions"
    _write(ext_dir, "shout.py", _REGISTER_TOOL.format(name="shout", value="LOUD"))

    out = run_once(
        "shout something",
        provider=_tool_then_text("shout", "{}", "done"),
        fmt="transcript",
        session="ext-wiring-test",
        sessions_dir=str(tmp_path / "sessions"),
        workspace_root=str(tmp_path),
        agents_dir=str(tmp_path),
    )

    assert "LOUD" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extensions.py -v -k "extension_dirs or loads_project_local"`
Expected: FAIL — `ImportError: cannot import name '_extension_dirs' from 'harness.agent'`.

- [ ] **Step 3: Write the implementation**

In `harness/agent.py`, add `_extension_dirs` near `_coding_tools` (just above its definition, around line 628):

```python
def _extension_dirs(workspace_root: str | Path) -> tuple[Path, Path]:
    """Where extensions load from: a user-level directory, then a project-local
    one — always explicit paths at the construction site, never a config-file
    field (dev-notes/adr/0003)."""
    return Path.home() / ".carbon" / "extensions", Path(workspace_root) / ".carbon" / "extensions"
```

In `run_once` (around lines 703-728), replace:

```python
    from harness.render import render_json, render_plain, render_transcript
    from harness.skills import load_skills
    from harness.workspace import Workspace

    provider = provider or Provider.from_env()
    workspace = Workspace(root=workspace_root)
    tracer = Tracer(model=provider.model)
    agent = Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=_coding_tools(
            workspace,
            exclude_session=session,
            provider=provider,
            model=provider.model,
            sessions_dir=sessions_dir,
        ),
        approve=_approver(yes),
        approval_required=APPROVAL_TOOLS,
        skills=load_skills("skills"),
        session=session,
        sessions_dir=sessions_dir,
        agents_dir=agents_dir or str(workspace.root),
        tracer=tracer,
    )
```

with:

```python
    from harness.extensions import load_extensions
    from harness.render import render_json, render_plain, render_transcript
    from harness.skills import load_skills
    from harness.workspace import Workspace

    provider = provider or Provider.from_env()
    workspace = Workspace(root=workspace_root)
    tracer = Tracer(model=provider.model)
    tools = _coding_tools(
        workspace,
        exclude_session=session,
        provider=provider,
        model=provider.model,
        sessions_dir=sessions_dir,
    )
    load_extensions(tools, *_extension_dirs(workspace.root))
    agent = Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        approve=_approver(yes),
        approval_required=APPROVAL_TOOLS,
        skills=load_skills("skills"),
        session=session,
        sessions_dir=sessions_dir,
        agents_dir=agents_dir or str(workspace.root),
        tracer=tracer,
    )
```

In `_run_repl` (around lines 797-835), replace:

```python
    from harness.orchestrator import Orchestrator
    from harness.skills import load_skills
    from harness.workspace import Workspace, git_worktree
```

with:

```python
    from harness.extensions import load_extensions
    from harness.orchestrator import Orchestrator
    from harness.skills import load_skills
    from harness.workspace import Workspace, git_worktree
```

and replace:

```python
    tools = _coding_tools(
        workspace, exclude_session="repl", provider=provider, model=provider.model
    )
    agent = Agent(
```

with:

```python
    tools = _coding_tools(
        workspace, exclude_session="repl", provider=provider, model=provider.model
    )
    load_extensions(tools, *_extension_dirs(workspace.root))
    agent = Agent(
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extensions.py -v`
Expected: PASS — all tests in the file, including the two appended in this task.

- [ ] **Step 5: Run the full test suite and verify gate**

Run: `uv run verify`
Expected: PASS. Pay attention to `tests/test_print_mode.py` and `tests/test_streaming.py` (both call `run_once`) — they must still pass unchanged, since `_extension_dirs`'s project directory (a `.carbon/extensions` that doesn't exist in their `tmp_path`) resolves to an empty `discover_extensions` list, a no-op.

- [ ] **Step 6: Commit**

```bash
git add harness/agent.py tests/test_extensions.py
git commit -m "feat(sdk): load extensions in the CLI before constructing Agent"
```

---

### Task 4: Export the loader from the `carbon` SDK package

**Files:**
- Modify: `carbon/__init__.py`
- Test: `tests/test_extensions.py` (append)

**Interfaces:**
- Consumes: `harness.extensions.{discover_extensions, load_extensions}` (Task 1).
- Produces: `carbon.discover_extensions`, `carbon.load_extensions` — an SDK consumer builds its own `ToolRegistry` and loads extensions onto it without reaching into `harness.*` internal paths, the same relationship `carbon.default_tools`/`carbon.ToolRegistry` already have.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extensions.py`:

```python
def test_carbon_package_exports_the_extension_loader():
    import carbon

    assert carbon.load_extensions is load_extensions
    assert carbon.discover_extensions is discover_extensions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extensions.py -v -k carbon_package_exports`
Expected: FAIL — `AttributeError: module 'carbon' has no attribute 'load_extensions'`.

- [ ] **Step 3: Write the implementation**

In `carbon/__init__.py`, add the import (alphabetical among the `harness.*` imports, after the `harness.agent` import):

```python
from harness.agent import Agent, run_once
from harness.extensions import discover_extensions, load_extensions
from harness.harness_config import (
```

Add both names to `__all__` (alphabetical order, matching the existing list):

```python
    "config_schema",
    "default_tools",
    "discover_extensions",
    "fake",
    "load_config",
    "load_env",
    "load_extensions",
    "provenance",
```

Update the module docstring's summary sentence to mention extensions:

```python
"""carbon — the public SDK surface (v0.1, the embedding seam).

The curriculum builds the harness one primitive per chapter across ``model/``,
``harness/``, and ``ui/``. This package is the curated, versioned surface external
code builds on: ``import carbon`` gives you exactly the agent, its structured
result, tools and their registry, an extension loader, the permission policy, the
provider and model seam, the editable-config door and its schema, and provenance
— and nothing else.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_extensions.py -v`
Expected: PASS — every test in the file.

- [ ] **Step 5: Run the full verify gate**

Run: `uv run verify`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add carbon/__init__.py tests/test_extensions.py
git commit -m "feat(sdk): export the extension loader from the carbon package"
```

---

### Task 5: Docs — README and CHANGELOG

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: nothing (docs only). References `dev-notes/adr/0003-extensions-tools-only-outside-the-editable-surface.md` (already on disk from brainstorming).

- [ ] **Step 1: Update README.md's "Beyond the curriculum: a living surface" section**

In `README.md`, find this paragraph (currently ending the section, right before "## How it is built"):

```
Two rules govern every addition: generic mechanism lives in the harness while
domain and policy live in the consumer, and no knob becomes editable until an
external miner and guard can distinguish its choices. The reasoning behind these
decisions lives in [dev-notes/](dev-notes/).
```

Insert a new paragraph directly after it (still before `## How it is built`):

```

The library surface also grows through extensions: a Python file under
`~/.carbon/extensions/` or a project's `.carbon/extensions/` exposing
`setup(registry: ToolRegistry) -> None`, which registers new tools or wraps
existing ones through the same `ToolRegistry` seam every consumer already
uses (`harness/extensions.py`; see
[dev-notes/adr/0003](dev-notes/adr/0003-extensions-tools-only-outside-the-editable-surface.md)).
Extensions load new code, so they sit outside `surface_manifest()` on
purpose — the self-improving loop can pick from the config door's bounded
menus, but it cannot point Carbon at an extensions directory, because no
such field exists for it to find.
```

- [ ] **Step 2: Update CHANGELOG.md**

In `CHANGELOG.md`, under the existing `## [Unreleased]` heading, add a new `### Added` subsection above the existing `### Changed` subsection:

```markdown
## [Unreleased]

### Added

- `harness/extensions.py`: a tools-only extension loader. A Python file under
  `~/.carbon/extensions/` or a project's `.carbon/extensions/` exposing
  `setup(registry: ToolRegistry) -> None` can register new tools or `wrap()`
  existing ones, using the same `ToolRegistry` seam consumers already use —
  no new hook system, no `Agent` change. Deliberately outside
  `surface_manifest()`: refinery cannot discover, create, or point at an
  extensions directory (dev-notes/adr/0003). Ships one example,
  `extensions/audit_log.py`. Exported from `carbon` as `load_extensions` /
  `discover_extensions`.

### Changed
```

(Leave the existing `### Changed` bullets under it exactly as they are — only the heading placement and the new `### Added` block above it change.)

- [ ] **Step 3: Verify the docs changes read correctly**

Run: `git diff README.md CHANGELOG.md`
Expected: the two insertions above, nothing else changed. Read both rendered sections back to confirm no broken Markdown (unclosed link brackets, etc.).

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: describe the extension loader in README and CHANGELOG"
```

---

### Task 6: Final verification gate

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Run the full verify gate one more time from a clean state**

Run: `uv run verify`
Expected: PASS — ruff format + lint, mypy, pytest (including every test added in Tasks 1-4), and the smoke import all green.

- [ ] **Step 2: Confirm the invariant by hand**

Run:
```bash
grep -rn "extension" harness/harness_config.py harness/harness_config.json
```
Expected: no output (or only incidental matches inside unrelated words — inspect any hit by hand). This is the same guarantee `test_extensions_module_never_imports_harness_config` and `test_surface_manifest_has_no_extension_field` check in code (Task 1); this step is the human/agent sanity check on top.

- [ ] **Step 3: Confirm git status is clean**

Run: `git status`
Expected: clean working tree — every change from Tasks 1-5 committed.

- [ ] **Step 4: No commit needed for this task** (verification only; nothing to add).
