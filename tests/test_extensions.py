"""Extension loader (dev-notes/adr/0003): discovery, tools-only registration
through the existing ToolRegistry seam, per-extension failure isolation, and
proof that none of it touches the editable config surface."""

from __future__ import annotations

import ast
from pathlib import Path

from harness.agent import _extension_dirs, run_once
from harness.extensions import discover_extensions, load_extensions
from harness.tools import Tool, ToolRegistry
from model import LLMResponse, Provider


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
        "def setup(registry):\n    registry.wrap('echo', lambda f: lambda **kw: f(**kw).upper())\n",
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
    path = Path(__file__).resolve().parents[1] / "harness" / "extensions.py"
    tree = ast.parse(path.read_text())
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any("harness_config" in m for m in imported_modules)
    # Also catch `from harness import harness_config` (module-name check above only
    # sees "harness") and any re-exported `CONFIG` import (e.g. `from harness.agent
    # import CONFIG`), the predecessor string check's equivalent.
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not any("harness_config" in n for n in imported_names)
    assert "CONFIG" not in imported_names


def test_surface_manifest_has_no_extension_field():
    from harness.harness_config import surface_manifest

    manifest = surface_manifest()
    names = {item["name"] for item in manifest["editable"] + manifest["locked_fields"]}
    # Frozen set: the config surface's full field list as of this feature's landing.
    # Any addition or removal must be a deliberate edit to this test — that's the
    # point (dev-notes/adr/0003): no extension-related field should ever appear here
    # silently.
    assert names == {
        "version",
        "system_prompt",
        "max_tool_steps",
        "default_context_limit",
        "approval_tools",
        "code_extensions",
        "verify_attempts",
        "require_run",
        "max_item_chars",
        "file_injection",
        "tool_output",
        "compaction",
        "retry",
        "compaction_prompt",
        "memory_search_limit",
        "attach_pattern",
        "temperature",
        "max_tokens",
    }


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
        extensions=True,
    )

    assert "LOUD" in out


def test_run_once_does_not_load_extensions_by_default(tmp_path, monkeypatch):
    # Off-by-default (final review): the project-local extensions dir sits inside
    # the agent's own writable workspace, so loading it unconditionally would let
    # write_file plant a file that auto-runs, unapproved, on every future call.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    ext_dir = tmp_path / ".carbon" / "extensions"
    _write(ext_dir, "shout.py", _REGISTER_TOOL.format(name="shout", value="LOUD"))

    out = run_once(
        "shout something",
        provider=_tool_then_text("shout", "{}", "done"),
        fmt="transcript",
        session="ext-off-by-default-test",
        sessions_dir=str(tmp_path / "sessions"),
        workspace_root=str(tmp_path),
        agents_dir=str(tmp_path),
    )

    # The tool was never registered, so the call comes back as a tool error in
    # the transcript rather than raising — run_once must not crash doing this.
    assert "LOUD" not in out
    assert "error: unknown tool 'shout'" in out


def test_carbon_package_exports_the_extension_loader():
    import carbon

    assert carbon.load_extensions is load_extensions
    assert carbon.discover_extensions is discover_extensions
