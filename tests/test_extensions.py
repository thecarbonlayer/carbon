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
    source = Path("harness/extensions.py").read_text()
    assert "harness_config" not in source
    assert "CONFIG" not in source


def test_surface_manifest_has_no_extension_field():
    from harness.harness_config import surface_manifest

    manifest = surface_manifest()
    names = {item["name"] for item in manifest["editable"] + manifest["locked_fields"]}
    assert not any("extension" in n for n in names)
