"""Extensions — load tools and tool middleware from outside the source tree.

An extension is a Python file exposing ``def setup(registry: ToolRegistry) ->
None``. ``setup`` calls the ToolRegistry methods that already exist and are
already public — ``register()`` for a new tool, ``wrap()`` to layer behavior
onto an existing one (logging, caching, fault injection) — so nothing new is
added to the tool-call path itself (dev-notes/adr/0003). Extension
directories are always given explicitly by the caller (the CLI's ``main()``,
or an SDK consumer building its own ``ToolRegistry``); nothing here reads
``harness/harness_config.json`` or ``CONFIG``, and extensions are never
visible in the editable or locked configuration surface — there is no field
for the self-improving loop to discover, create, or point somewhere.
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
