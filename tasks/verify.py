"""Deterministic commit gate: format, lint, types, tests, smoke import.

Runs locally. Hits no network — the live model run is `accept`.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, os.getcwd())
    failed: list[str] = []

    if _run(["ruff", "format", "--check", "."]) != 0:
        failed.append("ruff-format")
    if _run(["ruff", "check", "."]) != 0:
        failed.append("ruff-check")
    if _run(["mypy", "model", "harness", "ui", "tasks", "carbon", "extensions"]) != 0:
        failed.append("mypy")

    rc = _run(["pytest"])
    if rc not in (0, 5):  # 5 = "no tests collected", expected in the earliest chapters
        failed.append("pytest")

    print("\n$ smoke import", flush=True)
    try:
        importlib.import_module("harness.agent")
        importlib.import_module("model")
        importlib.import_module("ui.tui")
        importlib.import_module("tasks.checks")
        importlib.import_module("carbon")
        print("smoke import OK")
    except Exception as exc:  # noqa: BLE001
        print(f"smoke import FAILED: {exc}")
        failed.append("smoke")

    if failed:
        print(f"\nVERIFY FAILED: {failed}")
        return 1
    print("\nVERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
