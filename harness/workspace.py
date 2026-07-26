"""The workspace — a directory the agent owns.

A small, path-safe wrapper around a directory: write, read, and edit files whose
contents survive across calls. Every path is confined to the workspace root, so a
bad write can't escape to the host. By default it's a fresh scratch dir, so an
experiment can't touch your real project; point ``root`` at a real repo only if
you mean it.

At ch-03 this was just the class — the accept staged an ``AGENTS.md`` here to
prove auto-loaded instructions work. Now that the agent has a tool interface
(ch-05), the write/edit *tools* below let the model build a multi-file project:
each call goes through ``Workspace``, so every path is confined to the root and
the files survive across calls for ``bash`` (run in the same dir) to see.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from difflib import unified_diff
from pathlib import Path

from harness.tools import Tool


class Workspace:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = (
            Path(root).resolve() if root else Path(tempfile.mkdtemp(prefix="workspace-")).resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return p

    def write(self, path: str, content: str) -> str:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    def read(self, path: str) -> str:
        p = self._safe(path)
        return p.read_text() if p.is_file() else f"error: no such file: {path}"

    def edit(self, path: str, old: str, new: str) -> str:
        p = self._safe(path)
        if not p.is_file():
            return f"error: no such file: {path}"
        if old == "":
            # An empty `old` is a substring of everything and would silently
            # prepend `new` at offset 0 — reject it rather than corrupt the file.
            return f"error: `old` must be non-empty to edit {path}"
        text = p.read_text()
        matches = text.count(old)
        if matches == 0:
            return f"error: text to replace not found in {path}"
        if matches > 1:
            return (
                f"error: text to replace occurs {matches} times in {path}; "
                "include more surrounding context so the edit is unique"
            )
        updated = text.replace(old, new, 1)
        # Write beside the target and rename over it, so a crash mid-write can never
        # leave a half-file. Two details the naive version gets wrong: the temp name
        # must be unique (delegated workers share one workspace across threads, and a
        # fixed name lets two edits collide mid-write), and the mode must be copied
        # from the original (mkstemp is 0600, and a plain rename would silently strip
        # the executable bit off a script).
        fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".carbon-edit")
        temp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(updated)
            shutil.copymode(p, temp)
            temp.replace(p)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        diff = "".join(
            unified_diff(
                text.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
        return f"edited {path}\n{diff}".rstrip()


def write_file_tool(ws: Workspace) -> Tool:
    return Tool(
        name="write_file",
        description="Create or overwrite a file in the workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        func=ws.write,
    )


def edit_file_tool(ws: Workspace) -> Tool:
    return Tool(
        name="edit_file",
        description=(
            "Atomically replace one unique occurrence of `old` with `new` in a workspace "
            "file. Ambiguous matches fail; include enough surrounding context to identify "
            "one location. Returns the resulting unified diff."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
        func=ws.edit,
    )


def git_worktree(base: str | Path = ".") -> tuple[Workspace, Callable[[], None]] | None:
    """If ``base`` is inside a git repo, create an ephemeral detached worktree of
    HEAD and return ``(Workspace(worktree), cleanup)``. The agent gets the *real*
    codebase — tests run, deps resolve — but every edit lands in a throwaway
    worktree, so your actual checkout is never touched. Returns ``None`` when we're
    not in a git repo (the caller falls back to a scratch dir).

    If the worktree is a uv project, its deps are synced once here (offline, from
    cache) so the declared test command runs fast, not cold, mid-turn.
    """
    base = Path(base).resolve()
    inside = subprocess.run(
        ["git", "-C", str(base), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    root = subprocess.run(
        ["git", "-C", str(base), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    wt = Path(tempfile.mkdtemp(prefix="agent-worktree-"))
    add = subprocess.run(
        ["git", "-C", root, "worktree", "add", "--detach", str(wt), "HEAD"],
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        return None
    if (wt / "pyproject.toml").is_file():
        # Bounded so a cold cache (or a network stall) can't freeze launch before
        # the UI ever appears; a failed/slow sync just means a colder first test run.
        try:
            subprocess.run(["uv", "sync"], cwd=wt, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            pass

    def cleanup() -> None:
        subprocess.run(
            ["git", "-C", root, "worktree", "remove", "--force", str(wt)],
            capture_output=True,
        )

    return Workspace(root=wt), cleanup
