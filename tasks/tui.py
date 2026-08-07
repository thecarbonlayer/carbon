"""Launch the Textual TUI (ch-14).

    uv run tui                                    # resume the default 'cli' session
    uv run tui chat-1751130000                    # resume a session by name
    uv run tui .sessions/chat-1751130000.jsonl    # ...or by file path
    uv run tui --extensions                       # also load .carbon/extensions/

There is no in-app session switcher (the UI is one agent). The session argument is
how you reopen a specific run — including the ones a `/new` command created.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_session(arg: str | None) -> tuple[str, str | None]:
    """Map a CLI argument to ``(session_id, sessions_dir)``.

    A bare name (``cli``, ``chat-123``) keeps the default sessions dir. A path to a
    session file (``.sessions/chat-123.jsonl``) resolves to its stem + parent dir, so
    a session saved anywhere can be reopened. ``None`` → the default ``cli`` session.
    """
    if not arg:
        return "cli", None
    if arg.endswith(".jsonl") or "/" in arg or os.sep in arg:
        p = Path(arg)
        return p.stem, str(p.parent)
    return arg, None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extensions = "--extensions" in argv
    argv = [a for a in argv if a != "--extensions"]

    from harness.workspace import Workspace, git_worktree
    from ui.tui import AgentTUI

    session, sessions_dir = _resolve_session(argv[0] if argv else None)
    kwargs: dict = {"session": session, "extensions": extensions}
    if sessions_dir is not None:
        kwargs["sessions_dir"] = sessions_dir

    # Work in a git worktree of this repo (your checkout stays pristine), or a
    # scratch dir if we're not in a git repo — the coding-agent posture, in the UI.
    wt = git_worktree(".")
    if wt is not None:
        workspace, cleanup = wt
    else:
        workspace, cleanup = Workspace(), (lambda: None)
    try:
        AgentTUI(workspace=workspace, **kwargs).run()
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
