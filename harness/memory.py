"""Durable state (ch-09) + episodic retrieval (ch-16).

Conversation is not state. The harness persists the session as JSON-L (one
message per line, easy to append) so a killed agent can resume by reloading it.
The session boundary is the kill point — nothing survives unless it's written.

ch-16: a log isn't *memory* until you can recover the right slice. ``search_sessions``
does keyword text search across all stored sessions (no embeddings); ``search_memory_tool``
lets the model pull matching chunks from sessions that aren't in the current context.
"""

from __future__ import annotations

import json
import os
import string
import tempfile
from pathlib import Path

from harness.harness_config import CONFIG
from harness.tools import Tool

DEFAULT_DIR = ".sessions"


def _path(session_id: str, base: str | Path = DEFAULT_DIR) -> Path:
    # Session ids come from argv — take only the final path component so a name
    # like "../secret" can't write (or, via /reset, unlink) files outside `base`.
    safe = Path(session_id).name or "session"
    return Path(base) / f"{safe}.jsonl"


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Write rows to ``path`` atomically (temp file + rename in the same dir).

    A kill mid-write must not shred the previous good file — durable state is the
    whole point, and ``save_*`` runs after every turn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)  # atomic on POSIX; the reader never sees a partial file
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSON-L file, skipping any unparseable line.

    A killed agent can leave a half-written final line; one bad line must not
    make the whole session unrecoverable (it would crash resume in the REPL).
    """
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def save_session(session_id: str, messages: list[dict], base: str | Path = DEFAULT_DIR) -> None:
    _write_jsonl_atomic(_path(session_id, base), messages)


def load_session(session_id: str, base: str | Path = DEFAULT_DIR) -> list[dict]:
    path = _path(session_id, base)
    if not path.is_file():
        return []
    return _read_jsonl(path)


def _trace_path(session_id: str, base: str | Path = DEFAULT_DIR) -> Path:
    # A subdir so trace files are not picked up by the *.jsonl session globs.
    return Path(base) / "traces" / f"{session_id}.jsonl"


def save_trace(session_id: str, rows: list[dict], base: str | Path = DEFAULT_DIR) -> None:
    """Persist a session's trace events (ch-24) next to its messages, so the trace
    pane survives a restart instead of resetting to empty."""
    _write_jsonl_atomic(_trace_path(session_id, base), rows)


def load_trace(session_id: str, base: str | Path = DEFAULT_DIR) -> list[dict]:
    path = _trace_path(session_id, base)
    if not path.is_file():
        return []
    return _read_jsonl(path)


def delete_session(session_id: str, base: str | Path = DEFAULT_DIR) -> None:
    """Wipe a session's persisted messages and trace (ch-24 ``/reset``). Idempotent —
    missing files are fine, since reset should work whether or not anything was saved."""
    for path in (_path(session_id, base), _trace_path(session_id, base)):
        path.unlink(missing_ok=True)


def list_sessions(base: str | Path = DEFAULT_DIR) -> list[dict]:
    """List persisted sessions for the UI (ch-24): name, message count, mtime.

    Sorted most-recently-modified first so the active/recent ones surface at the
    top of the sessions pane. Reuses the same JSON-L files as load/save."""
    base_dir = Path(base)
    if not base_dir.is_dir():
        return []
    out: list[dict] = []
    for path in base_dir.glob("*.jsonl"):
        try:
            messages = sum(1 for line in path.read_text().splitlines() if line.strip())
        except OSError:
            continue
        out.append({"name": path.stem, "messages": messages, "mtime": path.stat().st_mtime})
    out.sort(key=lambda s: s["mtime"], reverse=True)
    return out


def search_sessions(
    query: str,
    base: str | Path = DEFAULT_DIR,
    limit: int = CONFIG.memory_search_limit,
    *,
    exclude: str | None = None,
) -> list[dict]:
    """Keyword text search across stored sessions. Returns the best-matching messages
    as {session, role, content}, ranked by how many query terms appear.

    ``exclude`` drops one session (the current one) so recall surfaces facts that
    *aren't* already in the live context. Query terms are stripped of surrounding
    punctuation so a natural phrasing like ``warehouse passcode?`` still matches
    ``passcode``."""
    terms = [t.strip(string.punctuation) for t in query.lower().split()]
    terms = [t for t in terms if t]
    if not terms:
        return []
    base_dir = Path(base)
    if not base_dir.is_dir():
        return []
    scored: list[tuple[int, dict]] = []
    for path in sorted(base_dir.glob("*.jsonl")):
        if exclude is not None and path.stem == exclude:
            continue
        for msg in _read_jsonl(path):
            content = str(msg.get("content", "") or "").lower()
            score = sum(term in content for term in terms)
            if score:
                scored.append(
                    (
                        score,
                        {
                            "session": path.stem,
                            "role": msg.get("role"),
                            "content": msg.get("content"),
                        },
                    )
                )
    scored.sort(key=lambda s: s[0], reverse=True)
    return [m for _, m in scored[:limit]]


def search_memory_tool(base: str | Path = DEFAULT_DIR, *, exclude: str | None = None) -> Tool:
    """A tool the model calls to recall facts from earlier sessions.

    ``exclude`` is the current session id — its lines are already in context, so
    recall should look at the *other* sessions (the module's stated purpose)."""

    def search_memory(query: str) -> str:
        hits = search_sessions(query, base=base, exclude=exclude)
        if not hits:
            return "no matching memory found"
        return "\n".join(f"[{h['session']}] {h['role']}: {h['content']}" for h in hits)

    return Tool(
        name="search_memory",
        description="Search past sessions for relevant facts by keyword.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=search_memory,
        mutates=False,
    )
