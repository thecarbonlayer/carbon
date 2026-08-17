"""Session-scoped runtime storage — the harness's private scratch, outside the repo.

The workspace is the user's durable project; everything the HARNESS creates at run
time (offloaded tool output today) is runtime state with a different lifecycle and a
different audience. Holding runtime state inside the workspace made it repo-visible
(git, scanners, sync) and gave it the workspace's unbounded lifetime — the measured
cost was a security task counting a harness cache file as a workspace leak.

The contract, which tests enforce clause by clause:
  - scratch is PRIVATE (0700, unpredictable name via mkdtemp) and OUTSIDE the repo;
  - an EPHEMERAL session's scratch lives exactly as long as the session:
    ``cleanup()`` removes it, callers run that in ``finally`` (success, failure,
    cancellation alike);
  - an abandoned session — one whose ``cleanup()``/``close()`` never ran (a crash,
    a caller that forgot) — leaks its ONE directory until a later ``scavenge()``
    — run at every session start — removes strays older than ``SCAVENGE_AGE_S``.
    This is a backstop for the session that couldn't run its own cleanup, not a
    substitute for running it: N abandoned sessions leak N directories, each
    sitting until scavenge's next pass, not "at most one" for the process;
    ``close()`` is the contract for everything else;
  - a DURABLE session (``local_session_env(..., session=..., sessions_dir=...)``)
    is the one exception to "cleanup() removes it": its scratch lives at a
    deterministic path under ``sessions_dir`` and is tied to the SESSION's
    lifetime, not to whichever process happens to have it open — because a
    persisted transcript can hold ``scratch://`` refs (``offload_to_file``) that
    must still resolve the next time the same session is reopened. Its
    ``cleanup()`` is a no-op; only ``delete_session_scratch()`` (or deleting the
    session itself) ends it. Structurally out of ``scavenge()``'s reach too — it
    lives under ``sessions_dir``, never the OS temp dir ``scavenge()`` sweeps;
  - another session cannot name it (unpredictable component, or — for a durable
    session — the sanitized session id) or read it (0700, sessions_dir included);
  - ``metadata`` states what kind of environment this is and its storage policy, so
    a results manifest can record what the measurement ran on.

Provider-neutral on purpose: a remote/container implementation replaces
``local_session_env``, not the strategies that write into ``scratch_root``.
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRATCH_PREFIX = "carbon-scratch-"
SCAVENGE_AGE_S = 24 * 3600

LOCAL_METADATA = {
    "kind": "local",
    "impl_version": 1,
    "storage_policy": "private scratch, removed at session close; strays scavenged after 24h",
}

DURABLE_METADATA = {
    "kind": "local",
    "impl_version": 1,
    "storage_policy": (
        "durable scratch: NOT removed at session close (cleanup() is a no-op); tied "
        "to the session's lifetime under sessions_dir, removed via "
        "delete_session_scratch() or when the session itself is deleted; never "
        "scavenged (outside the OS temp dir scavenge() sweeps)"
    ),
}


@dataclass(frozen=True)
class SessionEnvironment:
    session_id: str
    workspace_root: Path | None
    scratch_root: Path
    metadata: dict = field(default_factory=lambda: dict(LOCAL_METADATA))
    # A durable scratch (built by local_session_env(..., session=..., sessions_dir=...))
    # is tied to the SESSION's lifetime, not the Agent/process that happens to have it
    # open right now: a persisted transcript can hold scratch:// refs (offload_to_file)
    # that must still resolve the next time the same session is reopened. cleanup()
    # honors that below — only delete_session_scratch() (or deleting the session
    # itself) ends a durable scratch's life.
    durable: bool = False

    def cleanup(self) -> None:
        """Remove the scratch tree. Never raises — this runs in ``finally`` blocks,
        and a cleanup error must not mask the real exception in flight.

        A no-op when ``durable``: that scratch outlives the Agent/process that
        opened it BY DESIGN — it is deleted with the session
        (``delete_session_scratch``), not with whatever happened to call
        ``cleanup()`` this time."""
        if self.durable:
            return
        shutil.rmtree(self.scratch_root, ignore_errors=True)


def scavenge(max_age_s: float = SCAVENGE_AGE_S) -> int:
    """Remove abandoned scratch directories (a crash's leftovers) past their expiry.

    Opportunistic by design: it runs when the next session starts, so a machine that
    never runs carbon again keeps at most what the OS temp reaper would take anyway.
    Only prefixed, real directories are touched; a same-named symlink is ignored.

    A durable session's scratch (``local_session_env(..., session=..., sessions_dir=...)``)
    is structurally out of reach here: it lives under ``sessions_dir``, never under
    the OS temp dir this glob searches, so no ``max_age_s`` can make this function
    remove it — see ``test_scavenge_does_not_touch_durable_scratch``.
    """
    now = time.time()
    removed = 0
    for p in Path(tempfile.gettempdir()).glob(f"{SCRATCH_PREFIX}*"):
        try:
            if p.is_dir() and not p.is_symlink() and now - p.stat().st_mtime > max_age_s:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _safe_session_dirname(session: str) -> str:
    """Sanitize a caller-supplied session id into a single, contained path component.

    Same rule ``harness/memory.py:_path`` already applies to a session id headed for
    a filename: take only the FINAL path component (``Path(session).name``), so a
    value like ``"../../etc/passwd"`` collapses to ``"passwd"`` and can't walk a
    durable scratch dir outside ``sessions_dir``. ``memory.py`` falls back to
    ``"session"`` when that component is empty (e.g. the id was ``".."`` or ``"."``
    or ``"/"``); this mirrors that fallback too, so the two modules never disagree
    about what a given session id sanitizes to.
    """
    return Path(session).name or "session"


def local_session_env(
    workspace_root: str | Path | None = None,
    session_id: str | None = None,
    *,
    session: str | None = None,
    sessions_dir: str | Path | None = None,
) -> SessionEnvironment:
    """A session's runtime scratch. Two lifetimes, chosen by whether ``session`` is given:

    - Ephemeral (default, ``session=None``): a local scratch under the OS temp dir,
      0700 and unpredictable via ``mkdtemp`` (closes the shared-/tmp pre-creation
      attack without a custom parent dir). ``cleanup()`` really removes it.
    - Durable (``session=<id>``, ``sessions_dir=<dir>``): scratch lives at the
      deterministic path ``<sessions_dir>/<safe session>.scratch/`` (0700), so
      reopening the SAME session lands on the SAME path — required for a persisted
      transcript's ``scratch://`` refs (``offload_to_file``, harness/limits.py) to
      still resolve after a restart. ``cleanup()`` is a no-op (``durable=True``);
      remove it explicitly with ``delete_session_scratch``.
    """
    scavenge()
    root = Path(workspace_root).resolve() if workspace_root else None
    if session is not None:
        if sessions_dir is None:
            raise ValueError("sessions_dir is required when session is given")
        # mode= only takes effect at creation time (Path.mkdir does not chmod an
        # already-existing directory), so this hardens the case THIS call creates
        # sessions_dir; one created earlier by something else (e.g. memory.py's
        # save_session, which does not restrict its mode) is left as-is.
        sdir = Path(sessions_dir)
        sdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        scratch = sdir / f"{_safe_session_dirname(session)}.scratch"
        scratch.mkdir(mode=0o700, exist_ok=True)
        return SessionEnvironment(session, root, scratch, dict(DURABLE_METADATA), durable=True)
    sid = session_id or secrets.token_hex(8)
    scratch = Path(tempfile.mkdtemp(prefix=f"{SCRATCH_PREFIX}{sid}-"))
    return SessionEnvironment(sid, root, scratch, dict(LOCAL_METADATA))


def delete_session_scratch(session: str, sessions_dir: str | Path) -> None:
    """Explicitly end a durable session's scratch lifetime — the ``cleanup()`` a
    durable ``SessionEnvironment`` refuses to do for itself. Pairs with
    ``harness.memory.delete_session`` (its messages/trace) so a full session delete
    can remove both. Idempotent: a missing directory is not an error.

    Takes the session id, not a ``SessionEnvironment``, and recomputes the same
    sanitized path ``local_session_env`` would (see ``_safe_session_dirname``) — so
    a caller who only has the id on hand (e.g. a ``/reset``-style command) doesn't
    need a live env to remove it.
    """
    scratch = Path(sessions_dir) / f"{_safe_session_dirname(session)}.scratch"
    shutil.rmtree(scratch, ignore_errors=True)
