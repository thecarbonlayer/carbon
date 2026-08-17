"""Session-scoped runtime storage — the harness's private scratch.

The workspace is the user's durable project; everything the HARNESS creates at run
time (offloaded tool output today) is runtime state with a different lifecycle and a
different audience. Holding runtime state inside the workspace made it repo-visible
(git, scanners, sync) and gave it the workspace's unbounded lifetime — the measured
cost was a security task counting a harness cache file as a workspace leak.

The contract, which tests enforce clause by clause:
  - an EPHEMERAL session's scratch (the default) is PRIVATE (0700), UNPREDICTABLE
    (mkdtemp draws its name from a CSPRNG, not the session id), and lives entirely
    OUTSIDE the repo — the OS temp dir, unrelated to any project tree;
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
    trades BOTH ephemeral guarantees above for a deterministic path under
    ``sessions_dir`` instead: PREDICTABLE by design (reopening the same session must
    land on the same scratch — see the naming-collision note below), and not
    guaranteed to be outside the repo (``sessions_dir`` defaults to project-relative
    ``.sessions``; that it usually ends up gitignored is a caller convention, not
    something this module enforces). What it keeps: still PRIVATE (0700 — see the
    mode caveat below), and tied to the SESSION's lifetime rather than to whichever
    process happens to have it open, because a persisted transcript can hold
    ``scratch://`` refs (``offload_to_file``) that must still resolve the next time
    the same session is reopened. Its ``cleanup()`` is a no-op; only
    ``delete_session_scratch()`` — or deleting the session itself via
    ``harness.memory.delete_session``, which now calls that too — ends it.
    Structurally out of ``scavenge()``'s reach as well: it lives under
    ``sessions_dir``, never the OS temp dir ``scavenge()`` sweeps;
  - PRIVATE (0700) holds for the leaf scratch directory in both lifetimes, and for a
    freshly-created ``sessions_dir``. Two caveats, both about creation, not a hole
    anything can walk through: ``mkdir(parents=True)`` applies ``mode`` only to the
    directory it was actually asked to create, so any intermediate parent
    ``local_session_env`` has to create along the way (whichever of
    ``sessions_dir``'s own ancestors are ALSO missing) lands at the platform default
    instead — verified empirically at 0755 here, umask-dependent, not assumed from
    the docs; and a ``sessions_dir`` or scratch dir that already existed before this
    call (e.g. one ``memory.py``'s ``save_session`` created first, which does not
    restrict its own mode) is left at whatever mode it already had;
  - another session cannot NAME an ephemeral scratch (the mkdtemp component is
    unguessable). A durable session's name IS its sanitized session id — deliberately
    predictable, not secret — so two DIFFERENT raw ids that sanitize to the same
    component collide on purpose: ``"a/b"`` and ``"b"`` both land on ``"b.scratch"``,
    the identical collapsing ``harness/memory.py:_path`` already accepts for the
    ``.jsonl`` file (see ``_safe_session_dirname``). That is a caller-created naming
    collision, not a break-in from outside the sanitization rule;
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

from harness.limits import forget_spills

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
        ``cleanup()`` this time. The early return matters for more than the
        directory: it also means an ephemeral close never reaches
        ``forget_spills`` below for a durable scratch — a durable session's own
        earlier spills are named by a transcript that outlives THIS process (Task
        3), so forgetting them from ``_OURS`` here would let a later reopen's
        ``_prune`` reclaim a file that transcript still points at, the exact bug
        Task 3 fixed. See test_durable_cleanup_does_not_forget_ours_entries."""
        if self.durable:
            return
        shutil.rmtree(self.scratch_root, ignore_errors=True)
        # AFTER the rmtree, not before: forget_spills is pure in-memory
        # bookkeeping (harness.limits' module-global ``_OURS`` set only ever
        # grows without it — one Path per spill, for the rest of the PROCESS's
        # life, long after the directory it named is gone — Task 4), but this
        # method's own contract is "never raises," and ordering the directory
        # removal FIRST means even a bug in forget_spills can no longer also
        # skip the rmtree it used to sit in front of.
        forget_spills(self.scratch_root)


def _newest_mtime(root: Path) -> float:
    """The most recent modification time anywhere under ``root``, including ``root``
    itself.

    A directory's mtime only moves when a DIRECT child is added or removed — writing
    into an already-existing grandchild (a result landing in ``offload/x.txt`` when
    ``offload/`` was created earlier in the session) never bumps ``root``'s own mtime
    again, only ``offload/``'s. Judging staleness from ``root.stat().st_mtime`` alone
    therefore reads an actively-spilling session as idle the moment the session has
    simply run longer than ``max_age_s`` — see ``scavenge``. Walking the whole tree
    and taking the max is what makes "newest write anywhere" the actual signal.
    Falls back to ``root``'s own mtime when the tree holds nothing else (an empty or
    just-created scratch, before any child exists to outrank it).
    """
    newest = root.stat().st_mtime
    for child in root.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue  # a child removed mid-walk (another process's concurrent cleanup)
    return newest


def scavenge(max_age_s: float = SCAVENGE_AGE_S, *, root: str | Path | None = None) -> int:
    """Remove abandoned scratch directories (a crash's leftovers) past their expiry.

    Opportunistic by design: it runs when the next session starts, so a machine that
    never runs carbon again keeps at most what the OS temp reaper would take anyway.
    Only prefixed, real directories are touched; a same-named symlink is ignored.

    Staleness is judged from the NEWEST mtime anywhere in the tree (``_newest_mtime``),
    not the root directory's own — a live session whose only recent activity is a
    write into an already-existing child (``offload/x.txt``) does not bump the root's
    mtime again, so reading the root alone would reap a session still in use out from
    under the process using it — see
    ``test_scavenge_judges_staleness_from_the_newest_write_not_the_roots_own_mtime``.

    A durable session's scratch (``local_session_env(..., session=..., sessions_dir=...)``)
    is structurally out of reach here: it lives under ``sessions_dir``, never under
    the OS temp dir this glob searches, so no ``max_age_s`` can make this function
    remove it — see ``test_scavenge_does_not_touch_durable_scratch``.

    ``root`` overrides where this sweeps (default: the real OS temp dir,
    ``tempfile.gettempdir()``) — a seam for tests, not a production knob: every
    production caller (``local_session_env``) omits it and gets the real sweep,
    unchanged. A test exercising an aggressive ``max_age_s`` (short enough to run
    fast) against the REAL temp dir risks reaping a genuinely still-running
    session's scratch on a shared or long-lived machine — reproduced by a
    reviewer, a real 60-second-old stray on the machine this suite ran on. A
    dedicated ``root`` keeps a test's sweep confined to what the test itself
    created.
    """
    now = time.time()
    removed = 0
    sweep_root = Path(root) if root is not None else Path(tempfile.gettempdir())
    for p in sweep_root.glob(f"{SCRATCH_PREFIX}*"):
        try:
            if p.is_dir() and not p.is_symlink() and now - _newest_mtime(p) > max_age_s:
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
    durable scratch dir outside ``sessions_dir``. The empty-component fallback to
    ``"session"`` fires for ``"."``, ``"/"``, or ``""`` — verified empirically, NOT
    for ``".."``: ``Path("..").name`` is the literal two-character string ``".."``,
    not empty, so an id of exactly ``".."`` does not hit this fallback at all. It
    sanitizes to the ordinary, contained component ``".."``, landing the scratch at
    ``"...scratch"`` — a normal filename (three dots, then letters) that is safe
    precisely because it is NOT the path component ``".."``; nothing downstream
    interprets it as "go up a directory". This mirrors ``memory.py``'s fallback
    exactly (same trigger, same replacement), so the two modules never disagree
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
    durable ``SessionEnvironment`` refuses to do for itself. Idempotent: a missing
    directory is not an error.

    ``harness.memory.delete_session`` calls this directly now (its messages/trace,
    then this), so a session delete removes both without a caller having to
    remember to pair the two — ``ui/tui.py``'s ``/reset`` is the one production
    caller and gets it for free. This stays a public, standalone function for the
    rarer case of wanting only the scratch gone.

    Takes the session id, not a ``SessionEnvironment``, and recomputes the same
    sanitized path ``local_session_env`` would (see ``_safe_session_dirname``) — so
    a caller who only has the id on hand (e.g. ``memory.delete_session``, or a
    ``/reset``-style command) doesn't need a live env to remove it.
    """
    scratch = Path(sessions_dir) / f"{_safe_session_dirname(session)}.scratch"
    shutil.rmtree(scratch, ignore_errors=True)
