"""Session-scoped runtime storage — the harness's private scratch, outside the repo.

The workspace is the user's durable project; everything the HARNESS creates at run
time (offloaded tool output today) is runtime state with a different lifecycle and a
different audience. Holding runtime state inside the workspace made it repo-visible
(git, scanners, sync) and gave it the workspace's unbounded lifetime — the measured
cost was a security task counting a harness cache file as a workspace leak.

The contract, which tests enforce clause by clause:
  - scratch is PRIVATE (0700, unpredictable name via mkdtemp) and OUTSIDE the repo;
  - it lives exactly as long as the session: ``cleanup()`` removes it, callers run
    that in ``finally`` (success, failure, cancellation alike);
  - crashes leak at most one directory until ``scavenge()`` — run at every session
    start — removes strays older than ``SCAVENGE_AGE_S``;
  - another session cannot name it (unpredictable component) or read it (0700);
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


@dataclass(frozen=True)
class SessionEnvironment:
    session_id: str
    workspace_root: Path | None
    scratch_root: Path
    metadata: dict = field(default_factory=lambda: dict(LOCAL_METADATA))

    def cleanup(self) -> None:
        """Remove the scratch tree. Never raises — this runs in ``finally`` blocks,
        and a cleanup error must not mask the real exception in flight."""
        shutil.rmtree(self.scratch_root, ignore_errors=True)


def scavenge(max_age_s: float = SCAVENGE_AGE_S) -> int:
    """Remove abandoned scratch directories (a crash's leftovers) past their expiry.

    Opportunistic by design: it runs when the next session starts, so a machine that
    never runs carbon again keeps at most what the OS temp reaper would take anyway.
    Only prefixed, real directories are touched; a same-named symlink is ignored.
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


def local_session_env(
    workspace_root: str | Path | None = None, session_id: str | None = None
) -> SessionEnvironment:
    """A local scratch under the OS temp dir: 0700 and unpredictable via mkdtemp,
    which closes the shared-/tmp pre-creation attack without a custom parent dir."""
    scavenge()
    sid = session_id or secrets.token_hex(8)
    scratch = Path(tempfile.mkdtemp(prefix=f"{SCRATCH_PREFIX}{sid}-"))
    root = Path(workspace_root).resolve() if workspace_root else None
    return SessionEnvironment(sid, root, scratch, dict(LOCAL_METADATA))
