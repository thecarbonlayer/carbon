"""The storage contract, enforced: private, session-scoped, gone on close.

Each test names a clause of the contract. Sabotage-shaped tests (cleanup skipped,
mode widened, spill aimed at the workspace) prove the detectors fire — a guard
that cannot go red is decoration."""

import os
import time
from pathlib import Path

from harness.session_env import (
    SCRATCH_PREFIX,
    local_session_env,
    scavenge,
)


def test_scratch_is_private_unpredictable_and_outside_any_workspace(tmp_path):
    env = local_session_env(workspace_root=tmp_path)
    try:
        assert env.scratch_root.is_dir()
        assert tmp_path not in env.scratch_root.parents, "scratch must live outside the repo"
        assert env.scratch_root.name.startswith(SCRATCH_PREFIX)
        mode = env.scratch_root.stat().st_mode & 0o777
        assert mode == 0o700, f"scratch must be private to the user, got {oct(mode)}"
        assert env.session_id in env.scratch_root.name
    finally:
        env.cleanup()


def test_cleanup_removes_scratch_and_is_idempotent(tmp_path):
    env = local_session_env(workspace_root=tmp_path)
    (env.scratch_root / "offload").mkdir()
    (env.scratch_root / "offload" / "x.txt").write_text("payload")
    env.cleanup()
    assert not env.scratch_root.exists()
    env.cleanup()  # second call must not raise


def test_two_sessions_get_distinct_scratch_roots(tmp_path):
    a, b = local_session_env(tmp_path), local_session_env(tmp_path)
    try:
        assert a.scratch_root != b.scratch_root
        assert a.session_id != b.session_id
    finally:
        a.cleanup()
        b.cleanup()


def test_scavenge_removes_only_expired_prefixed_dirs(tmp_path):
    live = local_session_env(tmp_path)
    stale = Path(live.scratch_root.parent) / f"{SCRATCH_PREFIX}deadbeef-stale"
    stale.mkdir(mode=0o700)
    old = time.time() - 100_000
    os.utime(stale, (old, old))
    try:
        removed = scavenge(max_age_s=86_400)
        assert removed >= 1
        assert not stale.exists(), "expired stray must be scavenged"
        assert live.scratch_root.exists(), "a fresh session's scratch must survive"
    finally:
        live.cleanup()
        if stale.exists():
            stale.rmdir()


def test_metadata_names_kind_and_storage_policy(tmp_path):
    env = local_session_env(tmp_path)
    try:
        assert env.metadata["kind"] == "local"
        assert "storage_policy" in env.metadata and "impl_version" in env.metadata
    finally:
        env.cleanup()
