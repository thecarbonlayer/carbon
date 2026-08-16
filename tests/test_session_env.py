"""The storage contract, enforced: private, session-scoped, gone on close.

Each test names a clause of the contract. Sabotage-shaped tests (cleanup skipped,
mode widened, spill aimed at the workspace) prove the detectors fire — a guard
that cannot go red is decoration."""

import os
import tempfile
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


# --- the Agent's ownership of a SessionEnvironment -----------------------------
# Task 4: Agent constructs its own SessionEnvironment when none is supplied, and
# close() ends that scratch's lifecycle — but only when the Agent is the one that
# created it. A caller-supplied env is shared, and sharing is not ownership: it is
# the supplier's to clean, not this Agent's, or a worker closing on its way out
# would yank the scratch out from under the parent (or a sibling) still using it.
def test_agent_owns_and_cleans_an_env_it_created(tmp_path):
    from harness.agent import Agent

    a = Agent(agents_dir=str(tmp_path))  # no session_env given -> Agent creates one
    scratch = a.session_env.scratch_root
    assert scratch.is_dir()
    a.close()
    assert not scratch.exists()
    a.close()  # idempotent


def test_agent_never_cleans_a_caller_supplied_env(tmp_path):
    from harness.agent import Agent
    from harness.session_env import local_session_env

    env = local_session_env(tmp_path)
    try:
        a = Agent(agents_dir=str(tmp_path), session_env=env)
        a.close()
        assert env.scratch_root.exists(), "a shared env is the creator's to clean"
    finally:
        env.cleanup()


# --- leak detection at the real driving code paths -----------------------------
# The tests above call Agent.close()/env.cleanup() directly — real proof the
# CONTRACT is right, but not proof a real caller actually triggers it. The
# autouse `_sweep_scratch_dirs_this_test_leaked` fixture (conftest.py) removes
# every stray `carbon-scratch-*` dir left behind after EACH test regardless of
# why it was left — which is exactly why a regression in `Agent.close()` (the
# cleanup call quietly dropped, or a caller that stops running it in `finally`)
# would not fail the suite: the fixture hides the very symptom that first
# exposed this leak class (755 stray dirs after a full run, ~330x slower
# scavenge()). These assert inside the test body, snapshot-before/compare-after,
# BEFORE that fixture's own sweep gets a turn — so a broken close() shows up
# here, in seconds, instead of only in a slow scavenge() on some future session.
def _scratch_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"{SCRATCH_PREFIX}*"))


def test_run_once_leaves_no_scratch_dir_behind(tmp_path):
    """``run_once`` (print mode's real non-interactive entrypoint) builds an Agent
    that owns its env and closes it in ``finally`` — driven here, fully offline,
    through the fake provider. If that ``finally`` ever stopped calling ``close()``
    (or ``close()`` stopped calling ``cleanup()``), this is what would go red."""
    from harness.agent import run_once
    from model import fake

    before = _scratch_dirs()

    run_once(
        "say hi",
        provider=fake(scripted=lambda msgs: "hi"),
        sessions_dir=str(tmp_path),
        workspace_root=str(tmp_path),
        agents_dir=str(tmp_path),
    )

    leaked = _scratch_dirs() - before
    assert not leaked, f"run_once leaked scratch dirs: {leaked}"


def test_run_subagent_with_no_env_leaves_no_scratch_dir_behind(tmp_path):
    """A `run_subagent` call given no `session_env` builds a worker Agent that
    owns its own env — nothing else in the program holds a reference to it, so
    `run_subagent` is the only call that can close it. Same real-driver shape as
    the test above, for the delegation entrypoint instead of print mode."""
    from harness.subagents import run_subagent
    from model import fake

    before = _scratch_dirs()

    run_subagent(
        "look around",
        provider=fake(scripted=lambda msgs: "done"),
        agents_dir=str(tmp_path),
    )

    leaked = _scratch_dirs() - before
    assert not leaked, f"run_subagent (no env) leaked scratch dirs: {leaked}"
