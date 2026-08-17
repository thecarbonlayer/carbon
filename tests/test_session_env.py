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


# --- durable sessions: scratch shares the transcript's lifetime ----------------
# Task 3: a session's persisted transcript can hold scratch:// refs (offload_to_file,
# harness/limits.py) pointing into the scratch it was written from. If cleanup()
# deletes that scratch on every close() — and a reopened session gets a DIFFERENT
# root — every one of those refs is dead on reopen. Durable scratch ties the
# scratch's lifetime to the SESSION (a deterministic path under sessions_dir), not
# to whichever process/Agent happens to have it open this time.
def test_a_durable_session_keeps_its_scratch_across_close_and_reopen(tmp_path):
    """A session's transcript is persisted with scratch:// refs inside it. If close()
    deletes the scratch — and a reopened session gets a DIFFERENT root — every one of
    those refs is dead on reopen. The reference is session-scoped; the transcript
    that stores it is durable. They must share a lifetime."""
    from harness.session_env import local_session_env

    a = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    (a.scratch_root / "offload").mkdir(parents=True)
    (a.scratch_root / "offload" / "ab.txt").write_text("SPILLED")
    assert a.durable is True
    a.cleanup()
    assert (a.scratch_root / "offload" / "ab.txt").read_text() == "SPILLED"

    b = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    assert b.scratch_root == a.scratch_root, "reopening must land on the same scratch"


def test_two_sessions_do_not_share_scratch(tmp_path):
    from harness.session_env import local_session_env

    a = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    b = local_session_env(tmp_path, session="s2", sessions_dir=tmp_path / ".sessions")
    assert a.scratch_root != b.scratch_root


def test_deleting_a_session_removes_its_scratch(tmp_path):
    from harness.session_env import delete_session_scratch, local_session_env

    a = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    (a.scratch_root / "x.txt").write_text("x")
    delete_session_scratch("s1", tmp_path / ".sessions")
    assert not a.scratch_root.exists()


def test_an_ephemeral_session_still_cleans_up(tmp_path):
    from harness.session_env import local_session_env

    e = local_session_env(tmp_path)
    assert e.durable is False
    e.cleanup()
    assert not e.scratch_root.exists()


def test_session_id_cannot_escape_sessions_dir_via_traversal(tmp_path):
    """A session id is caller-supplied (argv, a REPL /new name, a TUI session picker)
    and becomes a directory name. harness/memory.py:_path already faces this exact
    problem for the *.jsonl filename and sanitizes it by taking only the final path
    component (``Path(session_id).name``) — this reuses that same rule, so a value
    like "../../../etc/passwd" can't walk the durable scratch outside sessions_dir.
    The sessions_dir itself, and the scratch inside it, must both stay 0700."""
    from harness.session_env import local_session_env

    sessions_dir = tmp_path / ".sessions"
    env = local_session_env(tmp_path, session="../../../etc/passwd", sessions_dir=sessions_dir)
    try:
        assert sessions_dir.resolve() in env.scratch_root.resolve().parents
        assert env.scratch_root.resolve().parent == sessions_dir.resolve()
        assert env.scratch_root.name == "passwd.scratch", "only the final component survives"
        mode = env.scratch_root.stat().st_mode & 0o777
        assert mode == 0o700, f"a durable scratch must be private too, got {oct(mode)}"
        sdir_mode = sessions_dir.stat().st_mode & 0o777
        assert sdir_mode == 0o700, f"sessions_dir must be private too, got {oct(sdir_mode)}"
    finally:
        env.cleanup()  # a no-op (durable), but exercises the real path, not a shortcut


def test_scavenge_does_not_touch_durable_scratch(tmp_path):
    """scavenge() globs the OS temp dir for SCRATCH_PREFIX*; durable scratch lives
    under sessions_dir instead, so it is structurally out of reach — prove it
    survives even the most aggressive scavenge() call, not just a well-timed one."""
    from harness.session_env import local_session_env, scavenge

    sessions_dir = tmp_path / ".sessions"
    env = local_session_env(tmp_path, session="s1", sessions_dir=sessions_dir)
    (env.scratch_root / "x.txt").write_text("keep me")
    scavenge(max_age_s=0)  # would remove every OS-temp scratch dir, however fresh
    assert env.scratch_root.exists()
    assert (env.scratch_root / "x.txt").read_text() == "keep me"


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


def test_agent_with_a_session_gets_a_durable_scratch_that_survives_close(tmp_path):
    """The actual bug this fixes, at the layer it was reported: an Agent opened with
    ``session=`` builds a DURABLE env, so ``close()`` does not remove its scratch, and
    a second Agent reopening the same session lands on the SAME scratch — the exact
    scenario a persisted transcript's scratch:// refs depend on."""
    from harness.agent import Agent

    sessions_dir = tmp_path / ".sessions"
    a = Agent(agents_dir=str(tmp_path), session="s1", sessions_dir=str(sessions_dir))
    assert a.session_env.durable is True
    (a.session_env.scratch_root / "offload").mkdir(parents=True)
    (a.session_env.scratch_root / "offload" / "ab.txt").write_text("SPILLED")
    a.close()
    assert (a.session_env.scratch_root / "offload" / "ab.txt").read_text() == "SPILLED"

    b = Agent(agents_dir=str(tmp_path), session="s1", sessions_dir=str(sessions_dir))
    try:
        assert b.session_env.scratch_root == a.session_env.scratch_root
    finally:
        b.close()


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
