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


def test_scavenge_judges_staleness_from_the_newest_write_not_the_roots_own_mtime(tmp_path):
    """A live session idling past ``max_age_s`` with its only recent activity one
    level down (a write into an already-existing ``offload/`` child) must not be
    reaped: creating ``offload/`` itself bumps the scratch ROOT's mtime once (adding
    a direct child touches its parent directory), but a file written *inside*
    ``offload/`` afterward bumps only ``offload/``'s own mtime, never the root's
    again. Judging staleness from the root's mtime alone — what ``p.stat().st_mtime``
    on the glob hit does today — reads a session that is actively spilling results
    as abandoned the moment it has run longer than ``max_age_s``, and the next
    ``carbon`` process to start ``scavenge()``s it out from under the one still
    using it.

    Reproduced with a real clock, not a backdated ``os.utime`` (the other scavenge
    test ages a dir artificially to prove the *removal* half works) — this proves
    the *survival* half, so the root's mtime is left exactly as directory creation
    left it, and only the sleep + later write make it stale by wall-clock time.

    Scoped to a dedicated ``root`` (``scavenge(..., root=...)``), never the real OS
    temp dir this function sweeps by default: ``max_age_s=1`` is aggressive enough
    (needed to keep the test fast) that running it against the real temp dir would
    reap ANY ``carbon-scratch-*`` directory idle for as little as a second —
    including a genuinely still-running session's, on a machine that also runs
    long-lived measurement sessions. A reviewer reproduced exactly that: a real
    60-second-old stray on the machine made a since-removed ``assert removed == 0``
    fail while the property under test (a live session survives) still held —
    ``removed`` depends on whatever else happens to be in the swept directory,
    never on this test's own fixture alone, which is also why that assertion is
    gone; the two assertions below already fully pin the actual property, the same
    way the sibling test above uses ``removed >= 1`` for the identical reason.
    """
    sweep_root = tmp_path / "scavenge-root"
    sweep_root.mkdir()
    scratch = sweep_root / f"{SCRATCH_PREFIX}mtime-test"
    scratch.mkdir(mode=0o700)
    (scratch / "offload").mkdir()  # bumps the root's mtime once, now
    time.sleep(1.2)  # both the root and offload/ itself age past max_age_s=1
    (scratch / "offload" / "x.txt").write_text("still alive")  # fresh, one level down
    scavenge(max_age_s=1, root=sweep_root)
    assert scratch.exists(), "a live session's newest write must save it from scavenge"
    assert (scratch / "offload" / "x.txt").read_text() == "still alive"


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


# --- CARBON_SCRATCH_TEST_ROOT: ownership by location, not by diffing ------------
# tests/conftest.py's session-scoped `_isolated_scratch_root` fixture relies on
# this mechanism to fix a P1: the old per-test fixture swept `carbon-scratch-*` by
# snapshot-diffing the REAL OS temp dir, which cannot tell "this test's own leak"
# from "a directory a concurrent, unrelated process (a live measurement) created
# in that same window" — and deleted the latter right along with the former. These
# two tests pin the mechanism directly, hermetically (a real system temp dir is
# never touched — both the "real" and the redirected root are tmp_path children).
def test_scavenge_default_root_follows_the_test_root_override(tmp_path, monkeypatch):
    """With CARBON_SCRATCH_TEST_ROOT set, scavenge()'s bare default (root=None —
    exactly how local_session_env() calls it) sweeps the REDIRECTED root: our own
    stray inside it is removed, and a stray outside it — standing in for a
    concurrent process's real scratch, sitting in what would be the real OS temp
    dir — is never even glob-reachable, let alone touched."""
    from harness.session_env import SCRATCH_PREFIX, scavenge

    real_system_temp = tmp_path / "real-os-tmp"
    real_system_temp.mkdir()
    foreign = real_system_temp / f"{SCRATCH_PREFIX}another-process-1234"
    foreign.mkdir(mode=0o700)
    (foreign / "evidence.txt").write_text("a concurrent measurement's spill")

    test_root = tmp_path / "pytest-owned-scratch"
    test_root.mkdir()
    ours = test_root / f"{SCRATCH_PREFIX}ours-stale"
    ours.mkdir(mode=0o700)
    old = time.time() - 100_000
    os.utime(ours, (old, old))

    monkeypatch.setenv("CARBON_SCRATCH_TEST_ROOT", str(test_root))
    # Nothing below should ever reach for this while the override is set — kept
    # pointed elsewhere so a fallback to it would show up as a wrong answer, not
    # a coincidentally-right one.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(real_system_temp))

    removed = scavenge(max_age_s=86_400)  # bare default root, no explicit root=

    assert removed == 1
    assert not ours.exists(), "our own stray, inside the redirected root, must go"
    assert foreign.exists(), "a directory outside the redirected root is untouched"
    assert (foreign / "evidence.txt").read_text() == "a concurrent measurement's spill"


def test_ephemeral_scratch_honors_the_test_root_override(tmp_path, monkeypatch):
    """The creation half of the same mechanism: a NEW ephemeral scratch lands
    under CARBON_SCRATCH_TEST_ROOT when it is set, never under the real OS temp
    dir — so nothing a test builds can ever collide with, or be mistaken for, a
    concurrent process's own scratch in the first place."""
    from harness.session_env import local_session_env

    real_system_temp = tmp_path / "real-os-tmp"
    real_system_temp.mkdir()
    test_root = tmp_path / "pytest-owned-scratch"
    test_root.mkdir()

    monkeypatch.setenv("CARBON_SCRATCH_TEST_ROOT", str(test_root))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(real_system_temp))

    env = local_session_env(tmp_path)
    try:
        assert env.scratch_root.parent == test_root
        assert not any(real_system_temp.iterdir()), "nothing landed in the 'real' temp dir"
    finally:
        env.cleanup()


def test_this_suites_own_fixture_actually_redirects_a_fresh_agent():
    """Integration check on the fixture itself (conftest.py's session-scoped
    ``_isolated_scratch_root``), not just the mechanism the two tests above pin in
    isolation: by the time ANY test runs, ``CARBON_SCRATCH_TEST_ROOT`` must already
    be set, and a perfectly ordinary, freshly-built ``Agent`` — no special wiring
    of its own — must land under it rather than under the real OS temp dir this
    process could otherwise share with a concurrent live measurement."""
    from harness.agent import Agent
    from harness.session_env import SCRATCH_TEST_ROOT_ENV

    configured = os.environ.get(SCRATCH_TEST_ROOT_ENV)
    assert configured, "the session fixture must set this before any test body runs"

    a = Agent()
    try:
        # The scratch's immediate parent is the fixture's own root — not merely
        # "somewhere under the real temp dir," which the fixture's root itself
        # also technically is (mkdtemp has to put its throwaway root somewhere).
        # Isolation comes from being confined to this ONE uniquely-named
        # subdirectory, not from avoiding the temp area altogether.
        assert a.session_env.scratch_root.resolve().parent == Path(configured).resolve()
    finally:
        a.close()


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
# CONTRACT is right, but not proof a real caller actually triggers it. conftest.py's
# session-scoped `_isolated_scratch_root` fixture redirects every ephemeral scratch
# this whole test session creates into its own throwaway root (never the real OS
# temp dir), so a per-test sweep is no longer part of the picture at all — but that
# is exactly why `_scratch_dirs()` below must follow the SAME redirect
# (`scratch_parent_dir()`) rather than hardcode `tempfile.gettempdir()`: hardcoded,
# it would watch a directory nothing lands in for the rest of the run, and
# `leaked = _scratch_dirs() - before` would be `set() - set()` regardless of
# whether `close()` actually ran — a guard that cannot go red. These assert inside
# the test body, snapshot-before/compare-after, independent of any fixture's own
# sweep — so a broken close() (the cleanup call quietly dropped, or a caller that
# stops running it in `finally`) shows up here, in seconds, the same way it did
# before the redirect existed.
def _scratch_dirs() -> set[Path]:
    from harness.session_env import scratch_parent_dir

    return set(scratch_parent_dir().glob(f"{SCRATCH_PREFIX}*"))


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
