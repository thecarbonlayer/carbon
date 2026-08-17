"""Test isolation (a ch-03 consequence).

From ch-03 the agent auto-loads ``AGENTS.md`` from its working directory. The suite
runs from the repo root, which *has* an ``AGENTS.md``, so a bare ``Agent()`` in an
earlier chapter's test would silently pick up the real project instructions and skew
its assertions. This is test infrastructure, not an agent primitive.

The fix is surgical on purpose: ignore only the *ambient* AGENTS.md (the default
``agents_dir="."``). We don't chdir or touch the real working directory, because
other chapters' tests legitimately rely on it (ch-08 reads ``pyproject.toml`` from
the repo root to prove read_file is workspace-scoped). Tests that exercise AGENTS.md
pass an explicit ``agents_dir`` and flow through to the real loader unchanged.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_agents_md(monkeypatch):
    import harness.agent as agent_mod

    real = agent_mod.load_agents_md

    def guarded(directory: str = ".", *args, **kwargs) -> str:
        if str(directory) in (".", ""):  # the ambient default — don't read the repo's file
            return ""
        return real(directory, *args, **kwargs)

    monkeypatch.setattr(agent_mod, "load_agents_md", guarded)


@pytest.fixture(scope="session", autouse=True)
def _isolated_scratch_root():
    """Give the whole suite its own scratch parent, so a leaked test scratch dir
    can never be mistaken for — or delete — a directory some OTHER process (a
    live measurement, running concurrently on the same machine) created under the
    real OS temp dir. An external review flagged the old approach P1.

    The old fixture swept ``carbon-scratch-*`` by snapshot-diffing the REAL OS
    temp dir once per test: anything NEW since the last ``yield`` was assumed to
    be this test's own leak and removed. That cannot tell "this test's leak" from
    "a different process's directory that happened to appear in the same
    window" — a live measurement running concurrently loses its scratch
    mid-attempt, which surfaces as a fabricated mechanical security failure, not
    as what it actually is (a test sweep with no way to tell the two apart).

    Fixed by OWNERSHIP, not diffing: replace ``harness.session_env``'s own
    ``scratch_parent_dir`` — the ONE function ``local_session_env`` and
    ``scavenge()``'s default root both call for "where does ephemeral scratch
    live" (see that module) — for the life of this pytest process, so every
    session env built anywhere, any way, during the whole suite lands under a
    throwaway root nothing else on the machine ever writes to. A concurrent
    process runs its OWN Python process with its OWN imported copy of
    ``harness.session_env`` — this patches only the copy loaded into THIS
    process — so a live measurement's scratch keeps landing in the real OS
    temp dir, structurally out of reach of anything below: not merely excluded
    by a check that could be wrong, but never glob-reachable from the
    redirected root at all. (See
    ``test_scavenge_default_root_follows_the_scratch_parent_dir_override`` and
    ``test_ephemeral_scratch_honors_the_scratch_parent_dir_override`` in
    ``test_session_env.py`` for the mechanism, pinned directly.)

    A monkeypatched FUNCTION, deliberately — an earlier version of this fixture
    used an env var (``CARBON_SCRATCH_TEST_ROOT``) honoured inside
    ``harness/session_env.py`` itself, which a review caught as reachable from
    PRODUCTION: an env var crosses via ``.env`` too
    (``model/provider.py``'s ``Provider.from_env()`` calls
    ``os.environ.setdefault`` for every key in that file, and every production
    entrypoint resolves a ``Provider`` before constructing its first ``Agent``)
    — one stray line in the file this project tells users to edit for their
    model endpoint would have been enough to silently redirect a REAL
    session's scratch (into the repo if the value were ``.``) and silently
    stop ``scavenge()`` from ever sweeping the real temp dir again. A swapped
    FUNCTION has zero production surface: nothing outside this fixture's own
    process can ever replace it, so there is no file, environment variable, or
    subprocess boundary left for a stray value to cross. ``pytest.MonkeyPatch()``
    is used directly, not the ordinary function-scoped ``monkeypatch`` fixture
    (which pytest does not allow at session scope) — the patch is undone
    explicitly in ``finally`` instead of automatically at a function's end.

    Considered and rejected: tracking the ``SessionEnvironment`` objects each
    test itself constructs (wrap ``local_session_env`` in a list-recording
    monkeypatch, sweep only those). That stays a per-test bookkeeping problem —
    every construction site would still need to feed the list, including every
    bare ``Agent(...)`` call across this suite that builds one internally with
    no test-visible hook — where redirecting the CREATION LOCATION, once,
    needs no per-call cooperation at all: a session env built anywhere, any
    way, lands under the redirected root automatically.

    Session-scoped, not per-test: the old fixture's snapshot-diff globbed the
    temp dir twice per test regardless of whether that test ever built an Agent
    (~54ms per glob measured against a real 91k-entry temp dir). Redirecting
    once, up front, costs one function swap for the whole session; nothing
    per-test remains to glob at all. What leaks during the run — measured
    directly against this fixture, not assumed: one stray scratch dir at the
    end of a full run, not dozens — now accumulates in a directory that starts
    empty rather than the host's entire ambient temp dir, so even
    ``local_session_env``'s own bare ``scavenge()`` call stays cheap
    throughout.

    The root's own name still starts with ``SCRATCH_PREFIX``
    (``carbon-scratch-pytest-session-<random>``, not an unrelated prefix): a
    run killed hard enough to skip this fixture's own ``finally`` (SIGKILL, an
    OOM-kill — the one failure mode no process-local cleanup can guard against)
    leaves the whole root behind, but named this way it stays glob-reachable by
    a FUTURE, unrelated ``scavenge()`` sweep of the real temp dir — reaped as
    one unit after ``SCAVENGE_AGE_S``, the same backstop every ordinary
    abandoned session already relies on, rather than orphaned forever under a
    prefix nothing will ever look for again.
    """
    import harness.session_env as session_env_mod

    root = Path(tempfile.mkdtemp(prefix=f"{session_env_mod.SCRATCH_PREFIX}pytest-session-"))
    mp = pytest.MonkeyPatch()
    mp.setattr(session_env_mod, "scratch_parent_dir", lambda: root)
    try:
        yield root
    finally:
        mp.undo()
        shutil.rmtree(root, ignore_errors=True)
