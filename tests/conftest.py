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

import os
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

    Fixed by OWNERSHIP, not diffing: ``CARBON_SCRATCH_TEST_ROOT`` (honoured by
    ``local_session_env`` and by ``scavenge()``'s own default root — see
    ``harness/session_env.py::scratch_parent_dir``) redirects every ephemeral
    scratch this PROCESS creates, for the rest of the session, into a throwaway
    directory nothing else on the machine ever writes to. A concurrent process
    never sees this env var — it is set only in THIS pytest process's own
    environment — so its scratch keeps landing in the real OS temp dir,
    structurally out of reach of anything below: not merely excluded by a check
    that could be wrong, but never glob-reachable from the redirected root at
    all. (See ``test_scavenge_default_root_follows_the_test_root_override`` and
    ``test_ephemeral_scratch_honors_the_test_root_override`` in
    ``test_session_env.py`` for the mechanism, pinned directly.)

    Considered and rejected: tracking the ``SessionEnvironment`` objects each
    test itself constructs (wrap ``local_session_env`` in a list-recording
    monkeypatch, sweep only those). That stays a per-test bookkeeping problem —
    every construction site would need to feed the list, including the ~69 bare
    ``Agent(...)`` calls across this suite that build one internally — where
    redirecting the CREATION LOCATION, once, needs no per-call cooperation at
    all: a session env built anywhere, any way, during this suite lands under
    the redirected root automatically.

    Session-scoped, not per-test: the old fixture's snapshot-diff globbed the
    temp dir twice per test regardless of whether that test ever built an Agent
    (~54ms per glob measured against a real 91k-entry temp dir on the machine
    this was measured on — pure overhead on every test that never touches
    scratch). Redirecting once, up front, costs one env var write for the whole
    session; nothing per-test remains to glob at all. What leaks during the run
    now accumulates in a directory that starts empty and holds at most a few
    hundred entries (this suite's own unclosed Agents) rather than the host's
    entire ambient temp dir, so even ``local_session_env``'s own bare
    ``scavenge()`` call stays cheap throughout — and the whole root is removed,
    unconditionally, when the session ends.
    """
    from harness.session_env import SCRATCH_TEST_ROOT_ENV

    root = Path(tempfile.mkdtemp(prefix="carbon-pytest-scratch-"))
    previous = os.environ.get(SCRATCH_TEST_ROOT_ENV)
    os.environ[SCRATCH_TEST_ROOT_ENV] = str(root)
    try:
        yield root
    finally:
        if previous is None:
            os.environ.pop(SCRATCH_TEST_ROOT_ENV, None)
        else:
            os.environ[SCRATCH_TEST_ROOT_ENV] = previous
        shutil.rmtree(root, ignore_errors=True)
