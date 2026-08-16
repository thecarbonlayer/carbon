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


@pytest.fixture(autouse=True)
def _sweep_scratch_dirs_this_test_leaked():
    """A safety net, not the contract: an ``Agent``/``SessionEnvironment`` a test
    owns is that test's to ``.close()``/``.cleanup()`` (the ownership tests in
    ``test_session_env.py`` assert exactly that, and keep their explicit calls —
    this fixture changes nothing about what THEY prove). But most of this suite's
    ~69 bare ``Agent(...)`` constructions never call ``close()`` at all, and each
    one abandons a real ``mkdtemp()`` directory that would otherwise sit until the
    next session's ``scavenge()`` (up to 24h — harness/session_env.py). Unswept,
    that is not merely untidy: a full suite run once left 755 stray
    ``carbon-scratch-*`` directories behind, and ``scavenge()``'s glob+stat over
    all of them measured ~330x slower with that many strays present (49.1ms) than
    clean (0.15ms) — a cost paid by the NEXT ``Agent()`` construction anywhere,
    test or production, not just by the test that leaked.

    Snapshot-before/remove-only-new-after, rather than sweeping the whole prefix
    unconditionally, so a directory some other process (or a test that
    legitimately keeps its env alive past its own yield) is responsible for is
    never touched — only what THIS test created and left behind.
    """
    from harness.session_env import SCRATCH_PREFIX

    root = Path(tempfile.gettempdir())
    before = set(root.glob(f"{SCRATCH_PREFIX}*"))
    yield
    for stray in set(root.glob(f"{SCRATCH_PREFIX}*")) - before:
        shutil.rmtree(stray, ignore_errors=True)
