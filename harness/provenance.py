"""Provenance — which harness state produced a result (v0.1, the embedding seam).

A consumer that measures the agent (an eval suite, a self-improving loop) needs to
attribute a result to the harness state that produced it. Real consumers were each
shelling out to git and reading the config version by hand. This returns the
primitives; the consumer composes its own fingerprint on top — a verifier hash, a
dirty-tree hash — because those are the consumer's policy, not gemma's
(dev-notes/adr/0002).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from harness.harness_config import CONFIG


def _git_sha(root: str | Path) -> str | None:
    """Short HEAD sha of the checkout at ``root``, or ``None`` when it is not a git
    repo or git fails. Best-effort by design: provenance must never crash a run."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def provenance(model: str | None = None, root: str | Path = ".") -> dict:
    """The harness-identity primitives: config surface version, model, checkout sha.

    ``config_version`` is the editable surface's own counter; ``gemma_sha`` is the
    checkout that produced the run (``None`` outside a git repo). A consumer layers
    its own identity (dataset, verifier version) onto this dict.

    The key stays ``gemma_sha`` after the project was renamed to carbon: this is a
    wire format, and every result record ever written carries it. Renaming it would
    orphan the committed measurement history (refinery's results/ and iterations/),
    which is cited evidence, not scratch data. New keys get the new name; this one
    is frozen.
    """
    return {
        "config_version": CONFIG.version,
        "model": model,
        "gemma_sha": _git_sha(root),
    }
