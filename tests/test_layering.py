"""Architecture, enforced rather than merely stated (ch-14, Phase 3).

Two guarantees, both derived from real imports via ``ast`` rather than declared by
hand and hoped to stay true:

1. **The outer boundary** — ``ui/ -> harness/ -> model/`` is one-way. AGENTS.md has
   said this since ch-00; nothing checked it. A model-layer or harness-layer module
   that starts importing upward would previously pass every other test.

2. **The inner seams** — ``harness/`` has ~21 flat modules with no declared layering,
   and ``agent.py`` (the loop) legitimately reaches into most of them. That breadth is
   expected for a module that ties every primitive together; what was NOT expected,
   and turned out to be true, is that the module-level import graph is already
   acyclic. This test makes that a checked invariant instead of an accident: every
   module gets a named seam (for readable failures) and the real topological depth is
   computed from the graph itself, never hand-maintained, because a hand-maintained
   depth table is exactly the kind of claim that goes stale silently (see refinery's
   own ``knob_coverage.py`` docstring on this same failure mode).

Only MODULE-LEVEL imports count. ``orchestrator.py`` and ``subagents.py`` import
``harness.agent`` lazily, inside function bodies, with an explicit comment ("avoids an
import cycle at module load") — a deliberate, already-documented pattern for the one
place the dependency is genuinely mutual: the loop constructs delegation tools from
these modules, and delegation constructs new loop instances. Restricting the graph to
module-level imports is what makes that pattern legible as "intentionally deferred"
rather than invisible; a *module-level* cycle would still fail loudly here.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- outer boundary: ui -> harness -> model, one-way -------------------------------

# Package -> packages it may import from (module-level). Absence means "may not
# import this package's modules at all". model is the floor; ui is the ceiling.
_ALLOWED_UPWARD: dict[str, frozenset[str]] = {
    "model": frozenset(),
    "harness": frozenset({"model"}),
    "ui": frozenset({"harness", "model"}),
}


def _module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported at MODULE scope — never inside a def/class,
    which is how a deliberate lazy import (breaking a real cycle) is told apart from
    an accidental one (which would fail at import time regardless, but this also
    catches the case where it wouldn't — e.g. an unused-at-runtime branch)."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
    return found


def test_outer_boundary_is_one_way():
    violations = []
    for pkg, allowed in _ALLOWED_UPWARD.items():
        for f in sorted((REPO_ROOT / pkg).rglob("*.py")):
            imported = _module_level_imports(f) & {"model", "harness", "ui"}
            illegal = imported - allowed - {pkg}
            if illegal:
                violations.append(f"{f.relative_to(REPO_ROOT)} imports {sorted(illegal)}")
    assert not violations, "one-way layering violated:\n" + "\n".join(violations)


# --- inner seams: harness/*.py, named and acyclic -----------------------------------

# Every harness/*.py module (minus __init__) mapped to a human-readable seam. A module
# with no entry here fails the completeness check below — the point is that adding a
# module forces a decision about where it belongs, not that the name changes behavior.
_SEAM: dict[str, str] = {
    "harness_config": "foundation",
    "events": "foundation",
    "instructions": "foundation",
    "policy": "foundation",
    "result": "foundation",
    "skills": "foundation",
    "tools": "tool execution",
    "checkpoint": "context management",
    "verification": "verification",
    "render": "observability",
    "orchestrator": "orchestration",
    "limits": "context management",
    "compaction": "context management",
    "context": "context management",
    "memory": "context management",
    "provenance": "foundation",
    "sandbox": "tool execution",
    "subagents": "orchestration",
    "workspace": "tool execution",
    "observability": "observability",
    "agent": "loop",
    "extensions": "tool execution",
}


def _harness_module_files() -> dict[str, Path]:
    return {f.stem: f for f in sorted((REPO_ROOT / "harness").glob("*.py")) if f.stem != "__init__"}


def _harness_graph() -> dict[str, set[str]]:
    files = _harness_module_files()
    graph: dict[str, set[str]] = {}
    for mod, f in files.items():
        deps = {d for d in _module_level_imports(f) if d != "harness"}
        # `from harness.X import Y` parses X as the module attribute, not the top-level
        # package name `_module_level_imports` returns for `import harness.X` — parse
        # ImportFrom targets explicitly against the harness package.
        tree = ast.parse(f.read_text())
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("harness")
            ):
                parts = node.module.split(".")
                if len(parts) > 1:
                    deps.add(parts[1])
        graph[mod] = {d for d in deps if d in files and d != mod}
    return graph


def test_every_harness_module_has_a_named_seam():
    modules = set(_harness_module_files())
    assert modules == set(_SEAM), (
        f"seam table is out of sync with harness/: "
        f"missing={sorted(modules - set(_SEAM))} stale={sorted(set(_SEAM) - modules)}"
    )


def test_harness_module_level_imports_are_acyclic():
    """The real regression this guards against: a NEW module-level import that closes
    a cycle. The existing agent<->orchestrator / agent<->subagents relationship stays
    outside this graph entirely, because those imports are deferred to function scope
    — that is the documented, deliberate escape hatch, and it must stay deliberate:
    a cycle appearing here means someone promoted a lazy import to the top of a file
    without noticing what that re-opens.
    """
    graph = _harness_graph()
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(graph, WHITE)
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GRAY
        stack.append(node)
        for dep in sorted(graph[node]):
            if color[dep] == GRAY:
                return stack[stack.index(dep) :] + [dep]
            if color[dep] == WHITE:
                if cycle := visit(dep):
                    return cycle
        stack.pop()
        color[node] = BLACK
        return None

    for mod in sorted(graph):
        if color[mod] == WHITE:
            if cycle := visit(mod):
                raise AssertionError(f"module-level import cycle: {' -> '.join(cycle)}")


def test_harness_topological_depth_is_well_defined():
    """A cheap corollary of acyclicity, checked directly: every module's dependency
    depth (longest chain to a zero-dependency module) is finite and computable. Not
    hand-maintained anywhere — recomputed from the graph every run, so it can never
    silently drift the way a hand-authored layer number would.
    """
    graph = _harness_graph()
    depth: dict[str, int] = {}

    def compute(node: str, trail: frozenset[str] = frozenset()) -> int:
        if node in depth:
            return depth[node]
        assert node not in trail, f"cycle reached while computing depth at {node}"
        d = 0 if not graph[node] else 1 + max(compute(dep, trail | {node}) for dep in graph[node])
        depth[node] = d
        return d

    for mod in graph:
        compute(mod)
    # agent.py ties every primitive together, so it must sit at the top of the real
    # import graph — if some OTHER module ties for the max depth, either agent grew a
    # peer nobody named, or agent stopped importing something it used to.
    assert depth["agent"] == max(depth.values()), (
        f"expected 'agent' at the top of harness/'s dependency graph, got depths={depth}"
    )
