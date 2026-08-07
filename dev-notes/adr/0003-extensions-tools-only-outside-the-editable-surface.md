# ADR 0003 — Extensions: tools-only, loaded outside the editable surface

## Status

Accepted.

## Context

Two sibling projects Carbon is in conversation with ship real extension
systems: Tau (`tau_coding/extensions/`, a Python port of Pi) and Pi
(`packages/coding-agent/src/core/extensions/`, TypeScript). Both discover
`.py`/`.ts` files from user- and project-level directories, call a `setup`
entrypoint with an API object, and let that object register tools, slash
commands, message renderers, and — the largest part of the surface — react to
roughly 25 (Tau) to 30+ (Pi) lifecycle events via `on(event, handler)`. A
large fraction of that event surface exists to support one use case:
permission-gating and middleware around tool calls (`tool_call`/`tool_result`
in both; Pi and Tau both ship a `permission-gate` example built on exactly
that pair).

Carbon already has two seams that cover this use case without a hook system:

- `Policy` (`harness/policy.py`): a per-agent permission rule — allow, deny,
  read-only, an approval callback — consulted before every tool call.
- `ToolRegistry.wrap()` (`harness/tools.py`): replaces a tool's function with
  `wrapper(original_func)` in place. Its own docstring already calls it "the
  generic mechanism behind fault injection, logging, caching, and permission
  middleware."

Adding a `tool_call`/`tool_result` hook pair on top of these would be a
second way to do something Carbon can already do, which ADR 0002's razor
rules out directly: a change belongs in Carbon only if it is a seam more than
one consumer would hang different domain logic on, and `wrap()` is already
that seam.

Separately, Carbon's self-improving loop (the sibling `refinery` repo) may
only edit `harness/harness_config.json`, validated through
`harness_config.py`'s schema and published via `surface_manifest()`. Per
`refinery/docs/carbon-quality-review.md`, that loop cannot inject code, weaken
verification, relax workspace boundaries, or make ambiguous edits legal. An
extension system is, by construction, a way to load and run new code, so
whatever shape it takes must sit entirely outside that config surface — not
discoverable, not creatable, not toggleable by anything that only edits JSON.

## Decision

**Scope: tools only.** An extension is a Python file exposing:

```python
def setup(registry: ToolRegistry) -> None: ...
```

Sync only — every `Tool.func` in Carbon is already sync, and supporting async
`setup` would introduce a second execution model for one feature. Inside
`setup`, an extension calls the `ToolRegistry` methods that already exist and
are already public: `register()` for a new tool, `wrap()` to layer behavior
onto an existing one (logging, caching, fault injection, argument rewriting),
`get()`/`names()` to introspect what's there. No command registry, no
lifecycle-event bus, no provider registration ship in v1. If a real need for
one of those emerges, it gets its own ADR — the same way T1.3 (`Policy`) and
T1.4 (registry introspection) each earned their place on the SDK seam roadmap
by a consumer hand-building the same thing twice.

**Discovery.** `harness/extensions.py` adds:

- `discover_extensions(directory) -> list[Path]` — every `*.py` file directly
  under `directory` (no subdirectories, no `pyproject.toml` manifest — that's
  what Tau's `[tool.tau]` table and Pi's `package.json` field add, and no
  Carbon consumer has asked for it yet; YAGNI until one does).
- `load_extensions(registry, *directories) -> list[str]` — for each directory
  in the order given, import each discovered file
  (`importlib.util.spec_from_file_location`), call its `setup(registry)`, and
  collect the loaded names. A directory that doesn't exist is silently
  skipped (matches `load_skills`'s handling of a missing `skills/` dir). A
  file with no `setup` attribute, or whose `setup` raises, is recorded as an
  error and skipped — it does not stop the rest of that directory or later
  directories from loading (matches how both Tau and Pi isolate a broken
  extension from the others).

**Wiring — no `Agent` change.** Skills need a new `Agent(skills=...)`
parameter because `Agent` renders them into the system prompt
(`skills_prompt`). Extensions only touch a `ToolRegistry`, and `Agent` already
takes one via `tools=`. So the call site (the CLI's `main()`/`_run_repl()` in
`harness/agent.py`, mirroring the existing `skills=load_skills("skills")`
line) does:

```python
tools = default_tools()
load_extensions(tools, Path.home() / ".carbon" / "extensions", Path(".carbon/extensions"))
agent = Agent(tools=tools, ...)
```

User directory first, then project directory, so a project-local extension
can override a user-level one of the same tool name (last `register()` wins,
which is already `ToolRegistry.register()`'s behavior — no new precedence
rule to implement). Both paths are Python literals at the CLI construction
site, not values read from any config file.

**Why this cannot become part of the editable surface.** `extensions_dir`
is deliberately *not* an `Agent.__init__` parameter, not a `HarnessConfig`
field, and not something `config_schema()`/`surface_manifest()` ever mentions.
The only way to point Carbon at an extensions directory is to edit the Python
source that constructs the `ToolRegistry` before handing it to `Agent`, or to
call `harness.extensions.load_extensions` directly as an SDK consumer would.
Refinery's loop reads and writes `harness_config.json` through
`load_config`/`surface_manifest()`; that door has no field to put an
extensions path in, and adding one would require a source change, which is
exactly the category of edit refinery is barred from making. The boundary
isn't a runtime check — it's the absence of a knob.

**Example.** `extensions/audit_log.py`, one real extension echoing how
`skills/sign-off/SKILL.md` is the one shipped example skill — with a
deliberate asymmetry, not a mirror: a skill auto-loads from `skills/`,
while an extension requires both explicit placement under `.carbon/extensions/`
*and* `--extensions` at the command line (see below). It wraps every tool
already in the registry to append a timestamped `tool, args, result` line
to `.carbon/audit.log`, and registers a new `read_audit_log` tool to read
that log back — exercising `register()` and `wrap()` in the same file,
which is also the evidence for this ADR's central claim that the existing
`ToolRegistry` seam is sufficient without a hook system.

**Loading is opt-in, off by default.** A final whole-branch review surfaced
a gap this ADR's original Decision missed: the project-local extensions
directory (`.carbon/extensions/`) sits *inside* the agent's own writable
workspace. Every other mutating tool in Carbon is approval-gated fresh, each
time its effect occurs — `write_file`, `edit_file`, `bash` all re-prompt on
every call. An extension file is qualitatively different: once a `write_file`
call plants one there, it auto-executes, unsandboxed, on every subsequent
`carbon` invocation, with no further approval at all. That is a standing
write vector the agent's own tools can arm against future runs, not a single
gated action.

The resolution: extension loading — both the user-level and the
project-local directories — sits behind an explicit `--extensions` CLI flag,
off by default, wired identically across all three entrypoints (print mode,
the REPL, the TUI). When the flag is absent, `load_extensions` is never
called and `_extension_dirs`/`Path.home()` are never even touched — loading
is a true no-op, not merely an empty result.

A container-isolated execution model for extensions was considered and
deferred. Extensions currently work by calling `registry.wrap()`/`register()`
directly on the harness's in-process `ToolRegistry`; a sandboxed process
couldn't reach that registry without inventing a serialization/RPC boundary
for tool dispatch, which is a materially larger redesign than this ADR's
scope. It also wouldn't by itself solve the no-re-approval property above —
a sandboxed extension could still auto-load and run unattended every time.
Out of scope for this iteration; the flag is the fix that matches the actual
risk.

## Consequences

- Carbon gets a real extension mechanism — new tools and tool middleware,
  loaded from outside the source tree, without editing `harness/tools.py` or
  recompiling anything — while staying materially smaller than Tau or Pi's
  systems. No `ExtensionAPI` wrapper object, no event bus, no command
  registry. The seam is exactly `ToolRegistry`, already public, already
  documented as the middleware mechanism.
- If a future consumer needs to react to something that isn't a tool call
  (e.g., turn start/end, compaction, session lifecycle), that is new
  mechanism and needs its own ADR and its own justification against ADR
  0002's razor — this decision does not pre-approve a hook system, it
  explicitly declines to build one now.
- Commands (REPL `/slash` commands) are out of scope for the same reason:
  Carbon has no command-registration mechanism at all yet, so adding one
  under "extensions" would be inventing a second new seam in the same
  change. A command registry, if ever needed, is a separate proposal.
- The refinery invariant is enforced by construction, not by a guard that
  could be bypassed: there is no field for an extensions path to occupy on
  the editable surface, so there is nothing there to widen, discover, or
  turn on. Tests assert `surface_manifest()`/`config_schema()` carry no
  extension-related key, and that `harness/extensions.py` never imports
  `harness_config`.
- Alternatives considered and rejected: a richer `ExtensionAPI` object
  wrapping `ToolRegistry` (rejected — an abstraction layer over a seam that's
  already public, adding indirection without adding capability); a
  `tool_call`/`tool_result` hook pair for permission-gating extensions
  (rejected — `Policy` and `wrap()` already cover it, see Context); a
  `pyproject.toml`-style extension manifest for multi-file extensions
  (rejected for v1 as unneeded complexity — flat `*.py` files cover the one
  shipped example and every SDK-seam-roadmap consumer's current need).
