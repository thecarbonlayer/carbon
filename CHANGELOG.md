# Changelog

The teaching curriculum is versioned by chapter tags (`ch-00`..`ch-14`) and does
not change. This file tracks the other axis: the `carbon` library and its
editable surface, which evolve continuously as external consumers and the
self-improving loop push on the seam. See
[dev-notes/adr/0001](dev-notes/adr/0001-version-the-evolution-separately.md).

The format follows [Keep a Changelog](https://keepachangelog.com/), and versions
follow [Semantic Versioning](https://semver.org/). The configuration's own
integer `version` field is the fine-grained counter underneath these releases.
One entry per release; commits stay fine-grained under a `feat(surface)` or
`feat(sdk)` scope.

## [Unreleased]

### Added

- `harness/extensions.py`: a tools-only extension loader. A Python file under
  `~/.carbon/extensions/` or a project's `.carbon/extensions/` exposing
  `setup(registry: ToolRegistry) -> None` can register new tools or `wrap()`
  existing ones, using the same `ToolRegistry` seam consumers already use —
  no new hook system, no `Agent` change. Deliberately outside
  `surface_manifest()`: refinery cannot discover, create, or point at an
  extensions directory (dev-notes/adr/0003). Ships one example,
  `extensions/audit_log.py`. Exported from `carbon` as `load_extensions` /
  `discover_extensions`. Off by default: loading requires an explicit
  `--extensions` flag on every entrypoint (print mode, the REPL, the TUI) —
  the project-local directory lives inside the agent's own writable
  workspace, so loading it unconditionally would let `write_file` plant a
  file that auto-runs, unapproved, on every later invocation.
- `apply_patch`: atomic, multi-hunk, multi-file unified-diff application
  (`harness/patch.py` + `Workspace.apply_patch`). `edit_file` already
  guarantees exact-match-or-refuse for one location; `apply_patch` gives the
  same guarantee — validate every file first with zero writes, commit all of
  them only once every file in the patch has validated — for a change
  spanning several files or several locations in one file, which is most
  real changes. Renames, copies, binary patches, and fuzzy/nearby context
  matching are refused outright rather than approximated: a hunk's context
  must match exactly at the claimed location or the whole patch is refused.
  Wired into `_coding_tools` and `approval_tools` alongside `write_file`/
  `edit_file` — it can touch several files in one call, so it gets the same
  approval gate.
- `token_budget_checkpoint`: a fourth compaction strategy alongside the two
  existing message-count ones, additive and default-neutral (the shipped
  default stays `structured_checkpoint`; `config_version` does not move for
  this entry). Cuts by token budget rather than a fixed message count — a
  `keep_tail` count is nearly nothing after a turn of acknowledgements and
  far over budget after a turn that read a file — snaps mid-turn cuts
  earlier so a summarizer is never handed half an exchange, carries the
  previous checkpoint forward as its own message rather than re-summarizing
  it (re-summarizing a checkpoint against no new material is exactly how a
  fact erodes across repeated compactions), and deterministically
  re-attaches read/modified file paths extracted from tool calls rather
  than trusting the summarizer's prose to keep them. An explicit
  `completion_reserve` replaces a pure trigger-fraction, since a reply is
  about the same size at 4k context and at 128k. A bounded oversize
  fallback reuses the existing truncation door rather than a private clamp.
- `Agent(deadline_s=...)`: a wall-clock bound on `Agent.run()`, parallel to
  `max_tool_steps`. Checked at the top of each loop iteration — a turn
  already in flight completes rather than being torn down mid-call, no new
  turn starts once the deadline has passed. `None` by default; every
  existing caller is unchanged. New `stop_reason` value: `"deadline"`.
- `LLM_REASONING_EFFORT`: requests a reasoning-effort level from models that
  support one (OpenAI's GPT-5.x series via OpenRouter, among others).
  Provider-level, like `LLM_MODEL`/`LLM_BASE_URL` — a property of which
  model is being measured, not of an individual call. Forwarded as
  OpenRouter's own wire format (`{"reasoning": {"effort": ...}}`) only when
  set; every other provider's requests are unaffected.

### Changed

- **Renamed the project and package `gemma` → `carbon`.** The repo was named
  after the model it happened to drive, which blurred the one distinction the
  curriculum exists to teach: the harness is not the model. `import gemma`
  becomes `import carbon`; the model is still whatever `LLM_MODEL` names, and
  still defaults to `google/gemma-4-26b-a4b`. No behavior changed.
- The `gemma_sha` key in `provenance()` is deliberately **not** renamed. It is a
  wire format, and every measurement record ever written carries it; renaming
  would orphan committed evidence. New keys use the new name.
- **`default_tools()` drops `calculator`, gains `list_files`/`search_text`.** The
  default registry only ever exposed `calculator` + `read_file`, while
  `list_files_tool()` and `search_text_tool()` — hardened the same way as
  `read_file` (workspace-confined, secrets excluded, symlink-escape checked) —
  sat implemented but unregistered, and the system prompt left exploration to
  the model's own judgment instead of pointing at a workspace-confined tool
  for it. `calculator` was a ch-05 teaching artifact a coding agent doesn't
  need — every check and test that used it as a stand-in for "some tool with
  a verifiable exact result" now uses `read_file`/`list_files`/`search_text`
  (or a small local fake) instead, so `calculator()` and its `Tool` wrapper
  were deleted outright rather than kept around as a permanently-unused
  opt-in. `Orchestrator` gained a real `tools=` constructor param in the same
  pass, closing the one place that previously had no way to give a worker
  anything other than the process-wide default.
- **`default_tools()` takes an optional `root`.** `harness/agent.py`'s
  `_coding_tools()` and `tasks/checks.py` had each grown their own copy of
  "register `read_file_tool(root)`, `list_files_tool(root)`,
  `search_text_tool(root)` on a fresh registry" — `_coding_tools()` even
  called the unrooted `default_tools()` first and then immediately
  overwrote all three of its tools with rooted ones, making that first call
  a no-op. `default_tools(root=None)` now does the rooting itself, and all
  three call sites use it directly.
- Default system prompt now explicitly points at `list_files`/`search_text`
  ahead of shelling out to `ls`/`grep`/`find` ("Explore before you edit:
  prefer list_files and search_text over ls/grep/find for finding the right
  files..."), rather than leaving exploration to the model's own judgment.
  `config_version` 3 → 5, also covering the retry-policy change below.
- **Default retry policy widened: 3 attempts/100ms base delay → 5
  attempts/2000ms.** Found live: an unattended batch run lost 35% of
  attempts to the same transient provider 502, and 3 attempts at a 100ms
  base delay adds up to well under a second of patience — nowhere near
  enough to survive a sustained provider-side hiccup. 5/2000ms (exponential
  backoff, ~30s total patience across 4 retries) was tuned for unattended
  runs, not interactive use, where fast failure still matters more than
  patience.
- `compact()` dispatches through a `_Strategy` registry instead of
  branching on strategy name inline at five separate points.
  Behavior-preserving — adding a future strategy now means one dict entry,
  not re-reading `compact()` for every place an old strategy name was
  checked. `truncate()` and the retry dispatch are now registry-shaped the
  same way, for consistency across every strategy-shaped config knob
  (compaction, tool_output/file_injection, retry).

## [0.4.0] - 2026-07-26

Closes the highest-impact output-quality gaps found by a differential review of
the agent's own output, and widens the self-improvement surface from scalar
tuning to bounded strategy selection, so Refinery has more to work with.

### Added

- `file_injection`, `tool_output`, `compaction`, and `retry` strategy objects,
  landing together as config v3.
- `surface_manifest()`, separating editable choices, locked fields, and immutable
  correctness/trust invariants.
- Ranged `read_file` access with line counts and continuation hints, plus
  workspace-confined `list_files` and `search_text` so a read-only agent can
  explore a tree without shell access.
- Tool-argument validation, explicit incomplete-response handling, and read-only
  worker binding to the parent's workspace, provider, and model.
- Forced compaction-and-retry recovery for context-window overflow.

### Changed

- **`Tool.mutates` (default `True`) — a behavior change for read-only policies.**
  `Policy(read_only=True)` previously denied only carbon's own mutator *names*,
  so a consumer tool called `save_report` ran freely. A tool now declares its
  own effect, and an undeclared one is refused: carbon cannot inspect a
  callable, and a boundary that guesses "harmless" is not a boundary. Existing
  read-only consumers must add `mutates=False` to tools that only read.
- `list_files`/`search_text` judge workspace confinement and secret refusal on
  the *resolved* target, so a symlink cannot name its way out of the root.
- Tool and sandbox output preserve both head and tail by default.
- Compaction uses a cumulative, structured, tool-aware checkpoint at 80% of the
  configured context limit.
- `edit_file` now rejects ambiguous matches, writes atomically, and returns a diff.
- Default completion budget increased from 1,024 to 4,096 tokens.

## [0.3.0] - 2026-07-24

Lets a consumer honor its own declared turn budget without reaching into
`harness.agent`'s module global. Additive.

### Added

- `Agent(max_tool_steps=...)`: a per-instance override of the tool-step budget the run loop enforces, alongside the existing `CONFIG.max_tool_steps` module default. `None` (the default) preserves prior behavior exactly.

## [0.2.0] - 2026-07-17

Lets a consumer's tool declarations carry through to the run result, so more of
its hand-built trace and truncation scaffolding can go. Both additive.

### Added

- `Tool.attributes`: static, consumer-defined metadata (a tier, a category) that carbon seeds into every `ToolCall.attributes` bag. carbon never reads it; the values are the consumer's.
- `Tool.max_result_chars`: a per-tool result budget. A chatty tool truncates at its own size instead of the global door clamp.

## [0.1.0] - 2026-07-16

Opens the embedding seam: the surface external code uses to build domain-specific
agents on the harness. Backlog and rationale in
[dev-notes/sdk-seam-roadmap.md](dev-notes/sdk-seam-roadmap.md). Every item is a
generic mechanism; domain and policy stay in the consumer (adr/0002).

### Added

- Structured run result from `Agent.run` (final text, tool calls, totals, turns, approvals, stop reason). `Agent.send` keeps returning the final text.
- Schema-constrained output mode on `chat()` and `Agent`.
- Public `ToolRegistry` introspection (get/wrap/override/list), a per-call attribute bag and `is_error` flag, and a `subscribe()` event stream.
- `Provider.from_env(root=)`, a public `load_env()`, model params as agent config, and a `provenance()` stamp.
- `config_schema()` introspection alongside the public `load_config` door.
- A curated, semantically versioned `carbon` package. Existing module paths keep working.

### Changed

- The approval gate consults a `Policy` object (allow, deny, read-only, path scope, predicate) instead of a global tool-name set plus a yes/no callback. Existing constructor arguments keep working through a compatibility layer.
