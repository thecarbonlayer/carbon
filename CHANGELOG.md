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

### Changed

- **Renamed the project and package `gemma` → `carbon`.** The repo was named
  after the model it happened to drive, which blurred the one distinction the
  curriculum exists to teach: the harness is not the model. `import gemma`
  becomes `import carbon`; the model is still whatever `LLM_MODEL` names, and
  still defaults to `google/gemma-4-26b-a4b`. No behavior changed.
- The `gemma_sha` key in `provenance()` is deliberately **not** renamed. It is a
  wire format, and every measurement record ever written carries it; renaming
  would orphan committed evidence. New keys use the new name.

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
