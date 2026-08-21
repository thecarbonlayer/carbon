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

- `offload_to_file`: a third `tool_output` truncation strategy, additive and
  default-neutral (the shipped default stays `head_tail`; `config_version`
  does not move for this entry). The two inline strategies decide which bytes
  to lose; this one refuses to lose any — an over-budget tool result is
  written complete to `.carbon/offload/<content-hash>.txt` under the agent's
  workspace, the model sees the familiar `head_tail` excerpt inline, and a
  footer names the file: the workspace-relative path, the line count, the
  `read_file` call that pages it, and the `search_text` pattern that jumps
  straight to a line. Only the tool-result door offers it: `file_injection`
  already reads from a re-openable file, and the compaction fallback runs with
  no workspace in hand. For the same reason a result that is *itself*
  re-readable (a `read_file` result, which arrives with a continuation hint)
  keeps that hint instead of being copied — what is already on disk needs a
  smaller range, not another copy, and that same downgrade is what stops a page
  the model asks for from spilling in turn. **The tool-result door is the only
  door**, and that is what keeps the strategy small: a result passes it exactly
  once, so nothing has to work out what an earlier cut did to text it cannot
  authenticate. `bash_tool` composes stdout, stderr and any timeout suffix and
  applies one blunt ceiling (well above anything real; above it, completeness
  is not promised) — no policy, no workspace, no file, and no import of door
  control from the isolation chapter. The overflow-shrink recovery path re-cuts
  inline with `head_tail` and writes nothing either; its tail slice is floored
  at the widest footer the door can write (and its budget floored high enough
  to afford that tail at every configured size), so a pointer at the end of an
  offloaded result survives the cut whole, and it appends its own line saying
  how much of the message is left rather than leaving the door's larger counts
  standing over a smaller body. One door, three trust domains: `truncate` cuts
  text the harness or the user chose and rewrites nothing, while
  `truncate_tool_result` defangs footer lookalikes in **the copy the model
  reads** — every strategy, including under-budget results, and before the cut,
  since quoting a lookalike lengthens it. The spilled file is never defanged: it
  is the tool's own bytes, because that copy gets re-read, diffed, applied and
  hashed (a defanged `git diff`, re-applied, wrote five corrupted lines and
  reported success — an add-only hunk has no context to mismatch). The
  invariant that replaced provenance: footer-shaped text may be matched in
  order to QUOTE it, never in order to BELIEVE it. So a footer of ours that
  comes back around is quoted like any other, and there is no pattern to bound
  and no secret to keep. What the footer will not claim is completeness it
  cannot check: a result that reaches the door already carrying a
  `…[truncated]` marker (the sandbox ceiling, a tool's own paging) is labelled
  "output as captured, already truncated upstream" instead of "Full output",
  since every count is true of the file and none of them is true of the
  command. The write is atomic
  (temp file + `os.replace`, which replaces a symlink at the target name
  instead of following it out of the workspace), keeps the user's umask, is
  refused before anything is created if `.carbon` is a symlink or resolves
  outside the workspace, and never trusts a pre-existing file at the hashed
  name without verifying its content; any failure — no workspace, a bad write,
  an unencodable result — degrades to the inline excerpt and says so in the
  marker, because this runs mid-turn with `tool_calls` already in the
  transcript. Housekeeping: the directory gets a `.gitignore` (a `*` when
  absent, a covering line appended when an existing one doesn't cover the
  spills, never a clobber), holds at most 64 spills but never reclaims one
  this session's transcript still points at, and is skipped by
  `list_files`/`search_text` unless a pattern names one spill file exactly —
  the footer's own `search_text` route, judged on the resolved path so a
  symlink into the scratch directory cannot reopen it.
  `Agent(tool_output=...)` overrides the policy per instance (default `None`
  keeps the editable surface in charge) — the experiment seam, same
  philosophy as `Tool.max_result_chars`, while the config file remains the
  improvement loop's surface.
- `Agent(workspace_root=...)`: where the agent's files live — the directory
  offloaded output is written under, and the root the model's `read_file`
  resolves against. Defaults to `agents_dir`, which every wiring here binds
  to the same directory, so no existing caller changes; a consumer that loads
  `AGENTS.md` from a neutral directory while rooting `read_file` at the
  workspace now has a way to say so, instead of every footer path being a
  dead route. The truncation policy threads the same way:
  `run_once(tool_output=...)` → `_coding_tools` → `delegate`/`fan_out`, so a
  subagent — a whole Agent, with a door of its own — cuts the way its parent
  does, and a worker given no registry gets the default tools rooted where it
  offloads, so the footers it is handed name paths its own `read_file` can
  resolve. `TruncationPolicy` validates the types and the
  ranges of its budget and tail fraction wherever it is constructed (a float
  budget used to reach a string slice mid-turn; `True` is a positive integer
  that cuts every result to one character), and `Agent` rejects an
  unimplemented strategy name at construction rather than mid-session.
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
- `LLM_PROVIDER_ORDER` / `LLM_QUANTIZATION`: pin the serving base behind a
  multi-provider router (OpenRouter). `LLM_PROVIDER_ORDER` names exactly one
  upstream provider, `LLM_QUANTIZATION` one serving precision. Provider-level,
  like `LLM_REASONING_EFFORT` — which serving base answers is a property of
  the model being measured, not of one request. Forwarded as OpenRouter's own
  provider-routing object (`{"provider": {"order": [...], "allow_fallbacks":
  false, "quantizations": [...]}}`), each field only when its pin is set, and
  fallbacks disabled whenever any pin is — a pin that can silently reroute is
  not a pin. Unset means nothing new is sent: local endpoints (LM Studio,
  Ollama) see byte-identical requests.
- `compaction.prompt_suffix`: the strategy-specific tail of the summarizer's
  instructions — the checkpoint headings, the carry-forward update instruction,
  the preserve-verbatim line — exposed as an optional config knob, additive and
  default-neutral (unset or null means the strategy's built-in suffix, and the
  assembled prompt is byte-identical to before, pinned literally by test;
  `config_version` does not move for this entry). `compaction_prompt` was only
  the BASE of what the summarizer sees: each strategy appended its tail in
  code, out of the editable surface's reach, so a prompt candidate could
  rewrite the base yet never touch the headings or update instruction the
  model actually reads. The suffixes are now data on the strategy registry
  (`default_suffix`); a configured `prompt_suffix` replaces any strategy's
  default, and the empty string strips the tail entirely, leaving the base
  prompt alone. A non-string value is rejected at load, naming the field, like
  every other malformed config value.

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
- **`harness.limits` grew a second entrance and renamed its re-cut helper.**
  `truncate()` keeps its signature and now cuts without rewriting anything, which
  is what an `@path` block and a generated checkpoint need; tool output goes
  through the new `truncate_tool_result()`, which defangs footer lookalikes in
  the model's copy only. `head_tail()` — public, unvalidated, and shaped exactly
  like a way around the door — is now `recut()`, documented as the post-door
  re-cut primitive and validating its budget and tail fraction through
  `TruncationPolicy` (`head_tail(text, 10, tail_fraction=2.0)` used to return 61
  chars for a budget of 10). None of the three was ever exported from `carbon`.
- **The two retention copies of a tool result are bounded, head AND tail, at the
  door's own budget.** `subscribe()`'s `tool_call` event and — when content
  capture is on — the `execute_tool` span's `gen_ai.input.messages` /
  `gen_ai.output.messages` are the pre-door result, which nothing under the loop
  caps: a subscriber could hold a 33MB `read_file` result for the life of the
  driver. They were clamped head-only at the fixed per-item ceiling, which was
  faithful to neither side — it ignored both the configured `tool_output` budget
  and a tool's declared `max_result_chars` (so it could keep less or more than
  the model received), and dropping the tail dropped exactly the `FATAL: …` last
  line that content capture is opt-in to see. Both now keep head and tail at the
  effective budget for that result; `Tracer.record_tool()` takes the bound as a
  `max_chars` keyword (default: the per-item ceiling) and applies it to the args
  as well, so a 33MB `write_file` is no longer stored whole on the input side
  while its result is trimmed. These are documented seams (adr/0002), so this is
  an API change rather than trace-size housekeeping. Spans are in-memory only —
  `dump_events()` persists the flat `Event` list and nothing else — so the bound
  is on a live process's resident trace and exporter payload, not on a file.
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
