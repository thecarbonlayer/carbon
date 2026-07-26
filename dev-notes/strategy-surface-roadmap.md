# Strategy-surface roadmap

The editable surface today is parameter knobs: a number, a string, a list.
Behind almost every one sits a hardcoded *strategy* that never got a name.
`max_item_chars: 4000` is a number, but "keep the first 4000 chars" is a
policy, frozen into code at its dumbest variant. The same shape repeats at
every seam where the harness discards, compresses, selects, or routes
information.

This document is the program for lifting those strategies into the surface,
one seam at a time. It is the concrete elaboration of
[sdk-seam-roadmap.md](sdk-seam-roadmap.md) T3.1 ("widen the versioned config
surface"), and it follows [ADR 0002](adr/0002-mechanism-in-gemma-domain-in-the-consumer.md):
carbon owns each strategy's *implementation* (mechanism); which strategy runs
at which seam, and with what params, is data in `harness_config.json`
(the consumer's choice, and the self-improving loop's knob).

## The mental model: three layers

Every knob in a harness lives in one of three layers. This is the simplest
frame we have found for explaining the work, and LangChain's deep-agents
team independently landed on the same three
(https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering):

1. **The prompt layer.** Text the model reads: the system prompt, injected
   context, summaries, nudges. Cheap to change, zero enforcement power.
   A prompt is an instruction; it cannot make anything true.
2. **The tool layer.** What the model can call and what comes back: which
   tools are exposed, their schemas and descriptions, argument validation,
   result budgets, the approval gate. This is a public API surface the
   model programs against.
3. **The middleware layer.** Hooks around model calls and tool calls that
   the model never sees as text but that shape everything it does see:
   truncation at the door, compaction, assembly order, memory recall,
   retries, loop detection, sampling params, the verification gate.

Knobs by layer, for this roadmap:

| Layer | Knobs |
|---|---|
| Prompt | `system_prompt`, `compaction_prompt`, per-policy summarize prompts (seam 1), onboarding injection text (seam 8b), nudge wording (seam 8) |
| Tools | tool exposure (seam 3), schemas and descriptions, per-tool `max_result_chars`, permission policy (T1.3 track) |
| Middleware | truncation strategies (seam 1), compaction shape and trigger (seam 2), memory recall (seam 4), sampling and phase scheduling (seam 6), assembly order (seam 7), verification reporting (seam 8), retry and loop detection (seams 8, 8b onboarding scan) |

Three questions define any proposed knob, and all three must have answers
before it enters the surface:

1. **Which layer does it live in?** (prompt, tools, middleware)
2. **What is its menu?** A bounded set of named, carbon-implemented
   strategies plus their params. Never freeform code: the editor picks,
   it does not write.
3. **Which tasks can see it?** A knob without held-in/held-out tasks that
   distinguish its menu entries is a knob the loop turns blind. No tasks,
   no knob.

## Ground rules (every seam, no exceptions)

1. **Bounded menus, never code hooks.** A seam's config field is
   `{"strategy": "<name from a fixed menu>", ...params}`. The menu is a
   vetted, carbon-implemented set. "Run arbitrary code here" is not a knob;
   it would dissolve the boundary the surface exists to draw.
2. **One seam per `config_version` bump.** Each seam lands with its own
   validation at the door, its own suite cluster, and its own re-baseline.
   Ten knobs in one bump means a regression nobody can attribute.
3. **Defaults reproduce current behavior byte-for-byte.** Landing a seam
   changes nothing until an editor turns the knob. The frozen curriculum
   (ch-00..14 episode tests, `tasks/checks.py`) must stay untouched and green.
4. **Every seam ships with miners and guards.** A strategy knob without
   held-in/held-out tasks that can tell its variants apart is a knob the
   loop will turn blind. Suite clusters live in
   `refinery` (grader far from editor), fixtures
   pinned to authoring-time constants, never live config values.
5. **A widening that can't explain itself is drift.** Candidate edits to any
   of these knobs go through the existing acceptance rule
   (`Δ_in ≥ 0, Δ_ho ≥ 0, max > 0`), per-task inspection, PR with evidence,
   human merge. Rollback is reverting one file.

## The seams

Cluster letters continue the existing suite (A–D shipped, E is truncation).

### 1. Truncation (Compress/Write) — SHIPPED IN CONFIG V3

Plan: `docs/superpowers/plans/2026-07-17-truncation-strategy-surface.md`.
Fields `file_injection` / `tool_output`; first menu `keep_head | head_tail`;
cluster E. Per-tool budgets stay as an override under the selected strategy.
`head_tail_summarize` and `offload_to_file` remain later additions, and must
ship with their own miners before entering the menu.

### 2. Compaction shape (Compress) — FIRST MENU SHIPPED IN CONFIG V3

- **Baked today:** summarize-the-middle with `keep_head=2`, `keep_tail=4`
  hardcoded in `compaction.py:compact()`, triggered only when
  `estimate_tokens > context_limit` in `agent.py:_maybe_compact`.
- **Current field:** `"compaction": {"strategy": ..., "keep_head": 2,
  "keep_tail": 4, "trigger_fraction": 1.0}`. Menu: `summarize_middle`
  (today), `drop_tool_results` (summarize only tool messages, keep
  reasoning verbatim; tool output is the bulk and the least quotable),
  `offload_middle` (write the middle to a session file, leave a pointer;
  the Write move applied to history). `trigger_fraction` makes the
  when-to-fire threshold tunable (Hermes fires at ~0.5, carbon at 1.0).
- **Shipped menu:** `summarize_middle | structured_checkpoint`. The latter
  serializes tool names, arguments, results, and prior summaries into a
  cumulative headed checkpoint. `drop_tool_results` and `offload_middle`
  remain roadmap entries.
- **Suite:** cluster A already mines compaction loss (A1 held-in, A3
  held-out), while G2 covers repeated cumulative compaction. Add future guards
  for a fact inside a *tool result* mid-history
  (separates `drop_tool_results` from `summarize_middle`), and a
  turn-count-heavy session where an early trigger_fraction wins.
- **Order note:** do this second; it reuses the strategy-interface pattern
  and half its miners already exist.

### 3. Tool exposure (Select)

- **Baked today:** `ToolRegistry.specs()` returns every registered tool,
  every turn (`tools.py`). Fine at 4 tools; the failure starts in the low
  tens.
- **Proposed field:** `"tool_exposure": {"strategy": "all"}`. Menu: `all`
  (today), `allowlist` (params: `tools: [...]`, a fixed subset),
  `query_match` (rank tool descriptions against the user turn, expose top-k;
  params `k`). A `phase_gated` variant (different sets while
  planning vs executing) waits for the orchestrator seam.
- **Suite:** cluster F. Miner: a registry padded with 30 plausible decoy
  tools (authoring-time fixtures) where the baseline agent mis-picks or
  stalls; guard: the D1/D2 calculator tasks must keep passing under any
  exposure strategy (the needed tool must never be selected away).
- **Reconciles with:** T1.4 (registry introspection) — exposure strategies
  read the same public registry surface consumers do. T2.1's new tools
  (grep/glob/ls) will triple the registry size and make this seam earn its
  keep.

### 4. Memory recall + forgetting (Select)

- **Baked today:** keyword text search over session JSON-L, top
  `memory_search_limit` hits (`memory.py:search_sessions`). No embeddings,
  no decay, no contradiction handling.
- **Proposed field:** `"memory_recall": {"strategy": "keyword", "limit": 5}`.
  Menu: `keyword` (today), `embedding` (the `feat/embedding-seam` branch),
  `hybrid` (keyword + embedding, rank-fused). A separate
  `"memory_forgetting"` field comes later: `none` (today) |
  `temporal_decay` | `supersede_on_contradiction`. Forgetting is the deeper
  knob (the deep-dive claim: the system that never forgets cannot find what
  matters), but it needs episodic state carbon doesn't track yet; don't
  smear it into the recall menu.
- **Suite:** cluster G. Miner: a fact stored under paraphrase (keyword
  misses, embedding hits); guard: an exact-identifier lookup
  (`ZQ-PASS-77KD`-style token where keyword wins and pure embedding can
  lose); held-out: a stale-vs-current conflict, two sessions apart.
- **Reconciles with:** the in-flight `feat/embedding-seam` branch IS this
  seam's `embedding` strategy; land the branch, then the menu.

### 5. Context assembly order (cache + position)

- **Baked today:** fixed assembly in `agent.py:send` — system prompt,
  delivered files, user turn; compaction note lands as a system message
  mid-history.
- **Proposed field:** `"assembly": {"strategy": "legacy"}`. Menu: `legacy`
  (today), `stable_prefix` (byte-stable system+tools prefix, volatile
  material only at the tail; the cache-discipline the deep dives call the
  first thing to get right), `context_last` (delivered files immediately
  before the user turn rather than before the whole history).
- **Suite:** cluster I. This one is measured on *cost and latency as
  tie-breakers* as much as pass rate: the runner already records turns;
  it would need token-usage capture per attempt (a runner change, flagged
  now). Behavioral miner: a delivered fact ignored when buried mid-window
  (lost-in-the-middle), passing when assembled at the tail.
- **Risk note:** touching assembly order can invalidate the verification
  gate's message-slice assumptions (`_enforce_run` reads indices); this
  seam needs the most careful regression guard against `[unverified]`
  stamps disappearing.

### 6. Sampling params (parameter knob, quick win)

Not a strategy, but a gap: `temperature` / `max_tokens` are ctor defaults
(`agent.py`, `model/client.py`), unreachable by the loop. Fold into the
surface as plain fields alongside T1.5's "model params as agent config".
No new cluster; existing clusters re-run under a candidate that changes
them. Cheap to land with any adjacent seam bump.

**Phase-scheduled upgrade (later, needs the orchestrator):** LangChain's
deep-agents work got its best Terminal Bench score from a "reasoning
sandwich" (max compute for planning and verification, less for
implementation), and its *worst* configured score from max-compute-everywhere
(53.9% vs 66.5%, timeouts). Once params are in the surface, a
`"model_params": {"strategy": "static" | "per_phase"}` menu keyed to the
orchestrator's plan/execute/verify phases is the natural second step. The
53.9% number is the standing reminder that more compute is not monotonic;
this knob ships with guards, not as a default.

### 7. Verification grading (verify)

- **Baked today:** `verify_attempts` re-prompts, then fail-closed
  `[unverified]` (`verification.py`). Single-run semantics per attempt.
- **Proposed field:** `"verification": {"strategy": "enforced_run",
  "attempts": 3}`. Menu additions are *reporting* strategies, not
  enforcement relaxations: `enforced_run` (today) and
  `enforced_run_repeated` (run the pinned command k times, report the
  fraction; pass^k semantics for flaky suites). Anything that weakens
  "a real observed run after the last mutation" is not a menu entry; the
  integrity core stays hardcoded (grader discipline, not policy).
- **Suite:** cluster J. Guard: the forged-receipt refusals (`echo`,
  `|| true`, pipes) must hold under every strategy; miner: a deliberately
  flaky seeded test where single-run verification passes dishonestly.
- **Boundary:** this is the seam where "editable" ends. The refusal logic
  itself must never enter the surface; the self-evolving deep dive's rule
  (the editor must not be able to reach its own grader) applies inside
  carbon too.

### 8. Retry and escalation (orchestrate) — BOUNDED BACKOFF SHIPPED IN CONFIG V3

- **Current:** `MAX_TOOL_STEPS` budget then hard stop
  (`agent.py:_run`); orchestrator plan/gate/retry has fixed retry counts
  (`orchestrator.py`).
- **Current field:** `"retry": {"strategy": "backoff", "max_attempts": 3,
  "base_delay_ms": 100}`. Menu: `fail_fast`, `backoff` (provider errors only),
  `escalate_subagent` (hand the failing step to a fresh subagent window
  once before giving up; the Isolate move as a retry policy).
- **Suite:** cluster H fault-injects a transient provider failure, a context
  overflow, and a permanently failing provider. The latter guards the
  five-attempt hard maximum. `escalate_subagent` remains future work.
- **Order note:** last of the behavior seams; depends on subagent plumbing
  being stable under T1.x and is the least mined by real failures so far.
- **Companion field, same seam:** `"loop_detection": {"strategy": "none" |
  "edit_count_nudge", "threshold": N}`. LangChain's LoopDetectionMiddleware
  counts per-file edits and injects a soft "reconsider your approach" nudge
  after N edits to the same file, targeting doom loops (10+ variations of a
  broken fix). carbon today has nothing between "keep going" and the hard
  `MAX_TOOL_STEPS` stop; a nudge tier is a distinct, cheap mechanism. Miner:
  a seeded bug whose obvious fix is wrong (the doom-loop bait), passing only
  when the agent changes approach.

### 8b. Session onboarding (Select, at t=0) — NEW

- **Baked today:** nothing. The agent starts with the system prompt,
  AGENTS.md if present, and whatever the user @-delivers; environment
  discovery (what files exist, what tools are installed) costs live turns
  and produces the "context discovery error" failure class.
- **Proposed field:** `"onboarding": {"strategy": "none"}`. Menu: `none`
  (today), `dir_map` (inject a bounded directory listing of the workspace at
  session start), `dir_map_toolscan` (also probe for available interpreters
  and test runners). Straight from LangChain's LocalContextMiddleware, one
  of the three levers behind their 52.8% → 66.5% Terminal Bench gain. Every
  injected block passes the truncation door (seam 1), so onboarding cannot
  flood the window.
- **Suite:** miner: a task requiring a file the agent isn't told about
  (baseline burns steps discovering it or fails); guard: onboarding output
  must never displace the user turn's own delivered context (A-cluster
  sentinels stay green).
- **Order note:** slot after tool exposure; it's low-risk, additive, and
  its win is measurable as reduced `turns` even where pass rates hold.

### 9. Permission policy (contain) — ALREADY SCHEDULED AS T1.3

The `Policy` object landed in v0.1 (`agent.py` ctor). Widening it into
declarative backends (allow/deny lists, path scope, predicate) is
sdk-roadmap T1.3 and proceeds on that track, not this one. When T1.3 lands,
revisit whether any of it belongs in the JSON surface or stays
consumer-side; ADR 0002 leans consumer-side (policy is the consumer's).

### 10. Sandbox selection (contain) — DECLINED, on record

The isolation-layer choice (scrubbed-local vs Docker vs stronger) stays out
of the surface. sdk-roadmap lists the soft sandbox as a deliberate non-gap
for a local daily driver, and a knob that can *weaken* containment is
exactly the kind of edit the editable surface must not offer to an
automated editor. Revisit only if a consumer runs untrusted code.

## Priority order

| # | Seam | Cluster | Why here |
|---|------|---------|----------|
| 1 | Truncation | E | In flight; template for the pattern |
| 2 | Compaction shape | A/G | First menu shipped; extend only with new guards |
| 3 | Tool exposure | F | T2.1 tool belt makes it urgent; guards trivial (D1/D2) |
| 4 | Session onboarding | L | Low-risk, additive; field-proven lever (LangChain); win shows up as fewer turns |
| 5 | Memory recall | G | `feat/embedding-seam` already builds strategy #2 |
| 6 | Sampling params | — | Quick win, ride along with any bump; per-phase variant waits for orchestrator |
| 7 | Assembly order | I | High leverage, but needs runner token capture + verification-gate care |
| 8 | Verification reporting | J | Valuable (pass^k honesty), sharpest integrity boundary |
| 9 | Retry/escalation + loop detection | H | Backoff shipped; escalation/loop nudge remain |

Each row is one plan (authored just-in-time against the then-current code,
the truncation plan's format), one `config_version` bump, one suite cluster
landing in the same change as its seam, one re-baseline.

## Evidence from the field

LangChain's deep-agents write-up
(https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering)
is an independent run of this same program: model frozen, harness knobs
tuned, measured on Terminal Bench 2.0 (52.8% → 66.5%, top-5). Three points
of contact worth keeping on record:

- **Convergent validation.** Their headline failure ("agents confirmed
  their own code looked acceptable without actual testing") is ch-12's
  reason to exist, and carbon's receipt-enforcing gate is stricter than
  their checklist middleware. Their Trace Analyzer loop (traces → parallel
  error analysis → cluster → targeted harness change, "like boosting") is
  the refinery mining pipeline, independently reinvented.
- **Imported knobs.** Session onboarding (8b), loop-detection nudges (8),
  and phase-scheduled model params (6) came from this post.
- **The cautionary number.** Their max-compute-everywhere run scored 53.9%,
  *below* their 63.6% mid-compute baseline, from timeouts. No knob on this
  roadmap is "obviously good"; every candidate goes through the acceptance
  rule, per-task, both piles.

## What this roadmap must never grow

The suite's task definitions, verifiers, or oracles (they live in
refinery); any consumer's judge criteria or reward shaping; a code
hook at any seam; a menu entry that weakens verification integrity or
containment. Strategies are mechanism; those are policy or grader, and they
stay where they are.
