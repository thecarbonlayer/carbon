# Generalization audit — the embedding seam vs. its three consumers

_2026-07-17. Point-in-time audit against carbon v0.2.0 and the three consumers as they
stood after adopting the seam. Line numbers may drift; treat them as anchors, not addresses._

## Why this exists

After shipping the embedding seam (v0.1.0 + v0.2.0) and having the three real consumers
adopt it, the question was: **have we squeezed as much generalization as possible, without
over-reaching?** "Generalization" here means the razor from
[adr/0002](adr/0002-mechanism-in-gemma-domain-in-the-consumer.md): generic mechanism belongs
in carbon; domain and policy stay in the consumer. Under-generalization is a consumer
hand-rolling a mechanism carbon should own. Over-generalization is carbon absorbing a
consumer's domain, or shipping surface nobody uses.

## Method

Four parallel analysts, one per repo, against a shared rubric (classify every non-domain
piece into: clean adoption / missed adoption / candidate new seam / correctly-domain):

- **the analyst** — a domain agent over a stats database (the cricket-stats consumer).
- **the evals** — the eval suite that scores it.
- **the editor** — `refinery`, the measurement + self-improving-loop half that drives and
  edits carbon from outside it.
- **carbon** itself — the opposite lens: over-reach, domain leakage, unused surface, sharp edges.

(Consumers are named descriptively, per the convention in
[adr/0002](adr/0002-mechanism-in-gemma-domain-in-the-consumer.md): only `refinery` is public.)

## The seam surface being audited (v0.1.0 + v0.2.0)

`Agent.run() -> RunResult` (`text`, `tool_calls`=`ToolCall{name,args,result,is_error,attributes}`,
`turns`, `approvals`, `stop_reason`, `totals`); `Agent.send()`; `Agent.subscribe(cb)`.
`Policy(allow/deny/read_only/require_approval/approve/mutators)`.
`Tool(...,attributes,max_result_chars)`; `ToolRegistry.register/get/names/wrap/call/specs`.
`Provider.from_env(root=)`, `load_env(root)`, `chat(...,response_format=)`, `fake`.
`provenance()`, `config_schema()`, `load_config()`, `CONFIG`; `Tracer`; `default_tools`.

## Verdict

**Roughly 85% squeezed.** The seam is restrained and correct: no consumer domain leaked into
carbon, back-compat is real (`send` is a shim over `run`; the old `approve`/`approval_required`
pair still builds a `Policy`), and the `attributes`/`provenance` bags are correctly left empty
for consumers to fill. The evals repo is exemplary (zero missed adoptions). What remains falls
into three buckets: two genuinely new cross-consumer seams carbon should absorb, one flagship
seam a consumer never adopted, and a set of carbon's-own-surface corrections (including one
truth-in-advertising bug).

---

## Bucket 1 — New cross-consumer seams carbon should absorb

These are the real remaining generalization. Both are confirmed by two consumers converging
on byte-identical code, which is the strongest seam signal there is.

### 1.1 A behavior-identity fold (`behavior_key`) — STRONGEST

Both consumers hand-roll the identical construction:

- `refinery/runner/guard.py:47` — `sha256(f"{config_version}|{model}|{runner_sha}|{dirty_sha}")[:12]`
- `evals/guard.py:33` — `sha256(f"{config_version}|{model}|{dataset_fp}")[:12]`

They also each reimplement the same policy shell around it (`StaleBaseline`, `baseline_status`,
`assert_resumable`) and the same governing decision: **config_version is carbon's behavior-version
declaration; gemma_sha is provenance, not behavior, so it is excluded from the key.** That last
decision is carbon's own semantics about its own config surface, and it should not be re-derived
in each consumer. refinery's `guard.py:16-18` docstring says outright that it mirrors the
sibling's `guard.py` "so the two consumers converge on one shape instead of diverging" — i.e.
they are manually keeping two copies in sync, which is exactly the duplication a seam removes.

**Proposed seam:** carbon provides a folding primitive next to `provenance()`, e.g.
`behavior_key(model, **extra_dims) -> str`, that seeds with `CONFIG.version` + `model`, folds
in consumer-supplied extra dimensions, and applies the canonical `sha256(...)[:12]` convention
and the exclude-gemma_sha rule. The consumer keeps supplying **which** extra dimensions
(refinery: `runner_sha` + `dirty_sha`; evals: `dataset_fp`) — that stays policy.

**Razor note:** the fold, the digest convention, and the exclude-sha rule are mechanism (carbon).
The invalidation policy (what to do on a mismatch: refuse, `--force`) is borderline. Ship the
fold primitive for certain; offer the `assert_resumable`/`baseline_status` shell only as an
optional convenience, since adr/0002 assigns "no silent reuse" to the consumer.

### 1.2 Reasoning-tolerant structured output

Three separate findings collapse into one fix:

- carbon's `response_format` is threaded into **every** loop turn (`harness/agent.py:369-378`,
  stored `:74`/`:88`), so it constrains tool-selection turns that should be emitting `tool_calls`,
  and it is forwarded verbatim with no `try/except` (`model/openai_compatible.py:59-60`).
- The local LM Studio / gemma-4-26b endpoint HTTP-400s on `{"type":"json_object"}`, so
  `evals/judge.py` **tried `response_format` and reverted** (comment at
  `judge.py:41-44`), keeping its regex extraction (`judge.py:45,78-83`).
- `the analyst` independently reimplements the same idea for its persona case:
  `extract_answer` + `_JSON_BLOCK` (`analyst/agent.py:40-53`) pull the last valid JSON object out
  of in-character prose. Note this case genuinely needs prose **plus** a JSON sidecar, so full
  `response_format` (which collapses the whole message to JSON) is not a drop-in even where the
  endpoint supports it.

**Proposed seam (three parts):**
1. Scope `response_format` to the final answer turn, not every loop turn.
2. Fail soft: when the endpoint rejects json-mode, degrade to unconstrained + a parse fallback
   instead of a hard failure.
3. Ship `parse_json_reply(content) -> dict | None`, a reasoning-tolerant extractor (find the
   outermost / last syntactically-valid JSON object, tolerate surrounding prose). The `value`/
   `kind` contract and the fenced-vs-bare choice stay in the consumer.

This is the "structured output that works on a local reasoning model" story. Every consumer hit
it; the judge reverted over it.

---

## Bucket 2 — A flagship seam a consumer never adopted

**`refinery` never adopted `RunResult`.** Its refactor took `load_env`, `Provider.from_env(root=)`,
`provenance()`, and `config_schema()`, but skipped the keystone. Every task still drives with
`a.send(prompt)` (bare text) and then reconstructs the structured outcome by walking `a.messages`:

- `runner/helpers.py:60` `tool_texts`, `:65` `tool_call_args`, `:78` `bash_runs`
- `runner/helpers.py:88-119` re-derives tool-call ordering that `RunResult.tool_calls` already
  delivers in order (about 40 lines that dissolve)
- `turns = len(a.messages)` as a stand-in for `RunResult.turns` (e.g. `runner/tasks/cluster_d.py:73`)

carbon's `RunResult` docstring says exactly this: "Real consumers did exactly that reconstruction
by hand." This is a **consumer-side** adoption gap, not a carbon gap. The single-`send` tasks
(clusters B, C, D) map directly onto one `a.run()`. (Caveat: the multi-`send` tasks A1/A3 inspect
compaction/recall over `a.messages`, where `RunResult` is genuinely less relevant.)

**But it also exposed a real carbon gap.** Two tasks reach into privates because `RunResult` does
not surface what they need:

- `runner/tasks/cluster_b.py:179` reaches `a._observed_pass` — there is no public "was this turn's
  code change verified?" verdict on `RunResult`.
- `runner/tasks/cluster_a.py:77` reaches `a._last_tokens` — there is no token/window usage on
  `RunResult` (only the tracer `totals`, which is cumulative).

Adding a `verified` verdict and per-run usage to `RunResult` closes both reaches. That is generic:
any consumer that gates on "did the harness verify the change" or measures window pressure wants it.

Related: **façade under-use.** refinery imports `from harness.agent import Agent`,
`from harness.tools import Tool, ToolRegistry` instead of `from carbon import ...` for symbols the
façade already exports. This is minor, but it also revealed a **façade-coverage gap**: `Workspace`,
`Sandbox`, `DEFAULT_SYSTEM`, `APPROVAL_TOOLS`, and `write_file_tool` are **not** in the `carbon`
façade, so those internal imports are unavoidable today. Worth deciding whether the façade should
export them.

---

## Bucket 3 — Tighten carbon's own surface (truth, not width)

### 3.1 The v0.1.0 CHANGELOG over-promises surface that does not exist — MUST FIX

- `CHANGELOG.md:45` and `sdk-seam-roadmap.md:38` advertise Policy "path scope" and "predicate".
  The shipped `Policy` (`harness/policy.py:30-36`) has only `require_approval, allow, deny,
  read_only, approve, mutators`. No path-scope field, no predicate hook.
- `CHANGELOG.md:38` / roadmap list `get/wrap/override/list` on the registry. There is no `override`
  method (`harness/tools.py`); overriding happens implicitly via `register()` re-assigning a name.
- `sdk-seam-roadmap.md:42-44` says `subscribe` lets a driver "gate mid-run." It cannot:
  `_emit` fires **after** `self.tools.call` executes (`harness/agent.py:412` runs the tool, `:418`
  emits), so a subscriber only observes; gating is `Policy`'s job. (The `subscribe` docstring itself
  is careful and says "reads or reacts"; only the roadmap over-claims.)

A versioned SDK's changelog cannot describe surface that is not there. Either implement these
(path-scope + predicate are plausibly useful) or strike them from the released-notes language.
This is a correctness debt introduced by the seam work, and it is the cheapest, highest-integrity
item on the list.

### 3.2 `subscribe` is unused by all three consumers

`harness/agent.py:122,418`. It is a third observation mechanism alongside `Tracer` and the
`events.py` OTel seam, its `tool_call` event duplicates what `tracer.record_tool` captures one
line above, and no consumer subscribes to it. It is also inconsistent: the event carries the
untruncated `result` (`agent.py:423`) while `RunResult.tool_calls` carries the truncated content
(`agent.py:346` reads back `clamp(result[,budget])`). Decision: drop it, or fix its payload and
correct its docs. Do not keep speculative surface "just in case."

### 3.3 Minor own-surface items

- `DEFAULT_MUTATORS = {write_file, edit_file, bash}` (`harness/policy.py:23`) are carbon's own
  built-in names, so it is legitimate mechanism, but `read_only=True` silently protects only
  agents that use those names (a consumer whose mutator is `db_write` gets no protection and no
  warning). Mitigated: `Policy.mutators` is overridable. Also, `default_tools()` registers only
  `calculator` + `read_file`, so `DEFAULT_MUTATORS` names tools absent from the default registry.
- `is_error` reconstruction drops the "denied" signal: a policy-denied call (marker `[denied…]`)
  lands in `RunResult` as `is_error=False` with no distinct denied signal (`harness/agent.py`
  `_collect_tool_calls` recomputes `is_error` from `res.startswith("error")` rather than reusing
  the loop's `status`).
- `max_result_chars=0` is treated as "unset" and falls back to the global clamp
  (`harness/agent.py:430`); the `if budget else` fork is redundant.
- No `register`/`override` guardrail: a hand-built `Tool(...)` re-registered under an existing name
  silently drops any `attributes`/`max_result_chars` the prior tool carried (`harness/tools.py:98`).
  This is the exact footgun `the evals` avoided by using `wrap()`; a guardrail would warn
  the next consumer that reconstructs instead of wraps.

---

## Smaller / single-consumer items

- **the analyst's `_tool_status` missed adoption:** its error branch (`analyst/agent.py:56-63`, consumed at
  `_trace_from` `:79`) string-sniffs `result.lower().startswith("error")` instead of reading the
  `ToolCall.is_error` carbon already computes. The "empty" branch is domain and stays.
- **`provenance()` could self-locate carbon's checkout.** Both `evals/runner.py:17`
  and `refinery/runner/carbon_env.py` hardcode `the carbon checkout` because `provenance(root=".")`
  defaults to the consumer's cwd (the wrong repo). carbon can resolve its own root from
  `Path(__file__)` as the default, keeping `root=` for pointing at a different checkout (which
  refinery genuinely needs, since it measures a specific carbon working tree).
- **AGENTS.md auto-load opt-out.** the analyst uses a `_NO_AGENTS_DIR` workaround (`analyst/agent.py:16-18,32`)
  because `_system_text` unconditionally calls `load_agents_md(self.agents_dir)`. A library-embedding
  consumer wants a clean system prompt; `agents_dir=None` (or `load_agents_md=False`) meaning "do not
  auto-load project instructions" would retire the workaround. Single-consumer signal, but clearly
  generic to any library embedder.
- **Checkout identity with a dirty-tree hash — DEFER.** refinery hand-shells
  `rev-parse HEAD` + `status --porcelain` + `diff HEAD` to build a full sha and a `dirty_sha`
  (`runner/carbon_env.py:41,93-102`). Generic, but only refinery needs it today (the analyst is
  content with provenance's short sha). A `provenance(..., include_dirty=True)` option would retire
  the raw-git block. Record and defer until a second consumer asks.

---

## What we got right (the balance)

- No cricket / eval / editor concept made it into carbon. adr/0002 held.
- Back-compat is real and verified: `send` shims `run`; the legacy gate args still build a `Policy`
  with the identical `[denied by approval gate]` marker; the `fake` provider swallows the new
  `response_format` kwarg, so offline `verify` is unaffected.
- `the evals` has zero missed adoptions and the exemplary `ToolRegistry.wrap` usage that
  preserves v0.2 metadata through fault injection.
- The `provenance()` shape correctly stops at three fields and lets consumers layer their own
  (`dirty_sha`, `runner_sha`, behavior key) rather than absorbing that policy.

---

## Prioritized next moves

| # | Item | Where | Effort | Kind |
|---|------|-------|--------|------|
| 1 | Fix CHANGELOG/roadmap over-promises (implement or strike path-scope / predicate / `override` / subscribe-"gates") | carbon | S | correctness debt |
| 2 | `behavior_key(model, **extra)` fold (+ optional resume shell) | carbon (v0.3.0) | M | new seam |
| 3 | `response_format` → final-turn + fail-soft, and `parse_json_reply()` helper | carbon (v0.3.0) | M | new seam + sharp-edge |
| 4 | Add `verified` verdict + per-run usage to `RunResult`; then adopt `RunResult` in refinery | carbon + refinery | M | seam gap + missed adoption |
| 5 | Decide `subscribe`'s fate; minors (denied signal, `max_result_chars=0`, register guardrail, DEFAULT_MUTATORS note) | carbon | S | own-surface tidy |
| 6 | the analyst `is_error` adoption; `provenance()` self-locate; AGENTS.md opt-out; façade coverage (`Workspace`/`Sandbox`/…) | mixed | S | polish |
| 7 | `provenance(include_dirty=)` | carbon | S | deferred until 2nd consumer |

Items 2 and 3 are the real new generalization. Item 1 is a bug introduced by the seam work and
should land first. Item 4 is the largest unadopted-seam gap.
