# SDK seam roadmap

The prioritized backlog for carbon's embedding seam: the surface external code
uses to build domain-specific agents on the harness. This is a living document,
not a chapter (see [ADR 0001](adr/0001-version-the-evolution-separately.md)). It
widens as consumers and the self-improving loop hit the seam's ceiling.

Governing razor (see [ADR 0002](adr/0002-mechanism-in-gemma-domain-in-the-consumer.md)):
**generic mechanism goes in carbon; domain and policy stay in the consumer.**

## Where the requirements came from

Three real consumers already import carbon and each hand-built the same generic
scaffolding. That reinvention is the specification.

| Consumer hand-built this | Because carbon lacks | Requirement |
|---|---|---|
| a result object over `send`; reads `turns` and `approvals` per run | a structured run result | T1.1 |
| JSON pulled out of prose in three places | schema-constrained output | T1.2 |
| read-only guards inside every tool; a fake config dir to disarm the gate | per-agent permission policy | T1.3 |
| reaching into a private `_tools` dict to wrap a tool; a self-describing trace | registry introspection + attribute bag | T1.4 |
| a second `.env` loader; explicit model params passed to `chat` | env bootstrap + model params | T1.5 |
| editing the config file and re-validating through `load_config`, hardcoding fields | a config schema export | T1.6 |
| git-shelling for the harness state that produced a result | provenance primitives | folded into T1.5 |
| per-tool truncation of results | tool-level result budgets | T2.3 |

## Tier 1 — the generic embedding seam

Six of these are additive. Only T1.3 changes how the loop decides.

- **T1.1 Structured run result.** `Agent.run()` returns an object: final text,
  tool calls (`name`, `args`, `result`, `is_error`, `attributes`), totals,
  turns, approvals count, stop reason. `Agent.send()` stays returning the final
  text so existing consumers are untouched.
- **T1.2 Schema output.** An opt-in "return JSON matching this schema" mode on
  `chat()` and `Agent`, so prose-to-JSON extraction disappears from consumers.
- **T1.3 Tools plus permission policy.** A declarative policy bound to an agent
  (allow, deny, read-only, path scope, predicate hook), consulted by the gate.
  The interactive human prompt becomes one policy backend among several. Also
  arms subagents to act under a real policy. This is the one behavior change.
- **T1.4 Registry introspection, attribute bag, subscribe.** Public
  get/wrap/override/list on the registry; a per-call `attributes` bag and an
  `is_error` flag; a `subscribe(callback)` event stream so a driver observes and
  gates mid-run instead of parsing messages afterward.
- **T1.5 Env, model params, provenance.** `Provider.from_env(root=)` and a
  public `load_env`; model params as agent config, not call-site constants; a
  small `provenance()` returning the harness-identity primitives
  (`config_version`, model, checkout sha). Consumers compose their own
  fingerprint on top.
- **T1.6 Config schema introspection.** Expose the config's fields, types, and
  bounds (derived from the frozen dataclass and its validation, so it cannot
  drift), and keep `load_config` a stable public door. Lets an external editor
  mutate the surface without hardcoding it.
- **T1.7 A curated `carbon` package.** A thin `carbon/__init__` re-exporting
  exactly the SDK surface, versioned with semantic versions. Internal module
  paths keep working, so migration is opt-in.

## Tier 2 — worker competence (needed once a persona writes code)

- **T2.1** Code navigation tools: `grep`, `glob`/`ls`, offset and limit on file reads.
- **T2.2** Fail-loud atomic edit (unique match), instead of first-occurrence replace.
- **T2.3** Per-tool truncation with continuation hints, instead of a blunt global clamp.
- **T2.4** Retry with backoff on model errors.
- **T2.5** Interrupted-tool-call transcript repair, for killed and resumed runs.

## Tier 3 — strategic

- **T3.1** Widen the versioned config surface (model params, sandbox limits, timeouts).
- **T3.2** Persona files and per-project config resolution. Daily-driver convenience only; a meta-harness composes personas in Python.
- **T3.3** A deterministic workflow API versus the LLM orchestrator. Deferred.
- **T3.4** An MCP client.
- **T3.5** A branching session tree.

## Deliberate non-gaps (do not "fix")

- The soft sandbox. For a local daily driver the machine is already the trust boundary.
- OpenAI-compatible providers only. The right coverage for LM Studio, Ollama, and OpenRouter.
- Shallow cost accounting. Local inference is free.

## What must never migrate into carbon

A consumer's tool taxonomy and database guards, a judge's criteria, checkpoint
weights, eval integrity rules, and any propose-and-validate or release workflow.
These are domain and policy. carbon provides the seams they hang on, nothing more.

## Suggested first slice

T1.1 through T1.4 form one coherent release. They pay off immediately: the
consumers' hand-built result objects, trace reconstruction, prose-to-JSON
extraction, and private registry wrapping all collapse into thin reads of
generic seams. T1.5 through T1.7 (env, provenance, config schema, packaging) are
a natural second release. The Tier 2 tool belt is the third, needed the moment a
persona writes code rather than answers questions.
