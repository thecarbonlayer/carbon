# ADR 0002 — Generic mechanism in carbon, domain and policy in the consumer

## Status

Accepted.

## Context

carbon is a substrate for building custom domain-specific agents. Real consumers
already exist: a cricket-stats analyst, an eval suite that scores it, and an
external editor that tunes the harness. Reading their code shows a clean
pattern. Each one hand-built the same generic scaffolding around carbon:

- a structured result object over `Agent.send` (which returns a bare string),
- a self-describing trace of tool calls with attributes,
- extraction of structured data from free-form model prose,
- wrapping a tool to inject a fault (by reaching into a private registry dict),
- loading the model environment from a foreign working directory,
- a fingerprint of which harness state produced a result.

Meanwhile each consumer's domain logic stayed, correctly, in the consumer: the
cricket tool taxonomy, the read-only database guards, the judge's quality
criteria, the checkpoint weights, the eval's integrity rules, the editor's
propose-and-validate loop.

The risk this creates is obvious. To "help" a consumer, we could pull its
subject matter into carbon and bloat the substrate.

## Decision

Generic mechanism goes in carbon. Domain and policy stay in the consumer.

A change belongs in carbon only if it is a seam that more than one consumer would
hang different domain logic on. Concretely, carbon grows: structured results,
tool wrap and override, a per-call attribute bag, permission-policy binding,
config schema introspection, environment and provenance primitives, a schema
output mode.

The following must never migrate into carbon, because they are subject matter,
not mechanism: a consumer's tool taxonomy or database guards, a judge's
criteria, checkpoint weights and reward shaping, eval integrity rules, and any
consumer's propose-and-validate or release workflow.

## Consequences

- carbon stays a thin, legible substrate. Consumers own their domain. The
  embedding seam is the contract between them.
- Every item on the [SDK seam roadmap](../sdk-seam-roadmap.md) is a seam, not a
  feature for one consumer. If a proposal encodes something only one consumer
  wants, it is a consumer change.
- Consumers can delete their hand-built scaffolding as carbon grows the seam, at
  their own pace. Nothing forces the deletion.
