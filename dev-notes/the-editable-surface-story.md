# The editable surface story (condensed)

The full essay lives in the studio
(`thecarbonlayer/deep-dives/editable-surface/essay.md`). This is the
engineering-notes version: the arc in one page, so the reasoning behind
[strategy-surface-roadmap.md](strategy-surface-roadmap.md) survives context
resets, new contributors, and time.

## The arc

1. **We built the harness ourselves.** Fifteen chapters, one primitive each,
   two gates (`verify` offline, `accept` live). The point of building from
   scratch is that someone on the team knows where every seam is.
2. **We built a self-improvement loop around it** (refinery, separate
   repo): 23 tasks in clusters, held-in shown to the proposer, held-out
   hidden, repeated attempts averaged as fractions, acceptance rule
   `Δ_in ≥ 0, Δ_ho ≥ 0, max > 0`, evidence in a PR, human merges. Grader and
   editor never share a home.
3. **The first win exposed the ceiling.** The winning candidate raised
   `max_item_chars` 4000 → 12000. It passed both piles. It was also the
   wrong kind of fix: it moved a cliff instead of changing the cutting
   strategy. The right fixes (keep both ends, summarize the middle, offload
   to disk) were code, and code is off-limits to the editor by design.
   Lesson: **the acceptance rule is necessary, not sufficient. Passing
   held-in and held-out proves a change made nothing worse where we
   measured; it cannot prove the change was the right one if the right one
   is not expressible on the surface.**
4. **So the surface itself got redesigned.** The builder walks every seam
   where the harness discards, compresses, selects, or routes information,
   names the strategy currently baked in, and lifts it into the surface as
   a bounded menu: `{"strategy": "<name from a fixed menu>", ...params}`.
   The scalar that used to be the whole knob becomes a parameter of the
   chosen strategy. Never a code hook; the editor picks, it does not write.
5. **Every knob answers three questions before it enters the surface:**
   which layer it lives in (prompt / tools / middleware), what its menu is,
   and which held-in/held-out tasks can tell its menu entries apart. No
   tasks, no knob. See the roadmap's "mental model" section for the layer
   mapping and per-seam clusters.
6. **Correctness is not a menu.** Unique atomic edits, tool-argument
   validation, incomplete-response refusal, worker workspace identity,
   verification freshness, and containment are published as immutable
   invariants. Refinery reads that do-not-propose list from
   `carbon.surface_manifest()` before authoring candidates.

## Standing reminders

- Read per-task numbers, never just the aggregate; a real regression once
  nearly shipped behind an unrelated flaky task's uptick.
- One seam per `config_version` bump, one cluster, one re-baseline; that is
  what keeps deltas attributable and rollback meaningful.
- External corroboration: LangChain's deep-agents post uses the same three
  layers and reports 52.8% → 66.5% on Terminal Bench from harness changes
  alone, plus a 53.9% all-max-compute run that proves no knob is
  "obviously good".
