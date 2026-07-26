# ADR 0001 — Version the evolution on its own axis, not as new chapters

## Status

Accepted.

## Context

The course is a finished fifteen-chapter spine, `ch-00` through `ch-14`, each
tag introducing one primitive in its mature form. That spine is a teaching
artifact. It is meant to end.

Two forces now push past the end of the spine:

1. **External consumers.** The harness is imported as a library to build
   domain-specific agents (a cricket-stats analyst, an external tuner of the
   harness itself, an eval suite). They need a stable surface to build on.
2. **A self-improving loop.** An external editor proposes edits to the
   harness's own configuration and measures the result. The best fix that loop
   can find is bounded by how wide the editable surface is. A recent proposed
   fix was rejected not because the loop reasoned poorly, but because the right
   knob did not exist yet. When the loop's best candidate is unsatisfying, that
   is the surface telling us where to widen it next.

So the surface will keep widening, on demand, with no fixed order and no end.
Bolting `ch-15`, `ch-16`, and onward onto the curriculum would misrepresent
this twice: it would imply a finished lesson where there is ongoing
maintenance, and a fixed sequence where the order is driven by need.

## Decision

Track two orthogonal axes.

- **Curriculum.** Frozen. Tagged `ch-00`..`ch-14`. It does not grow.
- **Evolution.** The library surface and the editable surface, versioned with
  semantic versions on the `carbon` package (`v0.x`), recorded in
  [CHANGELOG.md](../../CHANGELOG.md), with the configuration's own integer
  `version` field as the fine-grained counter underneath.

Conventions:

- Commit scope names the line: `feat(surface): ...` widens the knobs the loop
  can turn, `feat(sdk): ...` grows the API external consumers import.
- One CHANGELOG entry per release, grouped `Added` / `Changed`, roughly one
  bullet per consumer-visible change. Commits stay fine-grained; the release
  entry rolls them up.

## Consequences

- The chapter tags stay legible and finished. A follower checks out `ch-NN` to
  learn a primitive and reads the CHANGELOG plus `v0.x` to track what changed.
  The two never collide.
- The surface widens as a feedback response to the loop hitting its ceiling,
  not on a schedule.
- Additions are additive by default, so consumers are not forced to migrate on
  each release (see [0002](0002-mechanism-in-gemma-domain-in-the-consumer.md)).
- A release that changes behavior invalidates pinned eval baselines by design;
  the eval suite re-baselines against the new harness state.
