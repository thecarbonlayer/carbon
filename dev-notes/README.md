# dev-notes

Design decisions and the evolution backlog for carbon's library surface.

The fifteen-chapter curriculum (`ch-00`..`ch-14`) is the finished teaching
build. This directory tracks what happens after it: the embedding seam and the
editable surface that external consumers and the self-improving loop keep
pushing on.

- [adr/](adr/) — architecture decision records.
  - [0001](adr/0001-version-the-evolution-separately.md) — version the evolution on its own axis, not as new chapters.
  - [0002](adr/0002-mechanism-in-gemma-domain-in-the-consumer.md) — generic mechanism in carbon, domain and policy in the consumer.
- [sdk-seam-roadmap.md](sdk-seam-roadmap.md) — the prioritized backlog for the embedding seam, grounded in real consumers.
- [strategy-surface-roadmap.md](strategy-surface-roadmap.md) — the program for lifting hardcoded strategies into the editable surface, seam by seam (elaborates T3.1); each seam = one plan + one config bump + one suite cluster.
- [the-editable-surface-story.md](the-editable-surface-story.md) — the condensed arc behind that roadmap: why the first self-improvement win (a number) forced a redesign of the surface into layered strategy menus. Full essay in the studio.
- [generalization-audit.md](generalization-audit.md) — after v0.2.0, how much generalization is left: two new cross-consumer seams, one unadopted seam, and carbon's-own-surface corrections.
