# AGENTS.md — carbon

## Intent

This repo builds a real coding agent from scratch, one harness primitive at a time, to make a single
thesis concrete: **Agent = Model + Harness + UI.** The model rarely changes; the harness (loop,
instructions, context, tools, memory, sandbox, orchestration, subagents, verification, observability,
UI) is where the engineering and the leverage live.

The course is a **15-chapter spine** (`ch-00` … `ch-14`), each chapter introducing one primitive in its
mature form, demonstrated against a real model (Gemma via LM Studio) and proven with two gates. Each
chapter is its own commit, tagged, building cumulatively on the last.

## Working in this repo

- **Two-gate rule.** `verify` is the floor, `accept` is the truth. Ship a change only when both are
  green. Never claim a chapter works without a green `accept`.
  - `uv run verify` — ruff format + lint, mypy, pytest, smoke import (deterministic, offline)
  - `uv run accept ch-NN` — the real agent against a real model, asserting the capability
- **Chapters are cumulative and tagged.** Each `ch-NN` builds on the last. Keep changes scoped to one
  primitive per chapter; don't smear a primitive across chapters.
- **The harness owns the behavior.** When the agent misbehaves, suspect the harness (context, limits,
  tools, prompt assembly) before the model. See `harness/` for the primitive modules and the drive loop
  (`harness/agent.py`).
- **Three packages, one-way deps.** `model/` (provider seam + costing + a `fake` provider), `harness/`
  (the loop + every primitive, incl. the `events.py` OTel seam), `ui/` (the Textual TUI — the only
  package that imports textual). Dependencies point `ui/` → `harness/` → `model/`; the core never
  imports the UI. `tasks/` is dev tooling. The agent loop lives only in `harness/agent.py`; the REPL is
  the `agent` console script.
- **Real models, no mocks.** Demos and accept checks run against a live endpoint configured via `.env`
  (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`).
- Run it: `uv run agent "..."` (REPL), `uv run tui` (Textual UI), `uv run demo ch-NN` (chapter demo).

## Testing

The harness reads this command and enforces a passing run whenever it changes code:

```
uv run verify
```
