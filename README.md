# Build a Custom Agent Harness, from scratch

A coding agent, built in Python from the ground up, one harness primitive at a time. Each chapter
adds a single primitive, shows the capability it unlocks, and proves it against a real model. Nothing
is mocked.

> **Agent = Model + Harness + UI.** The model barely changes. The harness is the work.

## The thesis

The model is the part that almost never changes. The harness is everything around it: the loop, the
instructions, the context it sees, the tools it can call, where they run, what survives a crash, how
work is verified, and the UI that makes all of it visible. The harness is where the engineering lives,
and it is the reason two agents running the same model can behave completely differently. When an agent
"gets dumb" mid-task, it is almost always a harness problem, not a model problem.

This repo proves the point by building the harness from scratch. Each chapter adds one primitive, shows
it failing without that primitive, then shows it working with it, and runs the result against a real
model (Gemma via LM Studio locally). A chapter is not done until the agent actually does the thing.

## Who it is for

Engineers who want to understand how coding agents work by building one, not by reading a framework's
docs. The audience is comfortable with Python but new to harness engineering as a discipline. The goal
is intuition you can carry to any agent stack, not familiarity with one library.

## Setup

```bash
uv sync                       # create the env, install deps
cp .env.example .env          # point at your model endpoint (LM Studio / OpenRouter / Ollama / ...)
```

The agent talks to any OpenAI-compatible endpoint via the `model/` package (the provider seam).
Defaults target a local LM Studio server; set `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` to use
anything else.

```bash
uv run agent                 # the interactive REPL (replies stream token by token)
uv run agent "your prompt"   # one-shot, non-interactive — runs a single turn, then exits
uv run tui                   # the Textual TUI (ch-14) — tokens stream into a live block
uv run demo ch-NN            # the demo for a chapter
```

Print mode (a one-shot with a prompt argument) takes `--format {plain,json,transcript}`:
`plain` streams the answer to stdout, `json` emits a machine-readable object (reply +
trace totals), `transcript` shows every message and tool step. It is fail-closed on the
approval gate — bash/write/edit are denied unless you pass `-y/--yes`.

```bash
uv run agent "explain this repo" --format json      # scriptable / CI-friendly
uv run agent "fix the failing test" --yes           # let it run the gated tools
```

## The chapters

The course is 15 chapters (`ch-00` … `ch-14`), each introducing one harness primitive in its mature
form.

| Ch | Title | Primitive |
|----|-------|-----------|
| 00 | What is an agent? | The frame: Model + Harness + UI (theory, no code) |
| 01 | Model only | A single model call behind a thin provider seam |
| 02 | History | Conversation state persists across turns |
| 03 | Instructions | System prompt + auto-loaded project files (AGENTS.md) |
| 04 | Context delivery | `@path` references inject file content the model can't read itself |
| 05 | Tools | The tool interface + file tools + the approval gate |
| 06 | Context management | Cache-stable assembly + compaction + door caps |
| 07 | Skills | Advertise, then load `SKILL.md` procedures on demand |
| 08 | Execution environment | Run commands in a hardened sandbox |
| 09 | Durable state + memory | JSON-L sessions on disk + episodic search |
| 10 | Orchestration | Plan steps, gate each, execute, retry |
| 11 | Subagents | Spawn isolated agents; fan out; return answers, not transcripts |
| 12 | Verification | Run candidate code against an oracle; self-verify |
| 13 | Observability | Trace every LLM/tool call as an OTel `gen_ai.*` span tree |
| 14 | UI: Textual TUI | Transcript, live trace tree, approval modal |

The build deliberately starts at `ch-00`, with the question most material skips: what separates an
agent from a chatbot or a script, and why the harness, not the model, is where the leverage is. Only
after that grounding does the build begin.

## The primitive → module map

Each primitive maps to the module that owns it.

| Primitive | Module | What it does |
|-----------|--------|--------------|
| Provider seam | `model/provider.py`, `model/openai_compatible.py`, `model/client.py` | `Provider` + a free `chat()` for any OpenAI-compatible model |
| Fake provider | `model/fake.py` | A deterministic, offline provider behind the same seam |
| Cost tracking | `model/pricing.py` | Map model ids to rates; show tokens and dollars |
| History | `harness/agent.py` | The drive loop: model, tool calls, model, with persisted messages |
| Instructions | `harness/instructions.py` | Auto-load `AGENTS.md` onto the system prompt |
| Context delivery | `harness/context.py` | Inject `@path` file content the model cannot read itself |
| Context management | `harness/compaction.py`, `harness/limits.py` | Compact the middle past a budget; clamp per-item sizes at the door |
| Tools | `harness/tools.py`, `harness/workspace.py` | A tool registry; file read/write/edit over a scoped workspace |
| Skills | `harness/skills.py` | Load `SKILL.md` procedures; progressive disclosure |
| Execution environment | `harness/sandbox.py` | Run commands in hardened Docker or a scrubbed local subprocess |
| Durable state + memory | `harness/memory.py` | JSON-L sessions on disk; keyword search across past sessions |
| Orchestration | `harness/orchestrator.py` | Plan steps, gate each, execute, retry |
| Subagents | `harness/subagents.py` | Spawn isolated agents; fan out; return answers, not transcripts |
| Verification | `harness/verification.py` | Run candidate code against an assertion; return proof, not trust |
| Observability | `harness/observability.py`, `harness/events.py` | A flat trace plus an OpenTelemetry `gen_ai.*` span tree, with an exporter seam |
| UI | `ui/tui.py` | A Textual TUI: transcript, live trace tree, approval modal |

## Beyond the curriculum: a living surface

The fifteen chapters are a finished build. They end at `ch-14`, and each tag
teaches one primitive in its mature form. That spine does not grow.

What grows is the surface around it. The harness now has external consumers that
import it as a library to build their own domain-specific agents, and a
self-improving loop that proposes edits to its own configuration. That surface is
now a set of bounded Carbon-owned strategy menus, not merely scalar limits:
file and tool-output retention, compaction shape and trigger, completion budget,
and sampling policy. `carbon.surface_manifest()` publishes both the selectable
menu and the explicit immutable invariants. The loop can pick vetted mechanisms;
it cannot inject code, weaken verification, relax workspace boundaries, or make
ambiguous edits legal.

The tool belt grows the same deliberate way. `apply_patch` gives the same
exact-match-or-refuse, atomic-write guarantee `edit_file` already gives for one
location, extended to a change spanning several files or several locations in
one file — validated everywhere before anything is written anywhere. See
[CHANGELOG.md](CHANGELOG.md) for the full surface, chapter by chapter.

To keep these two rhythms honest, the repo versions them on separate axes. The
curriculum stays frozen and tagged `ch-00` through `ch-14`. The library and
editable surface evolve under semantic versions on the `carbon` package, recorded
in [CHANGELOG.md](CHANGELOG.md), with the configuration's own `version` field as
the fine-grained counter underneath.

Two rules govern every addition: generic mechanism lives in the harness while
domain and policy live in the consumer, and no knob becomes editable until an
external miner and guard can distinguish its choices. The reasoning behind these
decisions lives in [dev-notes/](dev-notes/).

The library surface also grows through extensions: a Python file under
`~/.carbon/extensions/` or a project's `.carbon/extensions/` exposing
`setup(registry: ToolRegistry) -> None`, which registers new tools or wraps
existing ones through the same `ToolRegistry` seam every consumer already
uses (`harness/extensions.py`; see
[dev-notes/adr/0003](dev-notes/adr/0003-extensions-tools-only-outside-the-editable-surface.md)).
Extensions load new code, so they sit outside `surface_manifest()` on
purpose — the self-improving loop can pick from the config door's bounded
menus, but it cannot point Carbon at an extensions directory, because no
such field exists for it to find. Loading is also off by default at
runtime: pass `--extensions` to opt in, in print mode, the REPL, and the
TUI alike. The project-local directory sits inside the agent's own writable
workspace, so without the flag, the agent's own `write_file` tool could
otherwise plant a file there that auto-runs, unsandboxed and with no
re-approval, on every future invocation. To try the shipped example in
print mode, the only entrypoint where the project-local directory is the
real checkout's (the REPL and TUI both run in an ephemeral git worktree, so
use `~/.carbon/extensions/` there instead):
`mkdir -p .carbon/extensions && cp extensions/audit_log.py .carbon/extensions/ && uv run agent --extensions "list the tools you have"`.

## How it is built (and verified)

Two gates, because "the tests pass" and "the agent actually works" are different claims.

- **`uv run verify`** is the floor: ruff (format + lint), mypy, pytest, smoke import. Deterministic,
  offline.
- **`uv run accept ch-NN`** is the truth: the real agent against a real model, asserting the chapter's
  capability end to end. A chapter is not done until the agent can really do the thing on a real model.

Each chapter is its own commit, tagged `ch-00` … `ch-14`. Check one out to see the project as it stood
at that point in the build:

```bash
git checkout ch-05     # the project at chapter 5 (Tools)
git checkout main      # back to the latest
```

## Layout

```
model/          # the provider seam + costing: provider / openai_compatible / fake / client / pricing
harness/        # the loop + every primitive: agent, context, tools, memory, skills, sandbox,
                #   orchestrator, subagents, verification, observability, events, ...
ui/             # the Textual TUI (the only package that imports textual/rich)
tasks/          # uv-run tooling: verify / accept / demo / tui
tests/episodes/ # one behavioral test file per chapter (test_ch01..test_ch14)
```

The code is three packages with dependencies pointing one way: `ui/` → `harness/` → `model/`. The core
never imports the UI. `model/` talks to a model, `harness/` is the loop and every primitive, `ui/`
renders it. The agent loop lives only in `harness/agent.py`; the REPL is the `agent` console script.

## Why it's built this way

The architecture is part of the lesson. A few principles run through every chapter:

**Boundaries that point one direction.** The model, the harness, and the UI depend in a straight line,
and the core never reaches up to the UI. That single rule is what lets you swap the model or replace the
interface without touching the loop. When the boundary is honest in the code, the thesis stops being a
slogan.

**Every seam is built to be faked.** The places where the agent meets the outside world, the model, the
trace exporter, the sandbox, are shaped so a test can stand in for the real thing with no network. We
optimized for what is easy to swap and verify, not for what looks elegant in a diagram. A seam you can
fake is a seam you can trust.

**Observability you can read, not import.** The trace is written to the OpenTelemetry conventions by
hand, so the standard is something you learn rather than a dependency you pull in. The same trace runs
offline for tests and exports to a real backend when you want it. Capturing prompts and results is
opt-in, off by default.

**The sandbox is soft on purpose; verification is not.** These are two different trust claims and the
code treats them differently. The execution boundary has a real containment path (hardened Docker:
network-none, non-root, cap-drop, memory and pid limits) and a *teaching-grade* local fallback that
scrubs the environment and confines the cwd but does not isolate the filesystem — and the docstrings say
so, rather than implying a guarantee the fallback can't keep. Building a production sandbox (gVisor, a
microVM, container-by-default) would bury the lesson under infrastructure, so the course names the limit
instead of hiding it. Verification integrity is the opposite: it is the thesis, so it is hard. The gate
that decides "did the tests pass" refuses forged receipts (an `echo` of the command, a `|| true`
wrapper), requires the passing run to come *after* the last code change, survives a mid-run compaction,
and fails closed — an unverified change is marked unverified, never returned as clean.

**Two gates, because "tests pass" is weaker than "it works."** A stub that returns the right shape
proves nothing about a real model. So every chapter passes an offline check and then does the thing for
real against a live model. The second gate is slower on purpose: the repo never claims a capability it
has not shown end to end.

## License

MIT. See [LICENSE](LICENSE).
