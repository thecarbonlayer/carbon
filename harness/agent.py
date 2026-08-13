"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-13 — Observability. The agent has done real work for many chapters; now we can
finally *see* it. A ``Tracer`` (``harness/observability.py``) threads through the
loop and records every step: each model call with its tokens, latency, finish
reason, and cost; each tool call with its arguments, result, and status; each
verification with pass/fail. The turn is wrapped in one parent span so the whole
run reads as a tree (OTel GenAI semantic conventions, in ``harness/events.py``).

The seam is deliberate. The default ``Tracer`` is silent and offline (a
``NullExporter``), so ``verify`` stays deterministic; drop in an OTLP-backed
exporter and the same spans flow to Jaeger/Honeycomb. Cost comes from
``model/pricing.py``. Trace persistence, dormant in ``memory.py`` since durable
state landed, fires now: a resumed session restores its trace too. The hooks are
additive — pass no tracer and the loop runs exactly as before.

Everything from ch-12 is unchanged: the enforced-run verification gate (the model
runs the test with bash; the harness will not accept "done" without an observed
passing run), subagents/fan-out, durable sessions, the hardened sandbox,
usage-based compaction, skills, the approval gate, and ``@path`` injection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness.compaction import compact, estimate_tokens
from harness.context import deliver
from harness.harness_config import CONFIG, RetryPolicy
from harness.instructions import load_agents_md, test_command
from harness.limits import truncate
from harness.memory import DEFAULT_DIR, load_session, load_trace, save_session, save_trace
from harness.observability import Tracer
from harness.policy import Policy
from harness.result import RunResult, ToolCall
from harness.skills import Skill, skills_prompt
from harness.tools import ToolRegistry
from model import LLMResponse, OnDelta, Provider, chat

# Behavioral knobs live in the editable surface (harness/harness_config.json);
# these names are pure re-exports so existing imports keep working.
DEFAULT_SYSTEM = CONFIG.system_prompt
MAX_TOOL_STEPS = CONFIG.max_tool_steps
DEFAULT_CONTEXT_LIMIT = CONFIG.default_context_limit  # ~tokens; compact above this
APPROVAL_TOOLS = CONFIG.approval_tools  # tools the gate guards
# a write/edit of one of these arms the test gate (a code change to verify)
CODE_EXTENSIONS = CONFIG.code_extensions


# --- retry strategies, registry-shaped to match compaction.py and limits.py --------
#
# `fail_fast` has no behavior of its own — it IS the absence of `backoff`'s — so a
# registry is not needed for either function to stay simple. It exists anyway so
# every strategy-shaped config knob (compaction, tool_output/file_injection, retry)
# looks up its dispatch the same way, rather than three different shapes plus a note
# explaining why this one is an exception.
def _backoff_delay(attempt: int, policy: RetryPolicy) -> float | None:
    """Delay in seconds before the NEXT try, or None if there is no next try.

    `attempt` is 1-indexed and incremented before this is consulted, so
    `max_attempts` bounds total tries, not retries: with max_attempts=3, a delay is
    offered at attempt=1 and attempt=2 (two retries), refused at attempt=3 — three
    tries total, matching the field's name.
    """
    if attempt >= policy.max_attempts:
        return None
    return policy.base_delay_ms * (2 ** (attempt - 1)) / 1000


def _fail_fast_delay(_attempt: int, _policy: RetryPolicy) -> float | None:
    return None


@dataclass(frozen=True)
class _RetryStrategy:
    next_delay: Callable[[int, RetryPolicy], float | None]


_RETRY_STRATEGIES: dict[str, _RetryStrategy] = {
    "backoff": _RetryStrategy(_backoff_delay),
    "fail_fast": _RetryStrategy(_fail_fast_delay),
}


class Agent:
    """A model wrapped in memory, a system prompt, context delivery, and tools."""

    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        system: str | None = None,
        agents_dir: str = ".",
        tools: ToolRegistry | None = None,
        approve: Callable[[str, str], bool] | None = None,
        approval_required: frozenset[str] | set[str] | None = None,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        skills: list[Skill] | None = None,
        session: str | None = None,
        sessions_dir: str = DEFAULT_DIR,
        max_tool_steps: int | None = None,
        deadline_s: float | None = None,
        verify_attempts: int = CONFIG.verify_attempts,
        require_run: bool = CONFIG.require_run,
        tracer: Tracer | None = None,
        temperature: float | None = CONFIG.temperature,
        max_tokens: int = CONFIG.max_tokens,
        response_format: dict | None = None,
        policy: Policy | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.system = system
        self.agents_dir = agents_dir  # where AGENTS.md is auto-loaded from
        self.tools = tools
        self.approve = approve
        self.approval_required = approval_required or set()
        # v0.1: the gate is a Policy object. When none is passed, build one from the
        # ch-05 approve/approval_required pair so every existing caller is unchanged.
        self.policy = policy or Policy(
            require_approval=frozenset(self.approval_required), approve=approve
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        # v0.1: an event stream a driver can observe mid-run (subscribe()).
        self._subscribers: list[Callable[[dict], None]] = []
        # per-turn counters, reset at the top of run()
        self._turn_model_calls = 0
        self._turn_approvals = 0
        self._stop_reason = "stop"
        self.retry_count = 0
        self.context_limit = context_limit
        # v0.3: per-instance override of the module-global tool-step budget
        # (MAX_TOOL_STEPS). None (the default) preserves today's behavior exactly —
        # every existing caller is unchanged. A consumer that needs to honor its own
        # declared turn budget (e.g. an agent-package's limits.max_turns) passes it
        # here instead of reaching into the module global (adr/0002: a generic seam,
        # not a consumer's config format).
        self.max_tool_steps = max_tool_steps
        # No wall-clock bound by default (None): `_run`'s only stop condition has
        # always been `max_tool_steps`, and every existing caller is unchanged unless
        # it opts in. A consumer with a real wall-clock budget (an eval suite pitting
        # this loop against a harness that DOES have one) passes it here — same
        # reasoning as `max_tool_steps` just above.
        self.deadline_s = deadline_s
        self.skills = skills or []
        self._last_tokens = 0  # model-reported usage from the last call (ch-08)
        self.session = session
        self.sessions_dir = sessions_dir
        # Resume: load prior conversation from disk if this session exists (ch-09).
        self.messages: list[dict] = load_session(session, sessions_dir) if session else []
        # Set true whenever the last turn triggered compaction — the REPL reads
        # this to surface that the window was managed (a demoable, visible event).
        self.just_compacted = False
        self.compaction_count = 0
        self._active_turn_start = 0
        self.verify_attempts = verify_attempts
        # ch-12: when a turn changes code, refuse "done" until a real passing run
        # of the project's declared test command is observed. require_run opts out.
        self.require_run = require_run
        self.tracer = tracer
        # ch-13: restore the persisted trace too, so it isn't empty on resume.
        if self.tracer is not None and session:
            self.tracer.load_events(load_trace(session, sessions_dir))

    def _approved(self, name: str, args: str) -> bool:
        # Fail closed: a tool marked as requiring approval with no approver is denied.
        return self.approve(name, args) if self.approve else False

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Observe this agent's events as they happen — ``turn_start``, each
        ``tool_call`` (name, args, result, is_error), and ``turn_end`` (carrying the
        RunResult). Each event is a plain dict, so a driver reads or reacts mid-run
        without importing carbon's internals (the embedding seam, adr/0002)."""
        self._subscribers.append(callback)

    def _emit(self, event: dict) -> None:
        for cb in self._subscribers:
            cb(event)

    def _save(self) -> None:
        # Durable state: persist the full history so a restart can resume it.
        if self.session:
            save_session(self.session, self.messages, self.sessions_dir)
            if self.tracer is not None:
                save_trace(self.session, self.tracer.dump_events(), self.sessions_dir)

    def _maybe_compact(self) -> None:
        # ch-08: prefer the model's reported usage; fall back to an estimate on turn one.
        self.just_compacted = False
        window = self._last_tokens or estimate_tokens(self.messages)
        policy = CONFIG.compaction
        trigger = int(self.context_limit * policy.trigger_fraction)
        if policy.completion_reserve:
            # An explicit reserve for the reply. `trigger_fraction` scales with the
            # window, but the completion does not — the reply that has to fit is about
            # the same size at 4k and at 128k, so a fraction that leaves room at one
            # leaves far too little or absurdly much at the other. Whichever door is
            # tighter wins, so adding a reserve can only ever compact earlier.
            trigger = min(trigger, self.context_limit - policy.completion_reserve)
        if window > trigger:
            managed = compact(self.messages, model=self.model, provider=self.provider)
            if managed is self.messages:
                return
            self.messages = managed
            self._last_tokens = 0  # recomputed from the next response
            self.just_compacted = True
            self.compaction_count += 1

    def _system_text(self) -> str:
        """Instruction layer = system prompt + project AGENTS.md + skills menu."""
        parts = [
            p
            for p in (
                self.system,
                load_agents_md(self.agents_dir),
                skills_prompt(self.skills),
            )
            if p
        ]
        return "\n\n".join(parts)

    def _payload(self) -> list[dict]:
        """System prompt first (if any), then the full conversation history."""
        sys_text = self._system_text()
        head = [{"role": "system", "content": sys_text}] if sys_text else []
        return head + self.messages

    def run(self, user_text: str, *, on_delta: OnDelta | None = None) -> RunResult:
        """Run one turn and return its structured outcome (v0.1, the embedding seam).

        Inject @path files, drive the loop, then — if this turn changed code —
        enforce a real passing run of the project's tests before returning. The
        result carries the final text plus what happened: the tool calls, the model
        calls (``turns``), the gated calls that ran (``approvals``), the stop reason,
        and the tracer totals. ``send`` is the string-returning shim over this.

        ``on_delta``, when given, streams this turn's tokens to the callback."""
        self._turn_model_calls = 0
        self._turn_approvals = 0
        self._stop_reason = "stop"
        if self.tracer:
            self.tracer.turn_start()  # ch-13: nest this turn's steps under one span
        # Compact BEFORE this turn's messages are appended, so ``turn_start`` (an
        # index) stays valid for the whole turn. Compacting mid-turn (inside _run)
        # would renumber the list and make the verification gate read the wrong
        # slice — silently skipping the required test run.
        self._maybe_compact()
        for block in deliver(user_text):  # @file references → injected context
            self.messages.append({"role": "user", "content": f"Context file:\n{block}"})
        self.messages.append({"role": "user", "content": user_text})
        turn_start = len(self.messages)
        self._active_turn_start = turn_start
        self._emit({"type": "turn_start"})
        reply = self._run(on_delta)
        turn_start = self._active_turn_start
        # gate "done" on a real test run (re-prompt runs stream too)
        reply = self._enforce_run(reply, turn_start, on_delta)
        self._save()  # durable state: persist after every turn
        result = RunResult(
            text=reply,
            tool_calls=self._collect_tool_calls(turn_start),
            turns=self._turn_model_calls,
            approvals=self._turn_approvals,
            stop_reason=self._stop_reason,
            totals=self.tracer.totals() if self.tracer else {},
        )
        self._emit({"type": "turn_end", "result": result})
        return result

    def send(self, user_text: str, *, on_delta: OnDelta | None = None) -> str:
        """Run one turn and return the final text — the ch-02..ch-14 contract,
        now a thin shim over ``run`` so every existing caller is unchanged."""
        return self.run(user_text, on_delta=on_delta).text

    def _collect_tool_calls(self, turn_start: int) -> list[ToolCall]:
        """Reconstruct this turn's tool calls from the transcript, pairing each
        assistant tool_call with its recorded tool result by id. carbon reports what
        ran; the ``attributes`` bag is left empty for a consumer to fill."""
        results: dict[str, str] = {
            m.get("tool_call_id", ""): str(m.get("content", ""))
            for m in self.messages[turn_start:]
            if m.get("role") == "tool"
        }
        calls: list[ToolCall] = []
        for m in self.messages[turn_start:]:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                res = results.get(tc.get("id", ""), "")
                # Seed the call's bag from the tool's static attributes (a fresh
                # copy, so a consumer mutating one call never touches the tool).
                tool = self.tools.get(name) if self.tools else None
                calls.append(
                    ToolCall(
                        name=name,
                        args=fn.get("arguments", ""),
                        result=res,
                        is_error=res.startswith("error"),
                        attributes=dict(tool.attributes) if tool else {},
                    )
                )
        return calls

    def _changed_code(self, turn_start: int) -> bool:
        """Did this turn write or edit a source file? The trigger for the gate — a
        code change to verify, not a prose file like facts.txt (by extension, the
        way a pre-commit hook decides what to run on)."""
        for m in self.messages[turn_start:]:
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") in ("write_file", "edit_file"):
                    try:
                        path = json.loads(fn.get("arguments", "{}")).get("path", "")
                    except json.JSONDecodeError:
                        path = ""
                    if any(path.endswith(ext) for ext in CODE_EXTENSIONS):
                        return True
        return False

    def _enforce_run(self, reply: str, turn_start: int, on_delta: OnDelta | None = None) -> str:
        """If this turn changed code, refuse "done" until a real passing run of the
        project's declared test command (AGENTS.md ``## Testing``) is observed *after*
        the change. The model runs it with bash; the harness only watches the
        receipts. Capped at verify_attempts so a model that will not run it cannot
        hang the loop. On exhaustion the reply is marked unverified — the gate never
        implies a pass it didn't see. No declared command, or no code change → no gate."""
        if not self.require_run:
            return reply
        command = test_command(self.agents_dir)
        if not command or not self._changed_code(turn_start):
            return reply
        for _ in range(self.verify_attempts):
            if self._record_pass(command, turn_start):
                return reply
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "You changed code but I don't see a passing run of the "
                        f"project's tests. Run `{command}` with the bash tool now — "
                        "it must exit 0 before you report done. Show the real output."
                    ),
                }
            )
            reply = self._run(on_delta)
            turn_start = self._active_turn_start
        # The last re-prompt's run hasn't been checked yet — check it, then fail closed.
        if self._record_pass(command, turn_start):
            return reply
        return (
            f"{reply}\n\n[unverified: this turn changed code but no passing `{command}` "
            f"run was observed after the change (tried {self.verify_attempts}×). "
            "Treat the change as NOT verified.]"
        )

    def _record_pass(self, command: str, turn_start: int) -> bool:
        """Check the gate once and record the verify span; returns whether it passed."""
        passed = self._observed_pass(command, turn_start)
        if self.tracer:
            self.tracer.record_verify(passed, 0.0, f"required: {command}")
        return passed

    @staticmethod
    def _is_test_run(arguments: str, command: str) -> bool:
        """True iff a bash call runs ``command`` up front, not wrapped or chained —
        so ``echo 'uv run verify'`` or ``uv run verify || true`` can't spoof the gate."""
        try:
            cmd = json.loads(arguments or "{}").get("command", "")
        except json.JSONDecodeError:
            return False
        cmd = str(cmd).strip()
        if not cmd.startswith(command):
            return False
        return not any(op in cmd for op in (";", "&&", "||", "|", "`", "$("))

    def _last_code_mutation(self, turn_start: int) -> int:
        """Index of the last assistant message this turn that wrote/edited source, or -1."""
        last = -1
        for i in range(turn_start, len(self.messages)):
            m = self.messages[i]
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") in ("write_file", "edit_file"):
                    try:
                        path = json.loads(fn.get("arguments", "{}")).get("path", "")
                    except json.JSONDecodeError:
                        path = ""
                    if any(path.endswith(ext) for ext in CODE_EXTENSIONS):
                        last = i
        return last

    def _observed_pass(self, command: str, turn_start: int) -> bool:
        """True iff this turn's transcript holds a bash call that ran ``command``
        (unwrapped) and exited 0, at or after the last code change — so a pass from
        *before* the final edit doesn't count as verifying it. Paired by tool_call_id
        so a failed run is never counted as a pass."""
        after = self._last_code_mutation(turn_start)
        ran_ids: set[str] = set()
        for i in range(turn_start, len(self.messages)):
            if i < after:  # a run before the last mutation can't have verified it
                continue
            m = self.messages[i]
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") == "bash" and self._is_test_run(fn.get("arguments", ""), command):
                    ran_ids.add(tc.get("id", ""))
        if not ran_ids:
            return False
        return any(
            m.get("role") == "tool"
            and m.get("tool_call_id") in ran_ids
            and str(m.get("content", "")).startswith("[exit 0")
            for m in self.messages[turn_start:]
        )

    def _run(self, on_delta: OnDelta | None = None) -> str:
        """Drive the model, executing tool calls until it produces a final answer.

        Compaction happens once per turn in ``send`` (before the turn is appended),
        never here — so the verification gate's ``turn_start`` index stays valid even
        across the re-prompt runs ``_enforce_run`` drives."""
        specs = self.tools.specs() if self.tools else None
        tool_step_budget = (
            self.max_tool_steps if self.max_tool_steps is not None else MAX_TOOL_STEPS
        )
        deadline = time.monotonic() + self.deadline_s if self.deadline_s is not None else None
        for _ in range(tool_step_budget):
            # Checked before starting a new turn, not mid-turn: a turn already in
            # flight (a blocking model call, a running tool) completes rather than
            # being torn down partway — the same "let in-flight work finish, just
            # start nothing new" posture the tool-step budget has always had.
            if deadline is not None and time.monotonic() >= deadline:
                self._stop_reason = "deadline"
                return "error: exceeded wall-clock deadline"
            resp, payload, t0 = self._model_call_with_recovery(specs, on_delta)
            self._last_tokens = int(resp.usage.get("total_tokens", 0)) or self._last_tokens
            if self.tracer:
                self.tracer.record_llm(
                    resp.usage,
                    time.perf_counter() - t0,
                    finish_reason=resp.finish_reason,
                    request_model=self.model,
                    messages=payload,  # optional content; captured only when enabled
                    output=resp.content,
                    tool_definitions=specs,  # the same list already sent to chat()
                )
            if resp.tool_calls and self.tools is not None:
                if resp.finish_reason == "length":
                    # A cut-off JSON argument can still look like a tool call to a
                    # forgiving provider. Never execute actions from an incomplete
                    # response; surface an explicit, diagnosable stop instead.
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                (resp.content or "")
                                + "\n[response truncated before tool calls completed; "
                                "no tool call was executed]"
                            ),
                        }
                    )
                    self._stop_reason = "incomplete_response"
                    return "error: model response was truncated; no tool calls were executed"
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": resp.tool_calls,
                    }
                )
                for tc in resp.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    t1 = time.perf_counter()
                    # Pass the Tool: read_only reads its declared effect, so a
                    # custom tool the mutator name-list has never heard of is
                    # refused rather than assumed harmless.
                    allowed, marker = self.policy.decision(name, args, tool=self.tools.get(name))
                    if not allowed:
                        result = marker
                        status = "denied"
                    else:
                        if name in self.policy.require_approval:
                            self._turn_approvals += 1
                        result = self.tools.call(name, args)
                        status = "error" if result.startswith("error") else "ok"
                    if self.tracer:
                        self.tracer.record_tool(
                            name, time.perf_counter() - t1, args=args, result=result, status=status
                        )
                    self._emit(
                        {
                            "type": "tool_call",
                            "name": name,
                            "args": args,
                            "result": result,
                            "is_error": status == "error",
                        }
                    )
                    # Keep the selected, Carbon-owned strategy fixed while a tool may
                    # ask for a smaller budget. The strategy itself is an editable,
                    # bounded choice; arbitrary truncation code is not.
                    tool = self.tools.get(name)
                    budget = tool.max_result_chars if tool and tool.max_result_chars else None
                    hint = None
                    if name == "read_file":
                        hint = "Use start_line/end_line to request the missing range."
                    content = truncate(
                        result,
                        CONFIG.tool_output,
                        budget=budget,
                        continuation_hint=hint,
                    )
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.get("id", ""), "content": content}
                    )
                continue
            if resp.finish_reason == "length":
                partial = resp.content or ""
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": partial + "\n[response truncated before completion]",
                    }
                )
                self._stop_reason = "incomplete_response"
                return (
                    f"{partial}\n\n"
                    "[incomplete: the model reached its output limit before finishing]"
                )
            self.messages.append({"role": "assistant", "content": resp.content})
            return resp.content
        self._stop_reason = "tool_budget"
        return "error: exceeded tool-step budget"

    @staticmethod
    def _context_overflow(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "context length",
                "context window",
                "maximum context",
                "too many tokens",
                "token limit",
            )
        )

    @staticmethod
    def _transient_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "429",
                "rate limit",
                "timeout",
                "timed out",
                "temporarily unavailable",
                "connection reset",
                "connection refused",
                "502",
                "503",
                "504",
            )
        )

    def _compact_active_history(self) -> bool:
        """Reclaim window without disturbing turn-relative gate indices.

        Both halves, always, in one pass — there is only one recovery attempt, so
        doing half the work spends it. Prefix compaction cannot reach an oversized
        tool result inside the *current* turn, and shrinking the current turn does
        nothing about a long earlier history; a session with both would compact the
        prefix, leave the tool result whole, and crash on the next overflow with its
        one attempt already used.
        """
        prefix = self.messages[: self._active_turn_start]
        current_turn = self.messages[self._active_turn_start :]
        # recent_token_reserve=0 forces the guaranteed-progress message-count cut
        # regardless of the configured steady-state strategy. Token budgeting is a
        # legitimate policy for the ordinary pre-turn door — "everything within the
        # reserve counts as recent, leave it" — but this call has a stronger, different
        # requirement: an overflow just happened, and returning the prefix unchanged
        # means the identical request overflows again with no path forward. A short
        # prefix (many small messages, few tokens — exactly a fault-injection fixture,
        # but not only that) can fit entirely inside a real token reserve and make
        # compact() correctly report nothing to summarize; here that correctness is a
        # regression, so this caller opts out of the reserve rather than the reserve
        # quietly making an exception for it.
        managed = compact(prefix, model=self.model, provider=self.provider, recent_token_reserve=0)
        compacted = managed is not prefix
        if compacted:
            self.messages = managed + current_turn
            self._active_turn_start = len(managed)
            self.just_compacted = True
            self.compaction_count += 1
        shrank = self._shrink_turn_tool_results()
        if compacted or shrank:
            self._last_tokens = 0
        return compacted or shrank

    def _shrink_turn_tool_results(self) -> bool:
        """Shrink this turn's oversized tool results in place; True if anything moved.

        Receipt-safe by construction. The verification gate pairs an *assistant*
        message's tool_calls with a tool message whose content starts with
        ``[exit 0`` — so this never touches assistant messages, and every truncation
        strategy in the menu keeps the head, where that marker lives.
        """
        budget = max(200, CONFIG.tool_output.budget // 4)
        shrank = False
        for message in self.messages[self._active_turn_start :]:
            if message.get("role") != "tool":
                continue
            content = str(message.get("content", ""))
            if len(content) <= budget:
                continue
            message["content"] = truncate(content, CONFIG.tool_output, budget=budget)
            shrank = True
        return shrank

    def _model_call_with_recovery(
        self, specs: list[dict] | None, on_delta: OnDelta | None
    ) -> tuple[LLMResponse, list[dict], float]:
        """Call the model with one forced overflow recovery and bounded retries."""
        policy = CONFIG.retry
        attempt = 0
        overflow_recovered = False
        while True:
            attempt += 1
            payload = self._payload()
            t0 = time.perf_counter()
            try:
                response = chat(
                    payload,
                    model=self.model,
                    tools=specs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=self.response_format,
                    provider=self.provider,
                    on_delta=on_delta,
                )
                # Count completed calls only. `turns` is evidence the improvement
                # loop reads; a flaky endpoint must not read as a chattier agent.
                # Recovery attempts are their own signal — see `retry_count`.
                self._turn_model_calls += 1
                return response, payload, t0
            except Exception as exc:
                if (
                    not overflow_recovered
                    and self._context_overflow(exc)
                    and self._compact_active_history()
                ):
                    overflow_recovered = True
                    self.retry_count += 1
                    continue
                strat = _RETRY_STRATEGIES.get(policy.strategy)
                delay = (
                    strat.next_delay(attempt, policy)
                    if strat and self._transient_error(exc)
                    else None
                )
                if delay is None:
                    raise
                self.retry_count += 1
                if delay:
                    time.sleep(delay)


# --- non-interactive print mode ---------------------------------------------
def _approver(yes: bool) -> Callable[[str, str], bool] | None:
    """Approval policy for a non-interactive run. There's no TTY to prompt at, so
    the default is fail-closed — return ``None`` and the agent's ``_approved``
    denies every gated tool. ``--yes`` opts into auto-approve for scripting/CI."""
    return (lambda name, args: True) if yes else None


def _extension_dirs(workspace_root: str | Path) -> tuple[Path, Path]:
    """Where extensions load from: a user-level directory, then a project-local
    one — always explicit paths at the construction site, never a config-file
    field (dev-notes/adr/0003)."""
    return Path.home() / ".carbon" / "extensions", Path(workspace_root) / ".carbon" / "extensions"


def _coding_tools(
    workspace,
    *,
    exclude_session: str | None,
    provider: Provider | None = None,
    model: str | None = None,
    sessions_dir: str | Path | None = None,
) -> ToolRegistry:
    """The mature agent's toolset, rooted at ``workspace`` — the one builder print
    mode, the REPL, and the TUI all use, so a delegated worker can't end up reading
    a different tree than the agent that spawned it.

    ``provider``/``model`` are threaded to the workers too: a consumer that passes a
    custom provider expects its delegates to use it, not to silently fall back to
    whatever the environment points at.
    """
    from harness.memory import search_memory_tool
    from harness.sandbox import Sandbox, bash_tool
    from harness.subagents import delegate_tool, fan_out_tool
    from harness.tools import default_tools
    from harness.workspace import apply_patch_tool, edit_file_tool, write_file_tool

    root = str(workspace.root)
    memory_dir = sessions_dir if sessions_dir is not None else DEFAULT_DIR

    def worker_tools() -> ToolRegistry:
        """What a delegated worker gets: the parent's workspace, read-only.

        A worker runs as its own Agent with its own Policy, so a mutating tool handed
        to a worker executes without ever reaching the parent's approval gate — a
        parent running fail-closed could delegate the write it just refused. Until
        delegation has a composite approval design, workers observe and report; the
        parent makes the changes.
        """
        registry = default_tools(root)
        registry.register(search_memory_tool(memory_dir, exclude=exclude_session))
        return registry

    tools = default_tools(root)
    tools.register(write_file_tool(workspace))
    tools.register(edit_file_tool(workspace))
    tools.register(apply_patch_tool(workspace))
    tools.register(bash_tool(Sandbox(trusted=True, timeout=120), workdir=root))
    tools.register(search_memory_tool(memory_dir, exclude=exclude_session))
    workers = worker_tools()
    tools.register(delegate_tool(model=model, provider=provider, tools=workers, agents_dir=root))
    tools.register(fan_out_tool(model=model, provider=provider, tools=workers, agents_dir=root))
    return tools


def run_once(
    prompt: str,
    *,
    provider: Provider | None = None,
    fmt: str = "plain",
    yes: bool = False,
    extensions: bool = False,
    on_delta: OnDelta | None = None,
    session: str | None = None,
    sessions_dir: str = DEFAULT_DIR,
    workspace_root: str = ".",
    agents_dir: str | None = None,
) -> str:
    """Run exactly one turn non-interactively and return it rendered as ``fmt``
    ("plain" | "json" | "transcript"). Fail-closed on approvals unless ``yes``.

    Each invocation is stateless by default (``session=None``): nothing persists and
    no history accumulates across calls — a one-shot is independent. Unlike the REPL
    (which works in a throwaway git worktree), print mode operates on
    ``workspace_root`` (the real project by default) — the deliberate one-shot
    posture: it's your command, gated by approval unless you pass ``--yes``, and
    loads no extensions unless ``extensions=True``."""
    from harness.extensions import load_extensions
    from harness.render import render_json, render_plain, render_transcript
    from harness.skills import load_skills
    from harness.workspace import Workspace

    provider = provider or Provider.from_env()
    workspace = Workspace(root=workspace_root)
    tracer = Tracer(model=provider.model)
    tools = _coding_tools(
        workspace,
        exclude_session=session,
        provider=provider,
        model=provider.model,
        sessions_dir=sessions_dir,
    )
    if extensions:
        load_extensions(tools, *_extension_dirs(workspace.root))
    agent = Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        approve=_approver(yes),
        approval_required=APPROVAL_TOOLS,
        skills=load_skills("skills"),
        session=session,
        sessions_dir=sessions_dir,
        agents_dir=agents_dir or str(workspace.root),
        tracer=tracer,
    )
    reply = agent.send(prompt, on_delta=on_delta)
    if fmt == "json":
        return render_json(reply, tracer, agent.messages)
    if fmt == "transcript":
        return render_transcript(agent.messages, tracer)
    return render_plain(reply)


def _stdout_sink(channel: str, text: str) -> None:
    """Stream tokens live: the visible answer to stdout, the model's thinking dimmed
    to stderr (so a piped ``stdout`` stays clean — just the answer)."""
    if channel == "reasoning":
        sys.stderr.write(f"\033[2m{text}\033[0m")
        sys.stderr.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="run one turn non-interactively and print the result, then exit. "
        "Omit it to open the interactive REPL.",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "json", "transcript"),
        default="plain",
        help="print-mode output shape (default: %(default)s). plain streams the "
        "answer; json/transcript emit once after the turn.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="auto-approve the gated tools "
        f"({'/'.join(sorted(CONFIG.approval_tools))}) in print mode. Without it, "
        "print mode is fail-closed and denies them (no TTY to prompt at).",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=DEFAULT_CONTEXT_LIMIT,
        help="token budget before the window is compacted (default: %(default)s). "
        "Set it low, e.g. 400, to watch compaction fire live.",
    )
    parser.add_argument(
        "--extensions",
        action="store_true",
        help="load .py extensions from ~/.carbon/extensions/ and .carbon/extensions/ "
        "(off by default: a file the agent's own write_file tool writes into "
        ".carbon/extensions/ would otherwise auto-run, unsandboxed and unapproved, "
        "on every future invocation).",
    )
    args = parser.parse_args()

    if args.prompt is not None:
        _run_print_mode(args)
        return
    _run_repl(args)


def _run_print_mode(args: argparse.Namespace) -> None:
    """One-shot: run a single turn on the current project and emit it as ``--format``."""
    provider = Provider.from_env()
    if args.format == "plain":
        # Stream the answer live to stdout; the returned render is what we streamed.
        run_once(
            args.prompt,
            provider=provider,
            fmt="plain",
            yes=args.yes,
            extensions=args.extensions,
            on_delta=_stdout_sink,
        )
        print()  # terminate the streamed line
    else:
        print(
            run_once(
                args.prompt,
                provider=provider,
                fmt=args.format,
                yes=args.yes,
                extensions=args.extensions,
            )
        )


def _run_repl(args: argparse.Namespace) -> None:
    from harness.extensions import load_extensions
    from harness.orchestrator import Orchestrator
    from harness.skills import load_skills
    from harness.workspace import Workspace, git_worktree

    # Work in a real project: a git worktree of this repo (your checkout stays
    # pristine — edits land in a throwaway worktree), or a scratch dir when we're
    # not in a git repo. This is the coding-agent posture: it works in your code.
    wt = git_worktree(".")
    if wt is not None:
        workspace, cleanup = wt
        print(f"working in a git worktree of this repo — {workspace.root}")
    else:
        workspace, cleanup = Workspace(), (lambda: None)
        print(f"not a git repo — working in a scratch dir — {workspace.root}")

    def approve(name: str, args_json: str) -> bool:
        return input(f"  approve {name}({args_json})? [y/N] ").strip().lower() in ("y", "yes")

    # Resolve the provider once so the model id is known up front — the tracer needs
    # it to price calls (else the REPL trace shows $0.0000), and the agent reuses it.
    provider = Provider.from_env()
    tracer = Tracer(model=provider.model)  # ch-13: record every step + price it
    tools = _coding_tools(
        workspace, exclude_session="repl", provider=provider, model=provider.model
    )
    if args.extensions:
        load_extensions(tools, *_extension_dirs(workspace.root))
    agent = Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        approve=approve,
        approval_required=APPROVAL_TOOLS,
        context_limit=args.context_limit,
        skills=load_skills("skills"),
        session="repl",
        agents_dir=str(workspace.root),  # read AGENTS.md (incl. ## Testing) from the project
        tracer=tracer,
    )
    print(
        "agent ready — streaming replies; observable runs (a trace with tokens + cost "
        "after each turn); change code and the harness enforces the project's tests "
        "before 'done'; /plan; durable sessions, approval gate, skills. Ctrl-D to exit."
    )
    orchestrator = Orchestrator()
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                print()
                break
            if not user.strip():
                continue
            if user.startswith("/plan "):
                task = user[len("/plan ") :].strip()
                if not task:
                    print("usage: /plan <task>")
                    continue
                result = orchestrator.run(task)
                print("plan:")
                for i, step in enumerate(result.plan, 1):
                    print(f"  {i}. {step}")
                print("results:")
                for i, (step, res) in enumerate(zip(result.plan, result.results, strict=False), 1):
                    print(f"  {i}. {step}\n     → {res}")
                continue
            print("bot> ", end="", flush=True)
            reply = agent.send(user, on_delta=_stdout_sink)  # tokens stream live
            print()  # end the streamed line
            if agent.just_compacted:
                print("[context compacted — kept the start and end, summarized the middle]")
            _ = reply  # already shown via the stream; keep the name for clarity
            print(tracer.timeline())
    finally:
        cleanup()


if __name__ == "__main__":
    main()
