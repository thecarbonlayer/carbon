"""Orchestration (ch-10).

A single model turn is not a workflow. The orchestrator plans a task into steps,
runs them in order through an agent, gates each step (approval), and retries on
failure — work moving through time with checkpoints, not one shot.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.tools import default_tools
from model import chat

if TYPE_CHECKING:
    from harness.observability import Tracer
    from harness.tools import ToolRegistry

_PLANNER = (
    "You are a planner. Break the task into 2-4 short imperative steps. "
    "Return ONLY a JSON array of step strings, nothing else."
)


@dataclass
class OrchestratorResult:
    plan: list[str]
    results: list[str]
    final: str


class Orchestrator:
    def __init__(
        self,
        model: str | None = None,
        tracer: Tracer | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.model = model
        self.tracer = tracer  # optional; default None => no behavior change (ch-10)
        self.tools = tools  # optional; default None => default_tools() (ch-10 -> now)

    def _plan(self, task: str) -> list[str]:
        # ch-13: wrap the planner's LLM call in a `plan` span when a tracer is given.
        t0 = time.perf_counter()
        text = chat(
            [{"role": "system", "content": _PLANNER}, {"role": "user", "content": task}],
            model=self.model,
            max_tokens=400,
        ).content.strip()
        if self.tracer is not None:
            self.tracer.record_plan(time.perf_counter() - t0)
        try:
            arr = json.loads(text[text.index("[") : text.rindex("]") + 1])
            steps = [str(s) for s in arr if str(s).strip()]
            if steps:
                return steps
        except (ValueError, json.JSONDecodeError):
            pass
        return [task]  # fallback: treat the whole task as one step

    def run(
        self,
        task: str,
        approve: Callable[[str], bool] | None = None,
    ) -> OrchestratorResult:
        from harness.agent import Agent  # lazy: avoids an import cycle at module load

        approve = approve or (lambda _step: True)
        plan = self._plan(task)
        worker = Agent(
            system="Execute each step using tools when needed. Be concise.",
            tools=self.tools if self.tools is not None else default_tools(),
            model=self.model,
        )
        results: list[str] = []
        for step in plan:
            if not approve(step):
                results.append(f"[skipped] {step}")
                continue
            results.append(self._run_with_retry(worker, step))
        return OrchestratorResult(plan=plan, results=results, final=results[-1] if results else "")

    @staticmethod
    def _run_with_retry(worker, step: str, attempts: int = 2) -> str:
        last = ""
        for _ in range(attempts):
            try:
                return worker.send(step)
            except Exception as exc:  # noqa: BLE001 — retry on any execution failure
                last = f"error: {exc}"
        return last
