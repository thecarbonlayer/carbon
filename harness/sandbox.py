"""Execution environment (ch-08) — the harness runs code, the model never does.

The model only ever asks; the harness executes, inside a boundary. The sandbox
prefers hardened Docker (``--network none``, non-root, scoped workdir) and falls
back to a scoped local subprocess when no Docker daemon is available.

"Start closed": no network, a fresh isolated workdir, and a scrubbed environment
(no inherited credentials), so untrusted code never sees the host's secrets. The
sandbox is the backstop, not the only defense.

**Contract — read this before trusting it.** The *Docker* backend is a genuine
containment boundary (network-none, non-root, cap-drop, memory + pid limits,
read-only rootfs). The *local* fallback is **teaching-grade, not a security
boundary**: it scrubs the environment and confines the cwd, and (as of the
hardening below) puts each command in its own process group so a timeout kills the
whole tree, and caps returned output — but it does *not* isolate the filesystem or
cap host memory. Untrusted code on the local fallback can still read host files.
Real isolation is a threat-model choice (gVisor / microVM / container-by-default);
this course keeps the local path simple on purpose and names the limit rather than
hiding it. See the README ("Why it's built this way": soft sandbox, hard verification).

The seam was introduced minimal at ch-05 (one chokepoint for code execution);
this is the hardening — the boundary that makes that chokepoint trustworthy.
Give it a ``workdir`` and the command runs in that persistent directory, so a
bash command can see a file a write tool just created (the workspace seam).
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import uuid
from dataclasses import dataclass

from harness.tools import Tool

# Minimal environment handed to sandboxed commands — note the absence of secrets.
_SCRUBBED_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C"}
_MAX_OUTPUT = 100_000  # cap returned output so a chatty command can't flood the window


def _cap(s: str | None) -> str:
    s = s or ""
    if len(s) <= _MAX_OUTPUT:
        return s
    # The subprocess safety cap is larger than the prompt door, but it must use
    # the same selected retention policy or a prefix-only cap here would destroy
    # the failure tail before Agent can preserve it.
    from harness.harness_config import CONFIG
    from harness.limits import truncate

    return truncate(s, CONFIG.tool_output, budget=_MAX_OUTPUT)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the process *group* so backgrounded descendants die with the timeout,
    not just the immediate shell.

    ``start_new_session=True`` makes the child its own group leader, so the group id
    equals ``proc.pid`` — use that directly rather than ``os.getpgid`` (which raises
    if the shell has already exited to background a job, leaving the tree orphaned)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    backend: str


class Sandbox:
    def __init__(
        self,
        image: str = "busybox",
        timeout: float = 15.0,
        prefer_docker: bool = True,
        trusted: bool = False,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.prefer_docker = prefer_docker
        # trusted: run in the REAL environment (uv/PATH/deps visible), unscrubbed.
        # For a coding agent working on your own project running your own test
        # command — the approval gate is the control, not network-none isolation.
        self.trusted = trusted
        self._docker: bool | None = None

    def _docker_up(self) -> bool:
        if not self.prefer_docker:
            return False
        if self._docker is None:
            try:
                self._docker = (
                    subprocess.run(
                        ["docker", "info"],
                        capture_output=True,
                        timeout=5,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.SubprocessError):
                self._docker = False
        return self._docker

    def run(self, command: str, workdir: str | None = None) -> SandboxResult:
        # A workdir makes the sandbox operate on a persistent workspace (bind-mounted
        # in docker, cwd locally) instead of a throwaway dir.
        if self.trusted:
            return self._run_local(command, workdir)
        if self._docker_up():
            return self._run_docker(command, workdir)
        return self._run_local(command, workdir)

    def _run_docker(self, command: str, workdir: str | None) -> SandboxResult:
        # Hardened: no network, non-root, capabilities dropped, writable only in /work.
        # /work is a throwaway tmpfs unless a workspace is bind-mounted.
        work = ["-v", f"{workdir}:/work"] if workdir else ["--tmpfs", "/work:rw,size=16m"]
        name = f"agent-sbx-{uuid.uuid4().hex[:12]}"  # so a timeout can kill the container
        argv = [
            "docker", "run", "--rm",
            "--name", name,
            "--network", "none",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--memory", "256m",
            "--pids-limit", "128",
            "--read-only",
            *work,
            "-w", "/work",
            self.image,
            "sh", "-c", command,
        ]  # fmt: skip
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            # `docker run --rm` leaves the container running when our client is killed —
            # stop it explicitly so it doesn't outlive the timeout.
            subprocess.run(["docker", "kill", name], capture_output=True)
            return SandboxResult("", "error: timed out", 124, "docker")
        return SandboxResult(_cap(proc.stdout), _cap(proc.stderr), proc.returncode, "docker")

    def _run_local(self, command: str, workdir: str | None) -> SandboxResult:
        # Fallback: scrubbed env + timeout. Uses the persistent workspace if given,
        # else a fresh throwaway dir. (network is NOT isolated here — that needs Docker.)
        cwd = workdir or tempfile.mkdtemp(prefix="sandbox-")
        # trusted → the real environment (your test runner needs uv/PATH/deps);
        # otherwise the scrubbed env (untrusted code sees no host secrets).
        env = os.environ.copy() if self.trusted else dict(_SCRUBBED_ENV, HOME=cwd, TMPDIR=cwd)
        backend = "trusted" if self.trusted else "local"
        # Own process group (start_new_session) so a timeout kills the whole tree,
        # including any backgrounded descendants — not just the immediate shell.
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            stdout, stderr = proc.communicate()
            return SandboxResult(_cap(stdout), _cap(stderr) + "\nerror: timed out", 124, backend)
        return SandboxResult(_cap(stdout), _cap(stderr), proc.returncode, backend)


def bash_tool(sandbox: Sandbox, workdir: str | None = None) -> Tool:
    """A bash tool whose commands run inside the sandbox. With a workdir, commands
    run in the persistent workspace (so they see files the edit tools wrote)."""

    def run_bash(command: str) -> str:
        r = sandbox.run(command, workdir=workdir)
        body = (r.stdout + r.stderr).strip()
        return f"[exit {r.exit_code} via {r.backend}]\n{body}"

    return Tool(
        name="bash",
        description="Run a shell command in an isolated sandbox and return its output.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        func=run_bash,
    )
