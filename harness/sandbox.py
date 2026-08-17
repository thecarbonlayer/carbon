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
whole tree — but it does *not* isolate the filesystem or cap host memory. Untrusted
code on the local fallback can still read host files. The size limit that keeps a
chatty command out of the model's window is not here at all: that is the agent's
door (ch-06), applied once to the assembled tool result. This chapter adds only a
blunt ceiling, set above anything real, so isolation needs no truncation policy,
no workspace, and no imports from the chapters above it.
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
from pathlib import Path

from harness.tools import Tool

# Minimal environment handed to sandboxed commands — note the absence of secrets.
_SCRUBBED_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C"}
# The shell's route into the session's scratch (harness/limits.py's ``shell_ref``
# builds the same name into text the model reads). A fixed mount under Docker — the
# container path is stable and says nothing about the host — the real path under
# local execution, where there is no mount namespace to hide behind. Either way the
# MODEL only ever sees the variable's NAME, `$CARBON_SCRATCH_DIR/offload/<file>`, so
# no host path reaches a prompt or a stored transcript. Defined here, not in
# limits.py: this module imports nothing from the chapters above it (see the module
# docstring), so a sandbox-side constant is what limits.py reaches for, never the
# reverse — and limits.py does so with a deferred import (see ``shell_ref``), because
# this module imports ``harness.tools``, which imports ``harness.limits`` for
# ``SCRATCH_SCHEME``; a module-level import back here would close that into a real
# three-module cycle, order-dependent on which of the three a caller happens to touch
# first (verified: the entry point that breaks it is ``import harness.tools`` first,
# since ``harness.tools`` pauses on this exact import line before its own ``Tool``
# class is defined, and this module then asks the still-loading tools module for it).
SCRATCH_ENV_VAR = "CARBON_SCRATCH_DIR"
DOCKER_SCRATCH_MOUNT = "/carbon-scratch"
# A blunt ceiling on what one command hands back, set well above anything real, and
# deliberately not a second truncation door: applying the agent's policy here re-cut
# text the agent's own door cuts again, and under a strategy that WRITES it spilled a
# file at this layer for the layer above to re-trim. The 100k it replaces was measured
# to protect nothing — both backends materialize the whole stream in this process
# before this line runs (a 256MB result peaked at 875MB either way), and the agent's
# door already accepts a 33MB read_file result. What is left is an honest bound: above
# it, completeness is NOT promised.
_MAX_OUTPUT = 10_000_000


def _cap(s: str) -> str:
    """The ceiling: head and tail, no policy, no workspace, no file. Tail included
    because the last thing a failing command writes is usually the reason it failed."""
    if len(s) <= _MAX_OUTPUT:
        return s
    half = _MAX_OUTPUT // 2
    return f"{s[:half]}\n…[truncated {len(s) - _MAX_OUTPUT} chars]\n{s[-half:]}"


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
        scratch_dir: str | Path | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.prefer_docker = prefer_docker
        # trusted: run in the REAL environment (uv/PATH/deps visible), unscrubbed.
        # For a coding agent working on your own project running your own test
        # command — the approval gate is the control, not network-none isolation.
        self.trusted = trusted
        # Falsy check, not `is not None`: an empty string is not a real location, and
        # `Path("")` normalizes to `.` — wrapping it here would turn "no scratch
        # configured" into "the process's own cwd," which is not what an empty value
        # means. Normalized once, here, so every reader downstream sees Path-or-None.
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None
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
        #
        # The streams come back whole. Cutting them here, per stream, was the wrong
        # seam: the caller concatenates stdout and stderr (and a timeout appends a
        # suffix), so anything a per-stream cut appended landed in the MIDDLE of the
        # result the model gets. Nothing is saved by cutting early either — both streams
        # are fully buffered in this process by the time we see them.
        if self.trusted:
            return self._run_local(command, workdir)
        if self._docker_up():
            return self._run_docker(command, workdir)
        return self._run_local(command, workdir)

    def _run_docker(self, command: str, workdir: str | None) -> SandboxResult:
        # Hardened: no network, non-root, capabilities dropped, writable only in /work.
        # /work is a throwaway tmpfs unless a workspace is bind-mounted.
        work = ["-v", f"{workdir}:/work"] if workdir else ["--tmpfs", "/work:rw,size=16m"]
        # Read-only on purpose, unlike /work: the spill is evidence to be read, and a
        # container that could rewrite it could forge what the harness later
        # attributes to it. Omitted entirely (no mount, no env var) when there is no
        # real scratch dir — a stray `-e CARBON_SCRATCH_DIR=` naming a mount that was
        # never made would be worse than an unset var: a name that resolves to
        # nothing, instead of an honestly absent one.
        scratch = (
            [
                "-v",
                f"{self.scratch_dir}:{DOCKER_SCRATCH_MOUNT}:ro",
                "-e",
                f"{SCRATCH_ENV_VAR}={DOCKER_SCRATCH_MOUNT}",
            ]
            if self.scratch_dir is not None
            else []
        )
        # The fixed unprivileged uid below can't read the mount above: a spill is
        # 0600, owned by the invoking user, and a native Linux host's bind mount
        # preserves that host uid inside the container (unlike Docker Desktop's macOS
        # backend, which remaps ownership transparently) — so "nobody" gets EACCES on
        # its own session's evidence there, even though the same setup works on a Mac.
        # Relax this only when a scratch dir is actually mounted, and only to the
        # invoking user's own identity — never to a wider file mode. (harness/
        # limits.py's `_write_atomically` documents why a wider mode was tried and
        # reverted: it published complete tool output to every account on the host.)
        # `hasattr` guards platforms without `os.getuid` (Windows), where this
        # backend keeps the fixed uid, same as before this change.
        user = (
            f"{os.getuid()}:{os.getgid()}"
            if self.scratch_dir is not None and hasattr(os, "getuid")
            else "65534:65534"
        )
        name = f"agent-sbx-{uuid.uuid4().hex[:12]}"  # so a timeout can kill the container
        argv = [
            "docker", "run", "--rm",
            "--name", name,
            "--network", "none",
            "--user", user,
            "--cap-drop", "ALL",
            "--memory", "256m",
            "--pids-limit", "128",
            "--read-only",
            *work,
            *scratch,
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
        return SandboxResult(proc.stdout, proc.stderr, proc.returncode, "docker")

    def _run_local(self, command: str, workdir: str | None) -> SandboxResult:
        # Fallback: scrubbed env + timeout. Uses the persistent workspace if given,
        # else a fresh throwaway dir. (network is NOT isolated here — that needs Docker.)
        cwd = workdir or tempfile.mkdtemp(prefix="sandbox-")
        # trusted → the real environment (your test runner needs uv/PATH/deps);
        # otherwise the scrubbed env (untrusted code sees no host secrets). Either
        # way the scratch var rides along below — the route it advertises is not a
        # secret, and withholding it from untrusted code would make the route work
        # only for trusted wiring and silently vanish everywhere else.
        env = os.environ.copy() if self.trusted else dict(_SCRUBBED_ENV, HOME=cwd, TMPDIR=cwd)
        if self.scratch_dir is not None:
            # Set only when there is a real directory: an empty value would expand to
            # "" and turn "$CARBON_SCRATCH_DIR/offload/x" into an absolute read from
            # /. Unset is the honest state for "no scratch configured."
            env[SCRATCH_ENV_VAR] = str(self.scratch_dir)
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
            return SandboxResult(stdout, stderr + "\nerror: timed out", 124, backend)
        return SandboxResult(stdout, stderr, proc.returncode, backend)


def bash_tool(sandbox: Sandbox, workdir: str | None = None) -> Tool:
    """A bash tool whose commands run inside the sandbox. With a workdir, commands
    run in the persistent workspace (so they see files the edit tools wrote).

    No truncation policy: what the model sees is sized by the agent's door, which runs
    once on the whole string returned here."""

    def run_bash(command: str) -> str:
        r = sandbox.run(command, workdir=workdir)
        # Compose first, then the ceiling — this is the only place the whole result
        # exists. The exit header stays outside it: the verification gate reads a
        # receipt by its leading `[exit 0`.
        body = (r.stdout + r.stderr).strip()
        return f"[exit {r.exit_code} via {r.backend}]\n{_cap(body)}"

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
