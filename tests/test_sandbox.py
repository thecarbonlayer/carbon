"""The sandbox's half of the scratch route (ch-08, hardened for ch-06's spills).

``offload_to_file`` writes a spill under the session's private scratch and hands the
model a virtual ``scratch://offload/<hash>.txt`` ref that only ``read_file`` resolves.
A live measurement (iteration 5, task E4: recover a truncated artifact) scored 0/10 —
the transcripts showed 32 of 32 attempts to reach a spill went through bash (grep,
ls -F, a python one-liner), none of which can resolve ``scratch://``, so the model
fabricated an answer by re-deriving it instead. The ref looked like a path and
behaved like a private ``read_file`` API handle with exactly one consumer.

These tests pin the second adapter: the sandbox carries an optional ``scratch_dir``
and exposes it to bash as ``$CARBON_SCRATCH_DIR`` — set only when a real scratch dir
exists, never an empty string (which `Path("")` would normalize to `.`, turning a
rooted read into one from the process's own cwd instead of an honest unset var).
"""

from __future__ import annotations


def test_local_shell_can_read_a_scratch_artifact_via_the_env_var(tmp_path):
    """The route the model actually reaches for. 32 of 32 accesses in iteration 5's
    confirmation used bash; none could resolve scratch://, so recovery was 0/10."""
    from harness.sandbox import Sandbox

    scratch = tmp_path / "scratch"
    (scratch / "offload").mkdir(parents=True)
    (scratch / "offload" / "ab12.txt").write_text("NEEDLE-7F3A\n")
    sb = Sandbox(trusted=True, prefer_docker=False, scratch_dir=scratch)
    r = sb.run('grep NEEDLE "$CARBON_SCRATCH_DIR/offload/ab12.txt"', workdir=str(tmp_path))
    assert r.exit_code == 0 and "NEEDLE-7F3A" in r.stdout


def test_untrusted_local_shell_also_gets_the_scratch_route(tmp_path):
    """The scrubbed env drops host secrets but must still carry this var, or the
    route exists only for trusted wiring and silently vanishes elsewhere."""
    from harness.sandbox import Sandbox

    scratch = tmp_path / "scratch"
    (scratch / "offload").mkdir(parents=True)
    (scratch / "offload" / "cd34.txt").write_text("NEEDLE-9B1C\n")
    sb = Sandbox(trusted=False, prefer_docker=False, scratch_dir=scratch)
    r = sb.run('cat "$CARBON_SCRATCH_DIR/offload/cd34.txt"', workdir=str(tmp_path))
    assert "NEEDLE-9B1C" in r.stdout


def test_no_scratch_configured_leaves_the_var_unset_not_empty(tmp_path):
    """An empty CARBON_SCRATCH_DIR expands to "" and `cat "/offload/x"` reads from
    the filesystem root. Unset is the honest state."""
    from harness.sandbox import Sandbox

    sb = Sandbox(trusted=True, prefer_docker=False)
    r = sb.run('echo "[${CARBON_SCRATCH_DIR-UNSET}]"', workdir=str(tmp_path))
    assert "[UNSET]" in r.stdout


def test_docker_backend_mounts_scratch_read_only_at_a_fixed_path(monkeypatch):
    """Docker gets a FIXED container-side mount (``/carbon-scratch``): the container
    path is stable and says nothing about the host, unlike the local backend where
    the real path is the only option. Read-only on purpose — a container that could
    rewrite a spill could forge what the harness later attributes it wrote. Asserts
    the constructed argv directly rather than requiring a live docker daemon, which
    may not be running in this environment.

    Membership in ``argv`` as a whole is not enough to catch a real placement bug:
    ``docker run <flags> IMAGE <cmd...>`` treats everything AFTER the image name as
    the CONTAINER's own command, not a docker flag, so ``-v X -e Y`` sitting past the
    image is just two inert words handed to ``sh -c`` — the mount silently never
    happens while a bare ``in argv`` check keeps passing. So this slices to the
    flags that actually precede the image and asserts membership there instead.
    (Verified this actually catches that regression: temporarily moving the scratch
    flags after ``self.image`` in the implementation turns this test red; restoring
    the correct position turns it green again — see the task report for the
    mutation evidence.)

    Also covers why ``--user`` changes when a scratch dir is mounted: spills are
    0600, owned by the invoking user, and a native Linux host's bind mount preserves
    that host uid inside the container rather than remapping it — so the fixed
    unprivileged uid this backend otherwise runs as would get EACCES reading its
    own session's evidence there.

    The invoking identity is FAKED here (rather than read from this test process's
    own ``os.getuid()``) on purpose: asserting against the real ambient uid is weak
    in exactly the case worth knowing about — a root-run test process (a CI image)
    makes the expected value ``0:0``, which is indistinguishable from what a bug
    that just hardcoded ``"0:0"`` would ALSO produce. A distinctive, controlled,
    obviously-non-root pair pins the real contract: this backend uses THE INVOKING
    IDENTITY, not a coincidence. The root case itself gets its own dedicated test
    below, faked the same way, since almost no dev machine or ordinary CI actually
    runs pytest as root — without faking it, that branch would go untested in the
    overwhelming majority of runs even though it is a real, intended code path."""
    import os
    import subprocess

    from harness.sandbox import DOCKER_SCRATCH_MOUNT, SCRATCH_ENV_VAR, Sandbox

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "getuid", lambda: 4242)
    monkeypatch.setattr(os, "getgid", lambda: 4343)

    # _run_docker takes no path through `_docker_up`'s live probe — called directly,
    # exactly like `run()` calls it once the daemon is confirmed up — so no live
    # daemon is needed to check what argv this backend constructs.
    sb = Sandbox(trusted=False, prefer_docker=True, scratch_dir="/host/session-42/scratch")
    sb._run_docker("echo ok", workdir=None)

    argv = captured["argv"]
    # Only flags BEFORE the image name are docker-run options; anything after is the
    # container's own command line — a flag stranded there is not a flag at all.
    docker_flags = argv[: argv.index(sb.image)]
    # The scratch bind-mount: host path, fixed container path, read-only.
    assert f"/host/session-42/scratch:{DOCKER_SCRATCH_MOUNT}:ro" in docker_flags
    assert f"{SCRATCH_ENV_VAR}={DOCKER_SCRATCH_MOUNT}" in docker_flags
    # The model must never see the host path — only the fixed mount name travels via -e.
    assert not any(str(a).startswith(SCRATCH_ENV_VAR) and "/host/" in str(a) for a in argv)
    # With a scratch dir mounted, the container runs as the invoking user — not a
    # wider file mode — so it can read its own 0600 spill on a native Linux host.
    assert "4242:4343" in docker_flags
    assert "65534:65534" not in docker_flags


def test_docker_backend_scratch_mount_runs_as_root_when_the_harness_does(monkeypatch):
    """Root-run is the INTENDED behavior here, not a case to guard against: a
    harness invoked as root (a CI image, Docker-in-Docker) writes root-owned 0600
    spills, and root is then the only uid that can read them back. An
    ``os.getuid() != 0`` guard would silently fall back to the fixed unprivileged
    uid and re-break the very route this task exists to open, for exactly the
    accounts most likely to actually hit it. Faked rather than relying on this test
    process's own identity, since almost no dev machine or ordinary CI runs pytest
    as root — this pins the branch as deliberate rather than leaving it exercised
    only by accident, on whichever machine happens to run the suite as root."""
    import os
    import subprocess

    from harness.sandbox import Sandbox

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.setattr(os, "getgid", lambda: 0)

    sb = Sandbox(trusted=False, prefer_docker=True, scratch_dir="/host/session-42/scratch")
    sb._run_docker("echo ok", workdir=None)

    argv = captured["argv"]
    docker_flags = argv[: argv.index(sb.image)]
    assert "0:0" in docker_flags


def test_docker_backend_omits_scratch_mount_when_no_scratch_configured(monkeypatch):
    """No scratch_dir → no mount, no env var, and the fixed unprivileged uid stays —
    mirrors the local-path unset test, on the docker path. A stray
    ``-e CARBON_SCRATCH_DIR=`` with no mount behind it would be worse than nothing: a
    name that resolves to a directory that isn't there. Likewise, running as the
    invoking user with nothing mounted for it to read would only widen exposure for
    no benefit — the identity relaxation is conditioned on the mount, not offered
    unconditionally."""
    import subprocess

    from harness.sandbox import DOCKER_SCRATCH_MOUNT, SCRATCH_ENV_VAR, Sandbox

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    sb = Sandbox(trusted=False, prefer_docker=True)  # no scratch_dir
    sb._run_docker("echo ok", workdir=None)

    argv = captured["argv"]
    docker_flags = argv[: argv.index(sb.image)]
    assert DOCKER_SCRATCH_MOUNT not in " ".join(argv)
    assert not any(str(a).startswith(f"{SCRATCH_ENV_VAR}=") for a in argv)
    assert "65534:65534" in docker_flags
