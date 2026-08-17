"""Door control (ch-06).

Hard per-item size limits, applied before anything enters the prompt. A single
huge file or tool output can drown the window (distraction / confusion /
poisoning); clamping each item at the door is the cheapest defense.

Three strategies, registry-shaped to match ``compaction.py``'s ``_STRATEGIES``:
every strategy-shaped config knob uses the same shape, so an external improver
(or a person) reads one pattern once, not one pattern per knob plus a note that
this one is "simple enough" to skip it. The two inline strategies decide which
bytes to *lose*; ``offload_to_file`` refuses to lose any — the complete result
goes to a file in the session's private scratch and the inline excerpt carries a
virtual ref, so what the door cut stays one ``read_file`` away instead of gone.

This is the ONE door: an item passes it exactly once, and no other layer applies a
policy to the same text — not the sandbox, not the overflow shrink. That is what
keeps this module short. A second door has to work out what an earlier one did to
text it cannot authenticate, and every mechanism for answering that (a footer
pattern, then a secret token) was attack surface bought to solve a problem the
extra door created.

One door, two entrances, because it serves three trust domains. ``truncate`` cuts
text the harness or the user chose (an ``@path`` block, a generated checkpoint) and
rewrites nothing; ``truncate_tool_result`` cuts text a tool handed back, which
nobody vouches for, and defangs the copy the MODEL reads. An entrance apiece rather
than a flag, because the difference is not cosmetic: a spill has to hold the tool's
bytes exactly (a defanged ``git diff``, re-applied, wrote five corrupted lines and
reported success — an add-only hunk has no context for the patcher to mismatch), so
the rewrite may only ever touch the excerpt on its way to the model.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from harness.harness_config import CONFIG, TruncationPolicy

MAX_ITEM_CHARS = CONFIG.max_item_chars  # re-export; the value lives in the editable surface

SCRATCH_SCHEME = "scratch://"
# Spills live under the SESSION's private scratch (harness/session_env.py), never the
# workspace: the workspace is the user's durable repo and the sync/commit unit, and a
# complete tool-output copy inside it inherits both audiences. The transcript carries
# only the virtual ref below; read_file resolves it. Bounded disk: MAX_OFFLOAD_FILES.
_OFFLOAD_DIRNAME = "offload"


def spill_ref(filename: str) -> str:
    """The ONE place a spill's location becomes transcript text. Virtual on purpose:
    no absolute host path enters the transcript (machine-private, and unstable the
    moment execution moves into a container or remote session), and a forged ref can
    only ever name a file inside this session's own scratch inventory."""
    return f"{SCRATCH_SCHEME}{_OFFLOAD_DIRNAME}/{filename}"


def shell_ref(filename: str) -> str:
    """The bash route to the same artifact, as an UNEXPANDED variable reference.

    `scratch://` is carbon's internal identifier and only `read_file` resolves it.
    Iteration 5 measured what that costs: every one of the model's 32 attempts to
    reach a spill went through bash — grep, ls, a python one-liner — and every one
    failed, after which it fabricated the answer by re-deriving it. An identifier
    that looks like a path but is only honoured by a single tool is a private API
    handle wearing a path's clothes. Both consumers get an adapter.

    The variable's NAME is baked into this module (below); its value is set by
    ``harness.sandbox`` on the ``Sandbox`` that actually runs the command. What
    THIS function writes, and what the footer below assembles it into, never
    carries an expanded path — only the name. That is narrower than "never a host
    path lands in the transcript": a command run AGAINST the mount can still print
    one of its own, since the shell expands ``$CARBON_SCRATCH_DIR`` before
    ``cut``/``grep``/etc. ever run, so a stale ref's "no such file" names the real
    path in THAT command's own stderr. Not scrubbed there either — a blanket
    rewrite can't tell "a path leaked into an error" from "a path is legitimately
    part of file content a command is displaying," and would risk corrupting the
    latter to fix the former.
    """
    # Deferred, not module-level: this module is imported BY harness.sandbox's own
    # dependency chain (sandbox -> tools -> limits, for SCRATCH_SCHEME), so a
    # module-level `from harness.sandbox import ...` here would close that into a
    # real three-module cycle. It would even resolve under some import orders — but
    # not the one where something imports harness.tools first, since tools.py pauses
    # on its own `from harness.limits import SCRATCH_SCHEME` line before its `Tool`
    # class is defined, and this module would then be asking the still-loading tools
    # module for it. Deferred to call time, every module involved has already
    # finished loading, regardless of which of the three a caller touches first.
    from harness.sandbox import SCRATCH_ENV_VAR

    return f"${SCRATCH_ENV_VAR}/{_OFFLOAD_DIRNAME}/{filename}"


# Bounded disk. The filename is a content hash, so nothing ever overwrites anything;
# a long session still spills one file per over-budget result until session close.
# Keep the newest N — the older a spill is, the less likely anything is still
# following its footer.
#
# EPHEMERAL sessions only. A DURABLE session (harness/session_env.py, Task 3) grows
# unbounded by this constant across a reopen instead — it is bounded by the SESSION's
# own lifetime (harness.memory.delete_session / delete_session_scratch), not by this
# count, because its footers are read by a transcript that outlives any one process.
# See ``_prune``'s ``durable`` parameter.
MAX_OFFLOAD_FILES = 64
_PART_SUFFIX = ".part"  # a spill mid-write; only ever named between mkstemp and rename

# Every file this process is responsible for: the spills its transcript now points at,
# and the temp files it is still writing. ``_prune`` skips them — a footer whose file
# was reclaimed underneath it is worse than a directory a little over its bound, since
# the model then follows a live-looking route to "no such file" with nothing telling it
# the copy ever existed. An EPHEMERAL session that spills more than MAX_OFFLOAD_FILES
# times therefore exceeds the bound until the process exits; that is the intended trade.
#
# Per-PROCESS is exactly what makes that reasoning WRONG for a DURABLE session: this
# set starts empty again on every reopen, so a durable session's own earlier spills —
# real, still named by its persisted transcript, just written by a now-dead process —
# are indistinguishable from "a previous run's strays" by membership in this set alone.
# ``_prune``'s ``durable`` parameter is what tells the two apart instead of relying on
# ``_OURS``, which structurally cannot: see test_durable_spills_survive_their_own_
# sessions_reopen (tests/test_offload_strategy.py), which reproduces the reviewer's
# measured "100 spills, reopen, spill one more, lose 37" without it.
_OURS: set[Path] = set()


def clamp(text: str, max_chars: int = MAX_ITEM_CHARS) -> str:
    """Truncate an item to ``max_chars``, with a marker noting what was dropped."""
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}\n…[truncated {dropped} chars]"


@dataclass(frozen=True)
class _Text:
    """The two copies a door handles: what the model reads, and what a spill keeps.

    They differ for exactly one reason — tool output is defanged on the way to the
    model and never on the way to disk — so ``shown`` is what every strategy cuts and
    ``complete`` is what ``offload_to_file`` writes and what its footer counts. For
    ``truncate``'s own callers the two fields are the same string.
    """

    shown: str
    complete: str


def _split_budget(content_budget: int, tail_fraction: float) -> tuple[int, int]:
    """The head/tail char split shared by ``head_tail`` and the offload excerpt."""
    tail_chars = max(1, int(content_budget * tail_fraction))
    return max(1, content_budget - tail_chars), tail_chars


def _cut_head_tail(text: str, content_budget: int, tail_fraction: float, marker: str) -> str:
    """The head+tail cut itself, on a plain string — the shape three callers share."""
    head_chars, tail_chars = _split_budget(content_budget, tail_fraction)
    return text[:head_chars] + marker + "\n" + text[-tail_chars:]


def _keep_head(
    text: _Text,
    content_budget: int,
    _tail_fraction: float,
    marker: str,
    _scratch: Path | None,
    _durable: bool,
) -> str:
    return text.shown[:content_budget] + marker


def _head_tail(
    text: _Text,
    content_budget: int,
    tail_fraction: float,
    marker: str,
    _scratch: Path | None,
    _durable: bool,
) -> str:
    return _cut_head_tail(text.shown, content_budget, tail_fraction, marker)


# --- the recovery footer ------------------------------------------------------
#
# One line, appended last, and the only thing standing between a cut result and the
# whole one. Two parties read it: the model (as an instruction) and ``_defang`` (to
# stop untrusted output from forging one). They must agree, so the pattern and the
# writer live side by side.
#
# The invariant, and the reason there is no third reader: **footer-shaped text may be
# matched in order to QUOTE it, never in order to BELIEVE it.** A tool result is
# attacker-controlled, so a footer-SHAPED line proves nothing about who wrote it — and
# the one revision that asked anyway handed the model a forged ``read_file``
# instruction with the harness's authority while writing the real payload nowhere.
# (test_offload_strategy enforces this structurally: the pattern may appear in a
# substitution, never in a branch condition.)
_LOOKALIKE_RE = re.compile(r"\[(?=Showing \d+ of \d+ chars\. (?:Full output|Output as captured))")
# The marker every cut in this harness writes, and the one string here that IS read in
# a branch — which is allowed, and worth being precise about. The invariant above bans
# believing a line that claims authority; this one only ever WEAKENS a claim of ours. A
# result carrying it was cut before this door saw it (the sandbox's ceiling, a tool's
# own paging), so what we spill is a cut of a cut, and "Full output" would be a lie
# about the command even while every count is true of the file. Planting one buys an
# attacker a more cautious footer and nothing else.
_UPSTREAM_CUT = "…[truncated "
# The widest footer this module can write, over the single-line case, the paging
# case, and every claim, with counts up to ten digits (a 10 GB result — this process
# could not hold one) and a filename at its fixed 20-char width (16 hex + ".txt").
#
# Re-measured at 389 (up from the prior revision's correctly-measured 343): adding
# the shell route flips which branch is widest — the PAGING case is now it, not the
# single-line one, because "or search it in a shell, e.g. grep -n '<what you need>'
# "$CARBON_SCRATCH_DIR/…"" outruns the single-line "or slice it in a shell, e.g.
# head -c <n> "$CARBON_SCRATCH_DIR/…"" by more than the paging call's own
# ", start_line=1, end_line=<n>" gave up.
#
# Pinned exactly at the measured value, no added headroom — a deliberate choice,
# not the default. The prior revision's 450-vs-343 gap was ALSO deliberate (a
# margin chosen on purpose, not a number that had gone stale — re-checked against
# that revision's own writer and both of its claims held), so this is not "fixing
# drift"; it is trading that margin away on purpose. Slack is invisible: a future
# wording change that grows the footer by less than an unused margin passes
# silently, and only the change that finally crosses the old ceiling shows up as a
# failure — pinned to whoever last touched this file, not whoever actually grew it.
# Pinning exactly means every wording change re-measures, immediately, correctly
# attributed to itself.
#
# Two consumers size themselves from it: the overflow shrink's tail floor
# (agent.py) and the accept gate's door-control allowance (tasks/checks.py). A
# footer that outgrew it would silently cost a pointer in the first and read as a
# door-control regression in the second, so test_offload_strategy measures this
# writer against this number directly rather than anyone re-deriving it by eye.
MAX_FOOTER_CHARS = 389


def _route(line_count: int, ref: str, shell: str) -> str:
    """The way back into a spill, for each of the two tools that can actually reach
    for one.

    Two routes, not one. The old second route (``search_text``, with its own
    ``pattern`` spelled out) depended on a real workspace path a walk could reach; a
    virtual ``scratch://`` ref has none, so it fell away and, for one revision,
    ``read_file`` was what was left — the only call offered, because it is the only
    tool that resolves the ref. That measured badly: iteration 5's transcripts
    showed 32 of 32 attempts to recover a spill went through bash (``grep``,
    ``ls -F``, a python one-liner), none of which can resolve ``scratch://``, and
    every one failed — task E4 (recover a truncated artifact) scored 0/10. The
    model reached for the tool it actually had, which this footer did not name. The
    shell route names it: ``shell_ref`` builds the same file's path as an
    UNEXPANDED ``$CARBON_SCRATCH_DIR`` reference, so quoting it here is never a
    host path, only ever the variable's name (harness/sandbox.py sets the value on
    the ``Sandbox`` that runs the command, not on this text).

    No range is computed for the read_file page. A suggested first page used to be
    sized against this door so its result could not come back over budget, but
    read_file's result carries a continuation hint, and a hint downgrades the
    strategy away from offload (see ``_door``) — so an over-budget page returns an
    excerpt plus "ask for a smaller range", never another spill. That mechanism
    prevents the circle for every page the model asks for; a computed first page
    only ever saved the first round trip.

    A single line is a different problem, not a smaller one: read_file pages by
    LINE, so there is no line boundary inside one line for it to page to — a
    start_line/end_line range would just be a route to "no such range", and that
    half of the old wording still holds. What no longer holds is the conclusion
    drawn from it: a shell slices by BYTE, so ``head -c``/``cut``/``dd`` can reach
    the middle of one long line even though read_file structurally cannot. That is
    the single-line gap iteration 5 named, closed the same way as the multi-line
    case — by naming a tool that can actually do it, rather than naming no call at
    all.

    The byte count is a placeholder (``<n>``), same as ``end_line=<n>`` above it,
    never a literal figure. An earlier revision wrote ``cut -c1-4000`` — a number
    with no relationship to THIS call's actual budget, so a small `tool_output`
    policy could suggest a slice bigger than the door it would re-enter through,
    spilling a second file to recover from the first. The door has no continuation
    hint of its own to fall back on here the way a re-submitted read_file page
    does (see the paragraph above) — the safe number depends on the budget in
    force *when the model runs the command*, which this footer, written now,
    cannot see ahead of. Naming no number is the honest version of the same
    choice already made for the read_file page above.
    """
    if line_count <= 1:
        return (
            f"one long line: read_file(path='{ref}') returns it whole, or slice it in "
            f'a shell, e.g. head -c <n> "{shell}"'
        )
    return (
        f"read_file(path='{ref}', start_line=1, end_line=<n>) to page it, "
        f"or search it in a shell, e.g. grep -n '<what you need>' \"{shell}\""
    )


def _footer(ref: str, shell: str, *, shown: int, total: int, lines: int, cut_upstream: bool) -> str:
    """Name the complete copy and hand the model the exact next call to make — one
    for each tool that can reach it (``read_file`` via ``ref``, a shell via ``shell``).

    Counts, not text: every number here describes the file on disk, and the widest
    footer this can produce is pinned by ``MAX_FOOTER_CHARS``, which a test measures
    against this writer directly.

    ``cut_upstream`` is the one thing the door cannot verify for itself. What it wrote
    is the complete result THE TOOL RETURNED; whether that was the whole command's
    output depends on layers above (the sandbox ceiling cuts a 12MB stream at 10MB, and
    the ``…[truncated]`` it leaves sits at the file's midpoint where neither the excerpt
    nor a first page reaches it). Where that marker is present the claim is downgraded
    rather than the counts changed — the counts were always right about the file.
    """
    claim = (
        "Output as captured, already truncated upstream (…[truncated …] sits inside it)"
        if cut_upstream
        else "Full output"
    )
    return (
        f"\n[Showing {shown} of {total} chars. {claim} ({lines} lines): "
        f"{ref} — {_route(lines, ref, shell)}]"
    )


def _defang(text: str) -> str:
    """Break footer lookalikes in the copy of untrusted text the MODEL reads.

    The footer is an instruction the *harness* issues. Tool output that forges one is
    aiming ``read_file`` at a file of its own choosing, so lookalikes are relabeled as
    the quoted text they are. (``read_file``'s workspace confinement and secret-file
    refusal are still the boundary; this removes the invitation, not the wall.)

    Accepted cost: text that is *honestly* footer-shaped — this file quoted in a log, a
    transcript of an earlier session, one of our own footers coming back around — is
    relabeled too. That is not a heuristic failing, it is the invariant working: nothing
    downstream believes a footer, so nothing here has to tell a real one from a forgery.

    What it must NOT touch is anything that gets written or re-applied. A defanged
    ``git diff`` spilled to a file and re-applied wrote five corrupted lines to disk
    without a single error, because an add-only hunk has no context line for ``git
    apply`` to mismatch — so this runs on the excerpt only, never on the spill, and
    never on ``@path`` content the user chose and a later ``apply_patch`` has to match.
    Paging a spill back in is itself a tool result, so it comes through here on the way
    in; the bytes on disk stay exactly what the tool returned.
    """
    return _LOOKALIKE_RE.sub("[quoted tool output: ", text)


# --- writing the complete copy ------------------------------------------------
class _OffloadUnavailable(Exception):
    """No complete copy could be written; the caller degrades to an inline excerpt."""


def _prune(offload_dir: Path, *, durable: bool = False) -> None:
    """Keep the newest ``MAX_OFFLOAD_FILES`` spills, drop older strays and dead temps.

    Never anything in ``_OURS``: this session's transcript names those files, so
    reclaiming one turns a live footer into "no such file". Strays (a previous run's
    spills) fill whatever room is left, oldest out first. Abandoned ``.part`` files go
    unconditionally — a write killed between mkstemp and rename, whose name no reader
    was ever given — and they have to go through here, or a process killed mid-write
    leaks past the disk bound the ``*.txt`` sweep is enforcing.

    ``durable=True`` skips the ``*.txt`` reclaim entirely (the abandoned-``.part``
    sweep above still runs either way — nothing, durable or not, ever names one of
    those). ``_OURS`` is a module-level, per-PROCESS set: empty again on every
    reopen, so it cannot tell "a previous PROCESS's strays" (safe to reclaim) apart
    from "a previous RUN of this same durable SESSION's real spills, still named by
    its persisted transcript" (must not be reclaimed) — the two look identical by
    membership in ``_OURS`` alone. ``durable`` is the caller's answer to that
    question instead: a durable scratch is bounded by the session's own lifetime
    (``harness.memory.delete_session`` / ``delete_session_scratch``), not by this
    count. See test_durable_spills_survive_their_own_sessions_reopen.
    """
    strays: list[Path] = []
    ours = 0
    for path in offload_dir.iterdir():
        if path in _OURS:
            ours += 1
            continue
        if path.suffix == _PART_SUFFIX:
            with suppress(OSError):
                path.unlink()
        elif path.suffix == ".txt":
            strays.append(path)
    if durable:
        return
    keep = max(0, MAX_OFFLOAD_FILES - ours)
    strays.sort(key=lambda p: p.stat().st_mtime)
    for stale in strays[: max(0, len(strays) - keep)]:
        with suppress(OSError):
            stale.unlink()


def _write_atomically(target: Path, payload: bytes) -> None:
    """Write via a private temp file and ``os.replace``.

    ``rename`` REPLACES whatever holds the target name instead of following it, which
    matters because the name is a deterministic hash: anything that can write into
    the scratch directory can pre-place it as a symlink to any file on the host, and
    a plain write would dutifully overwrite that file. The rename is also atomic, so
    the model reading the path on its very next turn never sees a half-written file.

    The mode is mkstemp's own 0600. An earlier pass widened it to 0644 for a sandboxed
    shell running as another uid — but the shell the coding wiring actually builds is
    ``Sandbox(trusted=True)``, the same uid, and the chmod overrode the user's umask to
    publish the *complete* tool output (a printenv, a token-bearing log) to every
    account on the host. Respect the umask.

    A route that needs a different uid to read this file DOES exist now — ch-08's
    Docker backend, reading its own session's scratch through a bind mount on a
    native Linux host, where the mount preserves the host's uid instead of remapping
    it the way Docker Desktop's macOS backend does. The fix there is still not a
    wider mode: ``harness/sandbox.py`` runs that one container AS the invoking uid
    (``--user {uid}:{gid}``, only when a scratch dir is actually mounted) instead of
    its usual fixed unprivileged one, so the SAME 0600 file that already excludes
    every other account on the host is exactly what that container needs too. The
    mode stays put; only the container's borrowed identity moves.
    """
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=_PART_SUFFIX)
    part = Path(tmp)
    _OURS.add(part)  # in flight: a concurrent fan_out worker's prune must not take it
    try:
        try:
            handle = os.fdopen(fd, "wb")  # the caller encoded; no second guess at it here
        except BaseException:
            os.close(fd)  # fdopen failing leaves the descriptor ours to close
            raise
        with handle:
            handle.write(payload)
        os.replace(tmp, target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        _OURS.discard(part)


def _holds(target: Path, payload: bytes) -> bool:
    """Does a REGULAR file at ``target`` already hold exactly ``payload``?

    The name is a content hash, but a hash is a claim about content that anything
    able to write into the scratch directory can make too: it can pre-place
    ``<known hash>.txt`` with content of its choosing, and skipping the write on "a
    file is already there" would have the footer label that content "Full
    output". So verify rather than assume — one read of a file we would otherwise be
    rewriting anyway. A symlink, a directory, or a fifo at the name is not our copy
    either, whatever it contains.
    """
    if target.is_symlink() or not target.is_file():
        return False
    try:
        return target.read_bytes() == payload
    except OSError:
        return False


# A containment check DOES belong here, even though the scratch parent is
# mkdtemp-private to the harness (session_env.py) rather than a path inside a
# repository someone else controls. A prior revision reasoned that away — "no
# workspace-owner symlink attack left to defend against" — but that reasoning
# only covers a HOSTILE PRE-EXISTING workspace; it misses that the model itself
# can plant the link mid-session. Carbon's coding wiring runs
# ``Sandbox(trusted=True)``, whose own docstring says it does not isolate the
# filesystem, and the scratch prefix (``carbon-scratch-``) is greppable in
# ``$TMPDIR`` — so nothing stops a command the model runs from symlinking
# ``<scratch>/offload`` to any directory it can write before this call ever
# fires. Verified rather than assumed: ``mkdir(parents=True, exist_ok=True)``
# does NOT raise on a pre-existing symlink-to-directory — ``is_dir()`` follows
# the link, sees a directory, and swallows the ``FileExistsError`` — so a spill
# would write straight through the link, and ``SessionEnvironment.cleanup()``'s
# ``shutil.rmtree`` then removes only the link (rmtree does not recurse into a
# symlinked subdirectory), leaving the spilled bytes on disk outside the
# session while ``scratch_root.exists()`` reports False to everything checking
# after the fact.
def _offload_dir(scratch_dir: Path | None) -> Path:
    if not scratch_dir:
        raise _OffloadUnavailable("no scratch storage to write under")
    root = Path(scratch_dir)
    landed = root / _OFFLOAD_DIRNAME
    # Checked before creating anything: mkdir(exist_ok=True) FOLLOWS a symlinked
    # directory instead of raising, and by the time an after-the-fact check could
    # refuse, the bytes are already outside the session.
    if root.is_symlink() or landed.is_symlink():
        raise _OffloadUnavailable("scratch directory is a symlink")
    landed.mkdir(parents=True, exist_ok=True)
    resolved = landed.resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise _OffloadUnavailable("offload directory escapes the scratch root")
    # NARROWED, not closed. A swap landing after this resolve — between here and
    # `_write_atomically`'s mkstemp, which re-walks `offload` BY NAME — still escapes,
    # and the footer would then advertise a route to a file outside the session.
    # Closing that needs O_NOFOLLOW / dirfd-relative operations, which is a larger
    # change than this guard. What is dead is the deterministic one-`ln -s` attack:
    # both halves above are pinned by tests, including a simulated race. Do not read
    # the TOCTOU test's existence as "races handled".
    return landed


def _spill(text: str, scratch_dir: Path | None, *, durable: bool = False) -> str:
    """Write the complete text under the session scratch; return its VIRTUAL ref.

    The filename is a content hash — deterministic per call, so a retried or repeated
    identical result re-lands on the same file instead of littering scratch. The
    encoding is explicit here and on ``read_file``'s side, so the round trip does not
    depend on the host's locale.

    ``durable`` is forwarded to ``_prune`` unchanged — see its docstring for why the
    per-process ``_OURS`` set cannot make this call on its own for a session whose
    transcript (and therefore whose live footers) can outlive this process.
    """
    landed = _offload_dir(scratch_dir)
    payload = text.encode("utf-8")
    target = landed / f"{hashlib.sha256(payload).hexdigest()[:16]}.txt"
    if not _holds(target, payload):
        _write_atomically(target, payload)
    # Written now or found already correct, the footer about to be returned names this
    # file: from here on pruning must leave it alone.
    _OURS.add(target)
    with suppress(OSError):  # housekeeping must never cost us the copy we just wrote
        _prune(landed, durable=durable)
    return spill_ref(target.name)


def _note(marker: str, note: str) -> str:
    """Add a clause to a truncation marker (which always ends in ``]``)."""
    return f"{marker[:-1]} {note}]"


def _why(exc: Exception) -> str:
    """A short cause for the marker. An OSError's ``str()`` carries the absolute path
    it failed on; model-visible text — and anything that records the transcript —
    must not, so the plain ``strerror`` is preferred where there is one."""
    return f"offload unavailable: {getattr(exc, 'strerror', None) or exc}"


def _offload_to_file(
    text: _Text,
    content_budget: int,
    tail_fraction: float,
    marker: str,
    scratch_dir: Path | None,
    durable: bool,
) -> str:
    """Recoverable truncation: write the COMPLETE text to a session scratch file, keep
    the ``head_tail`` excerpt inline, and append a footer naming the file.

    The footer names a VIRTUAL ``scratch://`` ref, never a host path: that is what the
    model's own ``read_file`` resolves, and an absolute host path in a transcript would
    leak machine-private detail into anything that records it — and go stale the moment
    execution moves into a container or remote session.

    Best-effort by construction. This runs mid-turn, after the assistant's tool_calls
    are already in the transcript, so raising would abandon the turn with an unanswered
    tool call and an unsaved session — strictly worse than the inline excerpt every
    other strategy would have produced. A failed write degrades to exactly that, and
    the marker says so instead of pretending a file exists.

    ``durable`` is forwarded to ``_spill`` unchanged — see ``_prune`` for why a
    durable session's own earlier spills must never be pruned on the strength of
    this process's (empty, on a reopen) ``_OURS`` set alone.
    """
    try:
        # The tool's bytes, not the model's copy of them: this file is re-read, diffed,
        # applied and checksummed, and a relabeled line inside it is silent corruption.
        ref = _spill(text.complete, scratch_dir, durable=durable)
    except (_OffloadUnavailable, OSError, UnicodeError) as exc:
        note = _note(marker, _why(exc))
        return _cut_head_tail(text.shown, content_budget, tail_fraction, note)
    head_chars, tail_chars = _split_budget(content_budget, tail_fraction)
    excerpt = _cut_head_tail(text.shown, content_budget, tail_fraction, marker)
    # The shown-count is the size actually produced (defanging happened before the cut,
    # so it cannot push the excerpt past the budget); every other count describes the
    # file, which is what the model is being sent to read.
    complete = text.complete
    # Same file, same filename — `ref` is `spill_ref(target.name)` (the ONE place a
    # spill's name becomes a virtual ref; see `_spill`), and `Path(...).name` recovers
    # that filename back out of it rather than plumbing a second copy through the
    # return value just for this call.
    shell = shell_ref(Path(ref).name)
    return excerpt + _footer(
        ref,
        shell,
        shown=head_chars + tail_chars,
        total=len(complete),
        lines=len(complete.splitlines()),
        cut_upstream=_UPSTREAM_CUT in complete,
    )


@dataclass(frozen=True)
class _TruncationStrategy:
    # The marker's POSITION differs per strategy — trailing for keep_head, inserted
    # between head and tail for head_tail and the offload excerpt — so each strategy
    # places it, rather than the caller gluing one fixed shape onto every strategy's
    # output. offload_to_file also appends its own footer AFTER the excerpt: only
    # the strategy knows the path it wrote, so only it can name the recovery route.
    # Trailing bool: ``durable`` (Task 3 follow-up) — ignored by the two inline
    # strategies (nothing prunable about text that never touches disk), consumed
    # only by offload_to_file, which is the only one that can reach ``_prune``.
    apply: Callable[[_Text, int, float, str, Path | None, bool], str]
    # Does the strategy leave a recovery route of its own? The caller's
    # continuation_hint is then dropped rather than competing with it (see truncate).
    routes_recovery: bool = False


_TRUNCATION_STRATEGIES: dict[str, _TruncationStrategy] = {
    "keep_head": _TruncationStrategy(_keep_head),
    "head_tail": _TruncationStrategy(_head_tail),
    "offload_to_file": _TruncationStrategy(_offload_to_file, routes_recovery=True),
}


def recut(text: str, max_chars: int, tail_fraction: float) -> str:
    """A POST-DOOR RE-CUT of text that has already been through a door.

    Not a door itself, and not a general truncation entry point: no policy, no strategy,
    no file, no defanging. Its callers (agent.py's overflow shrink, observability.py's
    bounded retention copies) are all handling text a door already sized, where cutting
    again is the whole job — defanging a second time would relabel our own footer as
    quoted text and blunt the route it names, and offloading again would file an
    *excerpt* as if it were the complete output.

    The arguments go through ``TruncationPolicy``'s validation rather than a second copy
    of those rules, because they used to go through none: ``tail_fraction=2.0`` asked
    for a tail twice the budget and got it, returning 61 chars for a budget of 10.
    """
    policy = TruncationPolicy("head_tail", max_chars, tail_fraction)
    if len(text) <= policy.budget:
        return text
    marker = f"\n…[truncated {len(text) - policy.budget} chars using head_tail.]"
    return _cut_head_tail(text, policy.budget, policy.tail_fraction, marker)


def strategy_names() -> frozenset[str]:
    """The truncation strategies that actually exist here — read from the registry, so
    a caller validating a name checks against the implementations rather than a second,
    hand-maintained copy of the list."""
    return frozenset(_TRUNCATION_STRATEGIES)


def truncate(
    text: str,
    policy: TruncationPolicy,
    *,
    budget: int | None = None,
    continuation_hint: str | None = None,
    scratch_dir: str | Path | None = None,
    durable: bool = False,
) -> str:
    """Apply one vetted truncation strategy to text the harness or the user chose.

    An ``@path`` block, a generated checkpoint: content nobody is attacking us with, and
    content a later ``apply_patch``/``edit_file`` may have to match character for
    character. So it is cut and never rewritten. Tool output goes through
    ``truncate_tool_result`` instead.

    ``durable`` mirrors ``scratch_dir``: it is meaningless unless this call reaches
    ``offload_to_file`` (today, no caller of THIS entrance passes a ``scratch_dir``,
    so it never does — see ``truncate_tool_result`` for the one that matters), and is
    accepted here anyway so the two entrances share one signature rather than one of
    them silently defaulting to "always prunable" the day a caller changes that.
    """
    return _door(
        _Text(text, text),
        policy,
        budget=budget,
        continuation_hint=continuation_hint,
        scratch_dir=scratch_dir,
        durable=durable,
    )


def truncate_tool_result(
    text: str,
    policy: TruncationPolicy,
    *,
    budget: int | None = None,
    continuation_hint: str | None = None,
    scratch_dir: str | Path | None = None,
    durable: bool = False,
) -> str:
    """The same door for TOOL OUTPUT — the untrusted domain, defanged on the way in.

    The model's copy is defanged; the spill is not. Both halves matter. Defanging
    before the cut, so an attacker cannot lengthen the excerpt past the budget by
    packing it with lookalikes (quoting one adds ~20 chars; measured 1.6x the budget
    from chosen input). And on the excerpt only, so the complete copy on disk is the
    tool's own bytes — the thing that gets diffed, applied and hashed.

    Every result comes through here, including under-budget ones and whatever strategy
    is selected: the guard that used to sit inside one strategy's over-budget path was
    inert in three of four realistic cases and never fired under the shipped default.

    ``durable`` is the caller's ``session_env.durable`` (agent.py passes it straight
    through): whether a spill this call makes must survive being pruned by a LATER,
    different process reopening the same session. See ``_prune``.
    """
    return _door(
        _Text(_defang(text), text),
        policy,
        budget=budget,
        continuation_hint=continuation_hint,
        scratch_dir=scratch_dir,
        durable=durable,
    )


def _door(
    text: _Text,
    policy: TruncationPolicy,
    *,
    budget: int | None = None,
    continuation_hint: str | None = None,
    scratch_dir: str | Path | None = None,
    durable: bool = False,
) -> str:
    """The door itself, shared by both trust domains.

    ``budget`` lets a tool declare a smaller result limit without changing the
    selected strategy. The marker is deliberately actionable: losing bytes is
    unavoidable, losing the fact that bytes were lost is not — and under
    ``offload_to_file`` the bytes are not even lost, only moved to a file in the
    session's private scratch that the marker's footer names. The inline strategies
    ignore ``scratch_dir`` (and ``durable``); offload degrades to an inline excerpt
    without a ``scratch_dir`` rather than guessing a directory (the process cwd is
    not the session's scratch). ``durable`` only matters to offload too — it is
    whether the file just written must survive being pruned by a LATER process that
    reopens this same session (harness/session_env.py; see ``_prune``).

    This is the one door: every result the model sees comes through here exactly once,
    which is what lets the rules below be this short.
    """
    max_chars = budget or policy.budget
    if len(text.shown) <= max_chars:
        return text.shown
    strat = _TRUNCATION_STRATEGIES.get(policy.strategy)
    if strat is None:
        raise ValueError(f"unsupported truncation strategy: {policy.strategy}")
    name = policy.strategy
    if continuation_hint and strat.routes_recovery:
        # One recovery route, not two — and where the model can simply ask again, the
        # caller's hint is the better one. ``continuation_hint`` is the caller saying
        # this text is already a view of a file on disk (agent.py sets it for
        # read_file): copying it would hand the model a duplicate of what it is already
        # holding, and the next page would duplicate the duplicate, one file deeper
        # every time. Exactly why file_injection is not offered this strategy at all —
        # what is already re-openable needs a smaller range, not another copy.
        #
        # Accepted cost: a WHOLE-file read (no start_line/end_line) has no header of
        # its own, so what comes back here is a bare body plus the hint — no totals, no
        # line count, nothing but "ask for a range". Improving that is read_file's job,
        # not this door's: the source is on disk and re-readable either way.
        name, strat = "head_tail", _TRUNCATION_STRATEGIES["head_tail"]
    # A strategy that routes its own recovery (the offload footer) suppresses the hint,
    # because the model can only follow one.
    hint = "" if strat.routes_recovery or not continuation_hint else f" {continuation_hint}"
    # Counted on the copy being cut, so the number describes what the model lost here —
    # under offload the footer's own totals then describe the file, which is bigger news.
    marker = f"\n…[truncated {len(text.shown) - max_chars} chars using {name}.{hint}]"
    # Falsy, not just ``None``: ``Path("")`` normalizes to ``.``, and a truthy-but-empty
    # scratch_dir wrapped there before this check would spill into the process's own
    # cwd instead of degrading — the empty string has to be caught here, before it is
    # ever wrapped in a Path, because a Path object is truthy no matter what it names.
    scratch = Path(scratch_dir) if scratch_dir else None
    return strat.apply(text, max_chars, policy.tail_fraction, marker, scratch, durable)
