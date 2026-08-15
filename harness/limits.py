"""Door control (ch-06).

Hard per-item size limits, applied before anything enters the prompt. A single
huge file or tool output can drown the window (distraction / confusion /
poisoning); clamping each item at the door is the cheapest defense.

Three strategies, registry-shaped to match ``compaction.py``'s ``_STRATEGIES``:
every strategy-shaped config knob uses the same shape, so an external improver
(or a person) reads one pattern once, not one pattern per knob plus a note that
this one is "simple enough" to skip it. The two inline strategies decide which
bytes to *lose*; ``offload_to_file`` refuses to lose any — the complete result
goes to a workspace file and the inline excerpt carries the path, so what the
door cut stays one ``read_file`` away instead of gone.

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

# Where offloaded results live, relative to the agent's workspace root. Inside the
# workspace on purpose: the model's own read_file resolves relative paths against
# that root, so the marker's path works verbatim. The files share the workspace's
# lifecycle (a throwaway worktree, a benchmark container) and are never meant to
# be committed.
OFFLOAD_SUBDIR = Path(".carbon") / "offload"
# Bounded disk. The filename is a content hash, so nothing ever overwrites anything;
# a long session on a persistent workspace would spill one file per over-budget
# result forever. Keep the newest N — the older a spill is, the less likely anything
# is still following its footer.
MAX_OFFLOAD_FILES = 64
_PART_SUFFIX = ".part"  # a spill mid-write; only ever named between mkstemp and rename

# Every file this process is responsible for: the spills its transcript now points at,
# and the temp files it is still writing. ``_prune`` skips them — a footer whose file
# was reclaimed underneath it is worse than a directory a little over its bound, since
# the model then follows a live-looking route to "no such file" with nothing telling it
# the copy ever existed. A session that spills more than MAX_OFFLOAD_FILES times
# therefore exceeds the bound until the process exits; that is the intended trade.
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
    text: _Text, content_budget: int, _tail_fraction: float, marker: str, _root: Path | None
) -> str:
    return text.shown[:content_budget] + marker


def _head_tail(
    text: _Text, content_budget: int, tail_fraction: float, marker: str, _root: Path | None
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
# The widest footer this module can write, over every route and every claim, with
# counts up to ten digits (a 10 GB result — this process could not hold one). Measured
# at 384 today. Two consumers size themselves from it: the overflow shrink's tail floor
# (agent.py) and the accept gate's door-control allowance (tasks/checks.py). A footer
# that outgrew it would silently cost a pointer in the first and read as a door-control
# regression in the second, so test_offload_strategy measures this writer against this
# number rather than anyone re-deriving it by eye.
MAX_FOOTER_CHARS = 450


def _route(line_count: int, rel: Path) -> str:
    """The two ways back into the spill, in the units the tools actually take.

    Both are offered because they answer different questions: paging reads the file in
    order, searching jumps to the one line that matters. The search route needs its
    ``pattern`` spelled out — the harness's scratch directory is kept out of an
    undirected walk (tools.py), so a bare query would report no matches on the very
    file this footer is pointing at.

    No range is computed for the page. A suggested first page used to be sized against
    this door so its result could not come back over budget, but read_file's result
    carries a continuation hint, and a hint downgrades the strategy away from offload
    (see ``_door``) — so an over-budget page returns an excerpt plus "ask for a
    smaller range", never another spill. That mechanism prevents the circle for every
    page the model asks for; a computed first page only ever saved the first round trip.

    The shell slice is offered, never promised: a subagent's registry is read-only
    tools with no bash in it, and naming a tool the reader may not have should read as
    "not for you" rather than as a dead end.
    """
    if line_count <= 1:
        return (
            f"one long line, so line paging cannot reach into it; a shell can slice it, "
            f"e.g. sed -n '1p' {rel} | cut -c1-4000"
        )
    return (
        f"read_file(path='{rel}', start_line=1, end_line=<n>) to page it, "
        f"or search_text(query='<what you need>', pattern='{rel}') to jump to a line"
    )


def _footer(rel: Path, *, shown: int, total: int, lines: int, cut_upstream: bool) -> str:
    """Name the complete copy and hand the model the exact next call to make.

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
        f"{rel} — {_route(lines, rel)}]"
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


# A line that already covers the spills; anything else is somebody's own file.
_IGNORE_COVERAGE = frozenset({"*", "**", "offload", "offload/", "/offload", "/offload/"})


def _mark_ignored(carbon_dir: Path) -> None:
    """Keep spilled output out of the user's history.

    The agent works inside a real repository, and ``.carbon`` is the harness's own
    scratch directory in it — without this, one ``git add -A`` after a long session
    commits the agent's own truncated tool results. An existing ``.gitignore`` is never
    clobbered (it may be the user's, or another tool's), but one that does not actually
    cover the spills — say a project-local ``extensions/`` line and nothing else — is
    appended to, because "a file exists" was never the property worth checking.
    """
    marker = carbon_dir / ".gitignore"
    if marker.is_symlink():
        # Never write through a link we did not create: a dangling one makes this
        # create its target, which is somewhere outside .carbon by definition.
        return
    try:
        existing = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        marker.write_text("*\n", encoding="utf-8")
        return
    except (OSError, UnicodeError):
        return
    if any(line.strip() in _IGNORE_COVERAGE for line in existing.splitlines()):
        return
    with marker.open("a", encoding="utf-8") as handle:
        handle.write(("" if existing.endswith("\n") or not existing else "\n") + "offload/\n")


def _prune(offload_dir: Path) -> None:
    """Keep the newest ``MAX_OFFLOAD_FILES`` spills, drop older strays and dead temps.

    Never anything in ``_OURS``: this session's transcript names those files, so
    reclaiming one turns a live footer into "no such file". Strays (a previous run's
    spills) fill whatever room is left, oldest out first. Abandoned ``.part`` files go
    unconditionally — a write killed between mkstemp and rename, whose name no reader
    was ever given — and they have to go through here, or a process killed mid-write
    leaks past the disk bound the ``*.txt`` sweep is enforcing.
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
    keep = max(0, MAX_OFFLOAD_FILES - ours)
    strays.sort(key=lambda p: p.stat().st_mtime)
    for stale in strays[: max(0, len(strays) - keep)]:
        with suppress(OSError):
            stale.unlink()


def _write_atomically(target: Path, payload: bytes) -> None:
    """Write via a private temp file and ``os.replace``.

    ``rename`` REPLACES whatever holds the target name instead of following it, which
    matters because the name is a deterministic hash: a workspace the agent does not
    own can pre-place it as a symlink to any file on the host, and a plain write would
    dutifully overwrite that file. The rename is also atomic, so the model reading the
    path on its very next turn never sees a half-written file.

    The mode is mkstemp's own 0600. An earlier pass widened it to 0644 for a sandboxed
    shell running as another uid — but the shell the coding wiring actually builds is
    ``Sandbox(trusted=True)``, the same uid, and the chmod overrode the user's umask to
    publish the *complete* tool output (a printenv, a token-bearing log) to every
    account on the host. Respect the umask; the one route that needs a wider mode
    doesn't exist here.
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

    The name is a content hash, but a hash is a claim about content that the
    *workspace's owner* can make too: a repository can pre-place
    ``.carbon/offload/<known hash>.txt`` with content of its choosing, and skipping the
    write on "a file is already there" would have the footer label that content "Full
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


def _offload_dir(workspace_root: Path) -> Path:
    """The resolved offload directory, created only once we know where it lands.

    Containment is checked BEFORE anything is created. ``.carbon`` may itself be a
    symlink the workspace's owner planted, and by the time an after-the-fact check can
    refuse, ``mkdir(parents=True)`` has already made a directory outside the workspace —
    the data was safe, the side effect was not. A symlinked ``.carbon`` or ``offload``
    is refused outright, pointing in or out: the harness's own scratch directory is a
    real directory here or it is nothing. ``resolve()`` on a path that does not exist
    yet still follows the links in the part that does, which is exactly the question.
    """
    root = Path(workspace_root).resolve()
    carbon = Path(workspace_root) / OFFLOAD_SUBDIR.parts[0]
    offload = Path(workspace_root) / OFFLOAD_SUBDIR
    if carbon.is_symlink() or offload.is_symlink():
        raise _OffloadUnavailable("offload directory is a symlink")
    landed = offload.resolve()
    if landed != root and root not in landed.parents:
        raise _OffloadUnavailable("offload directory escapes the workspace")
    offload.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):  # housekeeping must never cost us the copy
        _mark_ignored(carbon)
    return landed


def _spill(text: str, workspace_root: Path | None) -> Path:
    """Write the complete text under the workspace; return its workspace-relative path.

    The filename is a content hash — deterministic per call, so a retried or repeated
    identical result re-lands on the same file instead of littering the workspace. The
    encoding is explicit here and on ``read_file``'s side, so the round trip does not
    depend on the host's locale.
    """
    if workspace_root is None:
        raise _OffloadUnavailable("no workspace to write under")
    landed = _offload_dir(Path(workspace_root))
    payload = text.encode("utf-8")
    target = landed / f"{hashlib.sha256(payload).hexdigest()[:16]}.txt"
    if not _holds(target, payload):
        _write_atomically(target, payload)
    # Written now or found already correct, the footer about to be returned names this
    # file: from here on pruning must leave it alone.
    _OURS.add(target)
    with suppress(OSError):  # housekeeping must never cost us the copy we just wrote
        _prune(landed)
    return OFFLOAD_SUBDIR / target.name


def _note(marker: str, note: str) -> str:
    """Add a clause to a truncation marker (which always ends in ``]``)."""
    return f"{marker[:-1]} {note}]"


def _why(exc: Exception) -> str:
    """A short cause for the marker. An OSError's ``str()`` carries the absolute path
    it failed on; model-visible text — and anything that records the transcript —
    must not, so the plain ``strerror`` is preferred where there is one."""
    return f"offload unavailable: {getattr(exc, 'strerror', None) or exc}"


def _offload_to_file(
    text: _Text, content_budget: int, tail_fraction: float, marker: str, workspace_root: Path | None
) -> str:
    """Recoverable truncation: write the COMPLETE text to a workspace file, keep the
    ``head_tail`` excerpt inline, and append a footer naming the file.

    The footer's path is RELATIVE to the workspace root: that is the path the model's
    own ``read_file`` resolves, and an absolute host path in a transcript would leak
    machine-private detail into anything that records it.

    Best-effort by construction. This runs mid-turn, after the assistant's tool_calls
    are already in the transcript, so raising would abandon the turn with an unanswered
    tool call and an unsaved session — strictly worse than the inline excerpt every
    other strategy would have produced. A failed write degrades to exactly that, and
    the marker says so instead of pretending a file exists.
    """
    try:
        # The tool's bytes, not the model's copy of them: this file is re-read, diffed,
        # applied and checksummed, and a relabeled line inside it is silent corruption.
        rel = _spill(text.complete, workspace_root)
    except (_OffloadUnavailable, OSError, UnicodeError) as exc:
        note = _note(marker, _why(exc))
        return _cut_head_tail(text.shown, content_budget, tail_fraction, note)
    head_chars, tail_chars = _split_budget(content_budget, tail_fraction)
    excerpt = _cut_head_tail(text.shown, content_budget, tail_fraction, marker)
    # The shown-count is the size actually produced (defanging happened before the cut,
    # so it cannot push the excerpt past the budget); every other count describes the
    # file, which is what the model is being sent to read.
    complete = text.complete
    return excerpt + _footer(
        rel,
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
    apply: Callable[[_Text, int, float, str, Path | None], str]
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
    workspace_root: str | Path | None = None,
) -> str:
    """Apply one vetted truncation strategy to text the harness or the user chose.

    An ``@path`` block, a generated checkpoint: content nobody is attacking us with, and
    content a later ``apply_patch``/``edit_file`` may have to match character for
    character. So it is cut and never rewritten. Tool output goes through
    ``truncate_tool_result`` instead.
    """
    return _door(
        _Text(text, text),
        policy,
        budget=budget,
        continuation_hint=continuation_hint,
        workspace_root=workspace_root,
    )


def truncate_tool_result(
    text: str,
    policy: TruncationPolicy,
    *,
    budget: int | None = None,
    continuation_hint: str | None = None,
    workspace_root: str | Path | None = None,
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
    """
    return _door(
        _Text(_defang(text), text),
        policy,
        budget=budget,
        continuation_hint=continuation_hint,
        workspace_root=workspace_root,
    )


def _door(
    text: _Text,
    policy: TruncationPolicy,
    *,
    budget: int | None = None,
    continuation_hint: str | None = None,
    workspace_root: str | Path | None = None,
) -> str:
    """The door itself, shared by both trust domains.

    ``budget`` lets a tool declare a smaller result limit without changing the
    selected strategy. The marker is deliberately actionable: losing bytes is
    unavoidable, losing the fact that bytes were lost is not — and under
    ``offload_to_file`` the bytes are not even lost, only moved to a file under
    ``workspace_root`` that the marker's footer names. The inline strategies ignore
    ``workspace_root``; offload degrades to an inline excerpt without one rather than
    guessing a directory (the process cwd is not the agent's workspace).

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
    root = Path(workspace_root) if workspace_root is not None else None
    return strat.apply(text, max_chars, policy.tail_fraction, marker, root)
