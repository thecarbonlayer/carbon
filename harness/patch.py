"""Pure parsing and hunk-application for multi-hunk, multi-file unified diffs
(ch-v4.1). No filesystem access here — see ``Workspace.apply_patch`` for the I/O
orchestration (read current content, compute the new content via this module,
commit through the same atomic ``Workspace.write`` every other tool uses).

``edit_file`` extends beyond raw text-replace with one load-bearing property: an
ambiguous edit is REJECTED, never guessed at, and the write is atomic. Bash alone
cannot give an agent that same guarantee for a real, multi-file change — a
``patch``/``git apply`` invocation run via the ``bash`` tool can partially apply
(some hunks land, one fails, the tree is left half-migrated) and typically fuzzes
a nearby match rather than refusing one that does not sit exactly where the diff
claims. This module extends that same philosophy — exact match at the claimed
location or refuse, all files computed together or none of them — to a real
unified diff covering several hunks across several files.

Explicitly out of scope: renames, copies, binary patches, and fuzzy/nearby
context matching. Each raises ``PatchError`` naming what was refused, rather
than approximating it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


class PatchError(Exception):
    """A patch could not be applied. Internal control flow within this module —
    ``Workspace.apply_patch`` catches it and returns an ``"error: ..."`` string,
    matching ``write``/``edit``'s own convention rather than raising to callers."""


@dataclass(frozen=True)
class Hunk:
    old_start: int  # 1-indexed; 0 for a brand-new file
    old_count: int
    body: tuple[str, ...]  # raw lines, each prefixed ' ', '-', '+', or '\\'


@dataclass(frozen=True)
class FileDiff:
    old_path: str | None  # None means this file is being newly created
    new_path: str | None  # None means this file is being deleted
    hunks: tuple[Hunk, ...]
    header_line: int  # 1-indexed line in the ORIGINAL patch text, for error messages

    @property
    def target(self) -> str:
        """The workspace-relative path this diff acts on, whichever side names it."""
        path = self.new_path if self.new_path is not None else self.old_path
        if path is None:  # pragma: no cover — _parse never produces both-None
            raise PatchError(f"line {self.header_line}: both sides are /dev/null")
        return path


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_prefix(path: str) -> str | None:
    """``a/foo.py`` / ``b/foo.py`` -> ``foo.py``; ``/dev/null`` -> ``None``."""
    if path == "/dev/null":
        return None
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def parse_patch(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff (``git diff`` or plain ``diff -u`` output) into one
    ``FileDiff`` per file, each carrying its hunks in the order they appeared.

    Delimited on ``'--- '`` lines rather than requiring a preceding
    ``diff --git`` line, so both diff sources parse the same way.
    """
    lines = diff_text.splitlines()
    files: list[FileDiff] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        header_line = i + 1
        old_path = _strip_prefix(lines[i][4:].split("\t")[0])
        i += 1
        if i >= len(lines) or not lines[i].startswith("+++ "):
            raise PatchError(f"line {header_line}: '---' with no following '+++' header")
        new_path = _strip_prefix(lines[i][4:].split("\t")[0])
        i += 1
        hunks: list[Hunk] = []
        while i < len(lines) and lines[i].startswith("@@ "):
            m = _HUNK_HEADER.match(lines[i])
            if not m:
                raise PatchError(f"line {i + 1}: malformed hunk header: {lines[i]!r}")
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            i += 1
            body: list[str] = []
            while i < len(lines) and lines[i][:1] in (" ", "-", "+", "\\"):
                body.append(lines[i])
                i += 1
            hunks.append(Hunk(old_start=old_start, old_count=old_count, body=tuple(body)))
        if not hunks:
            raise PatchError(f"line {header_line}: file header with no hunks ({old_path!r})")
        files.append(
            FileDiff(
                old_path=old_path, new_path=new_path, hunks=tuple(hunks), header_line=header_line
            )
        )
    return files


def _hunk_sides(hunk: Hunk) -> tuple[list[str], list[str]]:
    """(old-side lines, new-side lines) — context lines belong to both."""
    old: list[str] = []
    new: list[str] = []
    for raw in hunk.body:
        marker, text = raw[0], raw[1:]
        if marker == " ":
            old.append(text)
            new.append(text)
        elif marker == "-":
            old.append(text)
        elif marker == "+":
            new.append(text)
        # '\' ("No newline at end of file") carries no content of its own.
    return old, new


def _no_trailing_newline(hunk_body: tuple[str, ...]) -> bool:
    return bool(hunk_body) and hunk_body[-1].startswith("\\")


def _apply_hunks_to_lines(
    original_lines: list[str], hunks: tuple[Hunk, ...], path: str
) -> list[str]:
    """Original file split into lines -> the new content's lines.

    Hunks are consumed in ascending ``old_start`` order — real unified diffs
    already emit them that way, sorting here is a defensive, not a corrective,
    step. Each hunk's old-side must match the original file EXACTLY at the
    position the hunk claims; any mismatch refuses the whole patch rather than
    searching nearby for a plausible target.
    """
    cursor = 0  # 0-indexed position in original_lines already emitted
    out: list[str] = []
    for hunk in sorted(hunks, key=lambda h: h.old_start):
        old_side, new_side = _hunk_sides(hunk)
        start = hunk.old_start - 1  # 0-indexed
        if start < cursor:
            raise PatchError(f"{path}: hunk at line {hunk.old_start} overlaps a previous hunk")
        out.extend(original_lines[cursor:start])
        end = start + len(old_side)
        actual = original_lines[start:end]
        if actual != old_side:
            raise PatchError(
                f"{path}: hunk at line {hunk.old_start} does not match the file's current "
                f"content — expected {old_side!r}, found {actual!r}"
            )
        out.extend(new_side)
        cursor = end
    out.extend(original_lines[cursor:])
    return out


def plan_changes(
    files: list[FileDiff], read: Callable[[str], str], exists: Callable[[str], bool]
) -> list[tuple[str, str | None]]:
    """Validate every file in ``files`` against the CURRENT content ``read``/``exists``
    report, and compute each one's proposed new content (``None`` means delete),
    without writing anything. Raises ``PatchError`` on the first file that fails to
    validate — the caller (``Workspace.apply_patch``) only commits once this
    returns cleanly for every file, which is what makes the whole patch
    all-or-nothing.
    """
    planned: list[tuple[str, str | None]] = []
    for fd in files:
        if fd.old_path is None:
            # New file: must not already exist, and its one hunk's new-side IS
            # the whole file (old_start/old_count are 0 for a genuine addition).
            if exists(fd.target):
                raise PatchError(f"{fd.target}: patch creates a file that already exists")
            _, new_side = _hunk_sides(fd.hunks[0])
            content = "\n".join(new_side)
            if not _no_trailing_newline(fd.hunks[0].body):
                content += "\n"
            planned.append((fd.target, content))
        elif fd.new_path is None:
            # Deletion: current content must match the hunk's old side exactly.
            if not exists(fd.target):
                raise PatchError(f"{fd.target}: patch deletes a file that does not exist")
            current = read(fd.target)
            old_side, _ = _hunk_sides(fd.hunks[0])
            expected = "\n".join(old_side)
            if not _no_trailing_newline(fd.hunks[0].body):
                expected += "\n"
            if current != expected:
                raise PatchError(f"{fd.target}: current content does not match the deletion hunk")
            planned.append((fd.target, None))
        else:
            if not exists(fd.target):
                raise PatchError(f"{fd.target}: no such file")
            current = read(fd.target)
            had_trailing_nl = current.endswith("\n")
            original_lines = current[:-1].split("\n") if had_trailing_nl else current.split("\n")
            new_lines = _apply_hunks_to_lines(original_lines, fd.hunks, fd.target)
            keep_trailing_nl = had_trailing_nl and not _no_trailing_newline(fd.hunks[-1].body)
            content = "\n".join(new_lines) + ("\n" if keep_trailing_nl else "")
            planned.append((fd.target, content))
    return planned
