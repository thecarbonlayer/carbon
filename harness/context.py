"""Context delivery (ch-04).

The model can't read files; the harness does. ``deliver`` scans user text for
``@path`` references, reads those files, and returns them as context blocks the
agent injects into the prompt — turning "look at @notes.txt" into the file's
actual contents in the window.

Each block is clamped (ch-06 door control): a single huge file can't be allowed
to flood the window, so it is truncated at the door before it ever enters the
prompt. The harness, not the model, opens the file — and decides how much fits.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness.harness_config import CONFIG
from harness.limits import truncate

# the pattern travels as a string in the editable surface; compiled here, at the use site
_ATTACH = re.compile(CONFIG.attach_pattern)


def deliver(user_text: str) -> list[str]:
    """Return a context block for each readable ``@path`` referenced in the text.

    Each block is clamped so a huge file can't flood the window (door control).
    """
    blocks: list[str] = []
    for match in _ATTACH.finditer(user_text):
        path = Path(match.group(1))
        if path.is_file():
            try:
                body = path.read_text()
            except (OSError, UnicodeDecodeError):
                # Unreadable or binary (`@image.png`) — skip it, don't crash the turn.
                continue
            blocks.append(
                truncate(
                    f"--- {path} ---\n{body}",
                    CONFIG.file_injection,
                    continuation_hint="Reference a smaller file or use read_file ranges.",
                )
            )
    return blocks
