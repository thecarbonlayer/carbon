"""The provider configuration object — the model seam, made explicit.

Every model call in the whole course goes through ``chat()`` (see
``model/client.py``). The agent never talks to a provider directly, so swapping
providers (Ch14) is just changing the ``LLM_BASE_URL`` / ``LLM_MODEL`` env vars —
nothing above this file changes.

Targets any OpenAI-compatible ``/chat/completions`` endpoint:
  - local dev / acceptance: LM Studio (defaults below)
  - CI / hosted: OpenRouter, OpenAI, Ollama, ...

A ``Provider`` is pure config (base_url / model / api_key). It can also carry an
optional ``responder`` — a callable that produces an ``LLMResponse`` directly —
so a fake provider can inject deterministic, offline responses without any HTTP.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"
DEFAULT_API_KEY = "lm-studio"

# A streaming sink: called per token with ``(channel, text)`` where channel is
# "content" (the visible answer) or "reasoning" (the model's thinking). Optional
# everywhere — pass ``None`` and the seam is the same blocking call as before.
OnDelta = Callable[[str, str], None]


def _unquote(value: str) -> str:
    """Strip an inline ``# comment`` and matching surrounding quotes from a value.

    Mirrors what python-dotenv does, so a key copied verbatim from a provider's
    docs (``LLM_API_KEY="sk-..."``) doesn't ship its quotes into the auth header.
    A ``#`` is only a comment when it follows whitespace, so URLs with fragments
    survive; inside quotes nothing is stripped.
    """
    value = value.strip()
    if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
        return value[1:-1]
    # Unquoted: drop a trailing ` # comment` (whitespace-preceded hash only).
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1].isspace():
            return value[:i].strip()
    return value


def load_env(root: str | Path = ".") -> None:
    """Minimal .env loader (no dependency). Real env vars always win.

    ``root`` is the directory holding ``.env``. It defaults to the process cwd, but
    a consumer running from elsewhere (an eval suite, a tuner) passes the carbon
    checkout root so it reads the same file the harness's own gates read, instead
    of reimplementing dotenv (the embedding seam, see dev-notes/adr/0002).
    """
    path = Path(root) / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):  # honor shell-style `export KEY=value`
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), _unquote(value))


# Back-compat alias for the pre-v0.1 private name.
_load_dotenv = load_env


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None = None
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class Provider:
    """The model seam, made explicit (ch-14). Any OpenAI-compatible endpoint.

    Swap the provider and nothing above this object changes.

    ``responder`` is an optional escape hatch: when set, ``chat()`` calls it
    instead of hitting the network. This is how the ``fake`` provider injects
    deterministic responses for offline tests. ``from_env`` always leaves it
    ``None`` — env-configured providers go over HTTP.
    """

    base_url: str
    model: str
    api_key: str = "x"
    responder: Callable[..., LLMResponse] | None = None
    # OpenRouter's unified reasoning-effort control (forwarded verbatim as
    # ``{"reasoning": {"effort": ...}}`` — see model/openai_compatible.py). Provider-
    # level, not a per-call chat() kwarg, because it's a property of WHICH model is
    # being measured (like base_url/model), not of an individual request. None means
    # "don't send it" — most models/providers neither support nor need it, and a
    # provider that doesn't recognize the field would otherwise get a silently
    # ignored (or rejected) parameter on every call.
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, root: str | Path = ".") -> Provider:
        load_env(root)
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("LLM_API_KEY", DEFAULT_API_KEY),
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or None,
        )


# Presets — same agent, one line to switch.
def lmstudio(model: str = DEFAULT_MODEL) -> Provider:
    return Provider(DEFAULT_BASE_URL, model, "lm-studio")


def openrouter(model: str, api_key: str) -> Provider:
    return Provider("https://openrouter.ai/api/v1", model, api_key)


def ollama(model: str) -> Provider:
    return Provider("http://localhost:11434/v1", model, "ollama")


def openai(model: str, api_key: str) -> Provider:
    return Provider("https://api.openai.com/v1", model, api_key)
