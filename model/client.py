"""The model seam — the free ``chat()`` entry point.

Every model call in the whole course goes through this function. It defaults to
the env-configured provider and dispatches:
  - to the provider's ``responder`` when one is set (e.g. the fake provider), or
  - to the OpenAI-compatible HTTP call otherwise.

Keeping ``chat`` a free, module-level function is deliberate: the agent imports it
as a name (``from model import chat``) so tests can ``patch.object(mod, "chat")``.
"""

from __future__ import annotations

from model.openai_compatible import complete_openai
from model.provider import LLMResponse, OnDelta, Provider


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 180.0,
    provider: Provider | None = None,
    on_delta: OnDelta | None = None,
    response_format: dict | None = None,
) -> LLMResponse:
    """One call to the model, through a provider (defaults to env config).

    ``max_tokens`` is generous on purpose: Gemma is a reasoning model and spends
    tokens thinking (``reasoning_content``) before it produces visible content.

    ``on_delta``, when given, streams tokens as they arrive. A ``responder``
    provider (the fake) has no network to stream, so its final content is replayed
    through the callback once — the streaming path stays exercised offline.

    ``response_format``, when given, is forwarded to the endpoint verbatim (e.g.
    ``{"type": "json_schema", "json_schema": {...}}`` or ``{"type": "json_object"}``)
    so a caller can constrain the model to structured output. carbon forwards it and
    parses nothing; the schema is the caller's (the embedding seam, adr/0002).
    """
    provider = provider or Provider.from_env()
    if provider.responder is not None:
        resp = provider.responder(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        if on_delta is not None:
            if resp.reasoning:
                on_delta("reasoning", resp.reasoning)
            if resp.content:
                on_delta("content", resp.content)
        return resp
    return complete_openai(
        provider,
        messages,
        model=model,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        on_delta=on_delta,
        response_format=response_format,
    )
