"""The OpenAI-compatible HTTP call.

One implementation of the model seam: POST to ``/chat/completions`` on any
OpenAI-compatible endpoint (LM Studio, OpenRouter, Ollama, OpenAI, ...). This is
the body that used to live inside ``chat()``; ``model/client.py`` now dispatches
here when a provider has no ``responder``.

Going fully provider-agnostic (an Anthropic-native call, Gemini, ...) is "drop a
file next to this one" — the harness never changes because it depends only on the
``chat`` seam, not on this function.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import httpx

from model.provider import LLMResponse, OnDelta, Provider


def complete_openai(
    provider: Provider,
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = 1024,
    timeout: float = 180.0,
    on_delta: OnDelta | None = None,
    response_format: dict | None = None,
) -> LLMResponse:
    """One call to an OpenAI-compatible model, through a provider.

    ``temperature=None`` sends no temperature field at all, so the provider's own
    default applies — the same behavior as a harness that never pins the knob.

    ``max_tokens`` is generous on purpose: Gemma is a reasoning model and spends
    tokens thinking (``reasoning_content``) before it produces visible content.

    When ``on_delta`` is given, the call streams: tokens are handed to the callback
    as they arrive and the same ``LLMResponse`` is assembled at the end. When it is
    ``None`` this is the original blocking POST, byte-for-byte unchanged.

    ``response_format``, when given, is forwarded to the endpoint verbatim to
    constrain the model to structured output.
    """
    base_url = provider.base_url.rstrip("/")
    model = model or provider.model
    api_key = provider.api_key

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format
    if provider.reasoning_effort:
        payload["reasoning"] = {"effort": provider.reasoning_effort}

    if on_delta is not None:
        return _stream_openai(base_url, api_key, payload, timeout, on_delta)

    resp = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible endpoints (notably OpenRouter) can return HTTP 200 with an
    # ``{"error": ...}`` body, or an empty ``choices`` on a content filter. Surface
    # the provider's own message instead of a bare KeyError/IndexError traceback.
    choices = data.get("choices")
    if not choices:
        detail = data.get("error") or data
        raise RuntimeError(f"model returned no choices: {detail}")

    choice = choices[0]
    msg = choice.get("message") or {}  # ``message: null`` on some filter finishes
    return LLMResponse(
        content=msg.get("content") or "",
        reasoning=msg.get("reasoning_content"),
        tool_calls=msg.get("tool_calls") or [],
        usage=data.get("usage") or {},  # ``usage: null`` must not become None
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )


def _iter_sse_chunks(response: httpx.Response) -> Iterable[dict]:
    """Yield the parsed JSON object from each ``data:`` line of an SSE stream.

    Skips keep-alive/blank lines and stops at the ``[DONE]`` sentinel. A line that
    isn't valid JSON is skipped rather than crashing the stream (the same
    forgiving posture as the JSON-L session reader)."""
    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _stream_openai(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: float,
    on_delta: OnDelta,
) -> LLMResponse:
    """Stream a chat completion, feeding tokens to ``on_delta`` as they land."""
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        return assemble_stream(_iter_sse_chunks(resp), on_delta)


def assemble_stream(chunks: Iterable[dict], on_delta: OnDelta | None) -> LLMResponse:
    """Fold OpenAI-compatible stream chunks into one ``LLMResponse``.

    Pure over an iterable of already-parsed chunk dicts (no HTTP), so the assembly
    logic is testable offline. Content and reasoning pieces are streamed to
    ``on_delta`` as they arrive; tool-call fragments are merged by ``index`` (the
    first fragment carries ``id``/``name``, later ones append ``arguments``); usage
    and finish_reason are taken from the trailing chunks.
    """
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage: dict = {}
    finish_reason: str | None = None

    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
                if on_delta is not None:
                    on_delta("content", delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
                if on_delta is not None:
                    on_delta("reasoning", delta["reasoning_content"])
            for frag in delta.get("tool_calls") or []:
                _merge_tool_call(tool_calls, frag)

    ordered = [tool_calls[i] for i in sorted(tool_calls)]
    return LLMResponse(
        content="".join(content),
        reasoning="".join(reasoning) or None,
        tool_calls=ordered,
        usage=usage,
        finish_reason=finish_reason,
        raw={},
    )


def _merge_tool_call(acc: dict[int, dict], frag: dict) -> None:
    """Merge one streamed tool-call fragment into the accumulator, keyed by index."""
    idx = frag.get("index", 0)
    call = acc.setdefault(
        idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
    )
    if frag.get("id"):
        call["id"] = frag["id"]
    if frag.get("type"):
        call["type"] = frag["type"]
    fn = frag.get("function") or {}
    if fn.get("name"):
        call["function"]["name"] = fn["name"]
    if fn.get("arguments"):
        call["function"]["arguments"] += fn["arguments"]
