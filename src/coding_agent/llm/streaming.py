"""Helpers for assembling OpenAI-compatible chat completion streams."""

from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any


def collect_chat_stream(
    chunks: Iterable[Any],
    on_token: Callable[[str], None] | None = None,
) -> Any:
    """Accumulate streamed text/tool-call deltas into a completion-like response."""
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = None

    for chunk in chunks:
        usage = getattr(chunk, "usage", None) or usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            content_parts.append(content)
            if on_token:
                on_token(content)

        for tool_delta in getattr(delta, "tool_calls", None) or []:
            index = tool_delta.index
            call = tool_calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "name": "",
                    "arguments": "",
                },
            )
            if tool_delta.id:
                call["id"] = tool_delta.id
            if tool_delta.type:
                call["type"] = tool_delta.type
            function_delta = getattr(tool_delta, "function", None)
            if function_delta:
                call["name"] += getattr(function_delta, "name", None) or ""
                call["arguments"] += getattr(function_delta, "arguments", None) or ""

    assembled_calls = [
        SimpleNamespace(
            id=call["id"] or f"streamed-call-{index}",
            type=call["type"],
            function=SimpleNamespace(name=call["name"], arguments=call["arguments"]),
        )
        for index, call in sorted(tool_calls.items())
    ]
    message = SimpleNamespace(
        content="".join(content_parts) or None,
        tool_calls=assembled_calls or None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )
