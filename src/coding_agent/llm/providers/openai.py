"""OpenAI-compatible provider adapter for the coding agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from coding_agent.llm.base import LLMClient
from coding_agent.llm.streaming import collect_chat_stream


class OpenAICompatibleClient(LLMClient):
    """Thin adapter over an OpenAI-like API transport."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.transport = transport

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.transport is None:
            raise RuntimeError("No OpenAI-compatible transport configured.")

        model = kwargs.pop("model", self.default_model)
        payload = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools
            payload.setdefault("tool_choice", "auto")

        transport = self.transport
        if hasattr(transport, "chat") and hasattr(transport.chat, "completions"):
            return transport.chat.completions.create(**payload)
        if callable(getattr(transport, "create", None)):
            return transport.create(**payload)
        raise RuntimeError("Transport does not expose an OpenAI-compatible create() API.")

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.complete(messages=messages, tools=tools, **kwargs)

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream provider deltas and return a completion-like aggregate."""
        chunks = self.complete(messages=messages, tools=tools, stream=True, **kwargs)
        return collect_chat_stream(chunks, on_token)
