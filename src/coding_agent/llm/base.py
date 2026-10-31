"""Abstract LLM client interface and shared request/response helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class LLMClient(ABC):
    """Portable interface for model providers that support text and tool calls."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a provider response object for a chat completion request."""
        raise NotImplementedError

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compatibility shim for existing code expecting generate_response()."""
        return self.complete(messages=messages, tools=tools, **kwargs)

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream when supported, or deliver a complete response as one chunk."""
        response = self.generate_response(messages=messages, tools=tools, **kwargs)
        if on_token:
            content = response.choices[0].message.content
            if content:
                on_token(content)
        return response
