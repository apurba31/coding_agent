"""Abstract LLM client interface and shared request/response helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
