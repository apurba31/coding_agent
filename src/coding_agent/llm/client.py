"""Groq LLM client."""

import os
from typing import Any, Callable

from groq import Groq

from coding_agent.llm.base import LLMClient
from coding_agent.llm.streaming import collect_chat_stream


class GroqClient(LLMClient):
    """Client for interacting with the Groq API."""

    def __init__(self, api_key: str | None = None, default_model: str = "llama3-70b-8192"):
        """Initialize the Groq client.

        Args:
            api_key: The Groq API key. If not provided, it will look for GROQ_API_KEY env var.
            default_model: The default model to use for generations.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment variables.")

        self.client = Groq(api_key=self.api_key)
        self.default_model = default_model
        self.default_temperature = 0.0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Complete a chat request through Groq.

        Args:
            messages: List of message dictionaries.
            model: Model to use (defaults to self.default_model).
            tools: Optional list of tool schemas in OpenAI format.
            tool_choice: Tool choice strategy ("auto", "none", or specific tool).
            temperature: Sampling temperature.

        Returns:
            The raw response object from the Groq API.
        """
        try:
            return self.client.chat.completions.create(
                **self._request_kwargs(messages, tools, kwargs)
            )
        except Exception as e:
            # Wrap or re-raise
            raise RuntimeError(f"Error communicating with Groq API: {e}") from e

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream response text while collecting tool-call deltas for the executor."""
        request_kwargs = self._request_kwargs(messages, tools, kwargs)
        request_kwargs["stream"] = True
        try:
            chunks = self.client.chat.completions.create(**request_kwargs)
        except Exception as error:
            raise RuntimeError(f"Error communicating with Groq API: {error}") from error
        return collect_chat_stream(chunks, on_token)

    def _request_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        request_options = dict(options)
        model = request_options.pop("model", None)
        tool_choice = request_options.pop("tool_choice", "auto")
        temperature = request_options.pop("temperature", self.default_temperature)
        request_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = tool_choice
        request_kwargs.update(request_options)
        return request_kwargs

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Preserve the original positional API while supporting the base interface."""
        if len(args) > 4:
            raise TypeError("generate_response accepts at most four positional options")

        if args and isinstance(args[0], list):
            positional_names: tuple[str, ...] = ("tools", "tool_choice", "temperature")
        else:
            positional_names = ("model", "tools", "tool_choice", "temperature")
        if len(args) > len(positional_names):
            raise TypeError("Too many positional options for generate_response")
        for name, value in zip(positional_names, args, strict=False):
            kwargs.setdefault(name, value)

        return self.complete(messages=messages, **kwargs)
