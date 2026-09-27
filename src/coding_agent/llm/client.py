"""Groq LLM client."""

import os
from typing import Any

from groq import Groq


class GroqClient:
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

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
    ) -> Any:
        """Generate a response from the LLM.

        Args:
            messages: List of message dictionaries.
            model: Model to use (defaults to self.default_model).
            tools: Optional list of tool schemas in OpenAI format.
            tool_choice: Tool choice strategy ("auto", "none", or specific tool).
            temperature: Sampling temperature.

        Returns:
            The raw response object from the Groq API.
        """
        model_to_use = model or self.default_model

        kwargs: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            # Wrap or re-raise
            raise RuntimeError(f"Error communicating with Groq API: {e}") from e
