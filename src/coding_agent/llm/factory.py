"""Construct configured language-model provider clients."""

import os
from typing import Literal

from openai import OpenAI

from coding_agent.llm.base import LLMClient
from coding_agent.llm.client import GroqClient
from coding_agent.llm.providers.openai import OpenAICompatibleClient

LLMProvider = Literal["groq", "nvidia"]
DEFAULT_GROQ_MODEL = "llama3-70b-8192"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def create_llm_client(provider: str = "groq", model: str | None = None) -> LLMClient:
    """Create a Groq or NVIDIA NIM client using credentials from the environment."""
    if provider == "groq":
        return GroqClient(default_model=model or DEFAULT_GROQ_MODEL)
    if provider != "nvidia":
        raise ValueError(f"Unsupported LLM provider {provider!r}; choose 'groq' or 'nvidia'.")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY must be set to use the NVIDIA provider.")

    model_name = model or os.environ.get("NVIDIA_MODEL")
    if not model_name:
        raise ValueError(
            "Set MINI_AGENT_LLM_MODEL or NVIDIA_MODEL to a model available in NVIDIA NIM."
        )

    base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
    transport = OpenAI(api_key=api_key, base_url=base_url)
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        default_model=model_name,
        transport=transport,
    )
