"""Tests for the LLM provider abstraction layer."""

from types import SimpleNamespace

from coding_agent.llm.base import LLMClient
from coding_agent.llm.providers.openai import OpenAICompatibleClient


class DummyOpenAIClient:
    def __init__(self):
        self.calls = []

    def chat(self):
        return self

    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello from provider."))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16),
        )


def test_llm_client_is_abstract_and_exposes_complete_api():
    assert hasattr(LLMClient, "complete")
    assert hasattr(LLMClient, "generate_response")


def test_openai_compatible_client_wraps_provider_and_tracks_usage():
    transport = DummyOpenAIClient()
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        default_model="gpt-4o-mini",
        transport=transport,
    )

    response = client.generate_response(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "noop"}}],
        temperature=0.2,
    )

    assert response.choices[0].message.content == "Hello from provider."
    assert response.usage.prompt_tokens == 11
    assert response.usage.total_tokens == 16
    assert transport.calls[0]["model"] == "gpt-4o-mini"
