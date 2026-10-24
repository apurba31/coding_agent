"""Tests for the LLM provider abstraction layer."""

from types import SimpleNamespace

from coding_agent.llm.base import LLMClient
from coding_agent.llm.client import GroqClient
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


def test_groq_client_preserves_legacy_positional_generation_arguments(monkeypatch):
    calls = []

    class FakeGroqTransport:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create),
            )

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[])

    monkeypatch.setattr("coding_agent.llm.client.Groq", FakeGroqTransport)
    client = GroqClient(api_key="test-key")
    tools = [{"type": "function"}]

    client.generate_response(
        [{"role": "user", "content": "hello"}],
        "test-model",
        tools,
        "required",
        0.25,
    )

    assert calls[0]["model"] == "test-model"
    assert calls[0]["tools"] == tools
    assert calls[0]["tool_choice"] == "required"
    assert calls[0]["temperature"] == 0.25


def test_groq_stream_response_emits_tokens_and_aggregates_text(monkeypatch):
    calls = []

    class FakeGroqTransport:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create),
            )

        def create(self, **kwargs):
            calls.append(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="hello", tool_calls=None))
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=" world", tool_calls=None)
                            )
                        ],
                        usage=None,
                    ),
                ]
            )

    monkeypatch.setattr("coding_agent.llm.client.Groq", FakeGroqTransport)
    client = GroqClient(api_key="test-key")
    tokens = []

    response = client.stream_response(
        [{"role": "user", "content": "hello"}],
        on_token=tokens.append,
    )

    assert calls[0]["stream"] is True
    assert tokens == ["hello", " world"]
    assert response.choices[0].message.content == "hello world"


def test_openai_compatible_stream_response_forwards_deltas():
    class StreamingTransport(DummyOpenAIClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="chunk", tool_calls=None)
                            )
                        ],
                        usage=None,
                    )
                ]
            )

    transport = StreamingTransport()
    client = OpenAICompatibleClient(
        api_key="test-key",
        transport=transport,
    )
    tokens = []

    response = client.stream_response(
        [{"role": "user", "content": "hello"}],
        on_token=tokens.append,
    )

    assert transport.calls[0]["stream"] is True
    assert tokens == ["chunk"]
    assert response.choices[0].message.content == "chunk"
