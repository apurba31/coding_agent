"""Tests for bounded deterministic LLM response caching."""

from types import SimpleNamespace

from coding_agent.llm.cache import CachedLLMClient
from coding_agent.observability.metrics import MetricsCollector


class FakeLLMClient:
    default_model = "test-model"
    default_temperature = 0.0

    def __init__(self):
        self.calls = []

    def generate_response(self, messages, tools=None, **kwargs):
        self.calls.append((messages, tools, kwargs))
        content = f"response-{len(self.calls)}"
        return SimpleNamespace(
            content=content,
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=10),
        )


def _messages(prompt):
    return [{"role": "user", "content": prompt}]


def test_cache_reuses_exact_deterministic_tool_free_requests():
    provider = FakeLLMClient()
    metrics = MetricsCollector()
    client = CachedLLMClient(provider, metrics=metrics)

    first = client.generate_response(_messages("explain recursion"))
    second = client.generate_response(_messages("explain recursion"))

    assert first.content == second.content == "response-1"
    assert first.usage.total_tokens == 10
    assert second.usage is None
    assert len(provider.calls) == 1
    assert metrics.snapshot().counters == {
        "llm.cache.misses": 1,
        "llm.cache.hits": 1,
    }


def test_cache_bypasses_tool_requests_and_nonzero_temperature():
    provider = FakeLLMClient()
    metrics = MetricsCollector()
    client = CachedLLMClient(provider, metrics=metrics)
    messages = _messages("inspect this")

    for _ in range(2):
        client.generate_response(messages, tools=[{"type": "function"}])
        client.generate_response(messages, temperature=0.4)

    assert len(provider.calls) == 4
    assert metrics.snapshot().counters == {"llm.cache.bypasses": 4}


def test_cache_is_bounded_and_evicts_least_recently_used_entries():
    provider = FakeLLMClient()
    metrics = MetricsCollector()
    client = CachedLLMClient(provider, max_entries=2, metrics=metrics)

    client.generate_response(_messages("A"))
    client.generate_response(_messages("B"))
    client.generate_response(_messages("A"))
    client.generate_response(_messages("C"))
    client.generate_response(_messages("B"))

    assert len(provider.calls) == 4
    assert metrics.snapshot().counters["llm.cache.hits"] == 1


def test_cache_bypasses_requests_that_cannot_be_canonicalized():
    provider = FakeLLMClient()
    metrics = MetricsCollector()
    client = CachedLLMClient(provider, metrics=metrics)
    messages = [{"role": "user", "content": object()}]

    client.generate_response(messages)
    client.generate_response(messages)

    assert len(provider.calls) == 2
    assert metrics.snapshot().counters["llm.cache.bypasses"] == 2


def test_cached_stream_emits_live_content_on_miss_and_replays_on_hit():
    provider = FakeLLMClient()
    client = CachedLLMClient(provider)
    messages = _messages("stream me")
    first_tokens = []
    second_tokens = []

    first = client.stream_response(messages, on_token=first_tokens.append)
    second = client.stream_response(messages, on_token=second_tokens.append)

    assert first.content == second.content == "response-1"
    assert first_tokens == second_tokens == ["response-1"]
    assert len(provider.calls) == 1
