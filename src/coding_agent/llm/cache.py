"""Bounded in-memory caching for deterministic, tool-free LLM requests."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
from typing import Any

from coding_agent.llm.base import LLMClient
from coding_agent.observability import MetricsCollector, get_metrics_collector

_MISSING = object()


class _CachedResponse:
    """Expose a cached provider response without reporting API token usage again."""

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def usage(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class CachedLLMClient(LLMClient):
    """Cache exact temperature-zero requests that do not expose tools."""

    def __init__(
        self,
        client: LLMClient,
        max_entries: int = 128,
        metrics: MetricsCollector | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.client = client
        self.max_entries = max_entries
        self.metrics = metrics or get_metrics_collector()
        self._responses: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        cache_key = self._eligible_cache_key(messages, tools, kwargs)
        if cache_key is None:
            self.metrics.increment("llm.cache.bypasses")
            return self.client.generate_response(messages=messages, tools=tools, **kwargs)

        cached_response = self._lookup(cache_key)
        if cached_response is not _MISSING:
            return _CachedResponse(cached_response)

        response = self.client.generate_response(messages=messages, tools=tools, **kwargs)
        self._store(cache_key, response)
        self.metrics.increment("llm.cache.misses")
        return response

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream cache misses and replay cached answer text as a single chunk."""
        cache_key = self._eligible_cache_key(messages, tools, kwargs)
        if cache_key is None:
            self.metrics.increment("llm.cache.bypasses")
            return self._stream_provider(messages, tools, on_token, kwargs)

        cached_response = self._lookup(cache_key)
        if cached_response is not _MISSING:
            self._emit_content(cached_response, on_token)
            return _CachedResponse(cached_response)

        response = self._stream_provider(messages, tools, on_token, kwargs)
        self._store(cache_key, response)
        self.metrics.increment("llm.cache.misses")
        return response

    def _eligible_cache_key(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> str | None:
        temperature = kwargs.get(
            "temperature",
            getattr(self.client, "default_temperature", None),
        )
        if tools or temperature != 0.0:
            return None
        return self._cache_key(messages, kwargs)

    def _lookup(self, cache_key: str) -> Any:
        with self._lock:
            if cache_key not in self._responses:
                return _MISSING
            self._responses.move_to_end(cache_key)
            self.metrics.increment("llm.cache.hits")
            return self._responses[cache_key]

    def _store(self, cache_key: str, response: Any) -> None:
        with self._lock:
            self._responses[cache_key] = response
            self._responses.move_to_end(cache_key)
            while len(self._responses) > self.max_entries:
                self._responses.popitem(last=False)

    def _stream_provider(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_token: Callable[[str], None] | None,
        kwargs: dict[str, Any],
    ) -> Any:
        streamer = getattr(self.client, "stream_response", None)
        if callable(streamer):
            return streamer(
                messages=messages,
                tools=tools,
                on_token=on_token,
                **kwargs,
            )
        response = self.client.generate_response(messages=messages, tools=tools, **kwargs)
        self._emit_content(response, on_token)
        return response

    @staticmethod
    def _emit_content(response: Any, on_token: Callable[[str], None] | None) -> None:
        if not on_token:
            return
        choices = getattr(response, "choices", None) or []
        if choices:
            content = getattr(choices[0].message, "content", None)
            if content:
                on_token(content)

    def _cache_key(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str | None:
        request = {
            "messages": messages,
            "kwargs": kwargs,
            "default_model": getattr(self.client, "default_model", None),
            "default_temperature": getattr(self.client, "default_temperature", None),
        }
        try:
            serialized = json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
