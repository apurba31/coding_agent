"""Per-execution token callback context for graph-backed agent requests."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_TOKEN_CALLBACK: ContextVar[Callable[[str], None] | None] = ContextVar(
    "coding_agent_token_callback",
    default=None,
)


@contextmanager
def use_token_callback(
    callback: Callable[[str], None] | None,
) -> Iterator[None]:
    """Make a token callback available to graph nodes for one invocation."""
    token: Token[Callable[[str], None] | None] = _TOKEN_CALLBACK.set(callback)
    try:
        yield
    finally:
        _TOKEN_CALLBACK.reset(token)


def get_token_callback() -> Callable[[str], None] | None:
    """Return the callback associated with the current graph invocation."""
    return _TOKEN_CALLBACK.get()
