"""Tests for aggregating streamed chat completion deltas."""

from types import SimpleNamespace

from coding_agent.llm.streaming import collect_chat_stream


def _chunk(content=None, tool_calls=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)],
        usage=usage,
    )


def test_stream_aggregates_text_and_tool_call_deltas():
    emitted = []
    chunks = [
        _chunk(content="Checking "),
        _chunk(
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="call-1",
                    type="function",
                    function=SimpleNamespace(name="read_", arguments='{"path":'),
                )
            ]
        ),
        _chunk(
            content="the repository.",
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id=None,
                    type=None,
                    function=SimpleNamespace(name="file", arguments='"README.md"}'),
                )
            ],
        ),
    ]

    response = collect_chat_stream(chunks, emitted.append)
    message = response.choices[0].message

    assert "".join(emitted) == "Checking the repository."
    assert message.content == "Checking the repository."
    assert message.tool_calls[0].id == "call-1"
    assert message.tool_calls[0].function.name == "read_file"
    assert message.tool_calls[0].function.arguments == '{"path":"README.md"}'
