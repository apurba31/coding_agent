"""Tests for prompt construction and the agent orchestration loop."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.memory.store import ConversationManager, ConversationStore
from coding_agent.search.models import SearchResult
from coding_agent.tools.models import ToolParameter
from coding_agent.tools.registry import ToolRegistry


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error:
            raise self.error
        return self.responses.pop(0)


def _executor(llm, registry=None, max_steps=4):
    return AgentExecutor(
        llm_client=llm,
        tool_registry=registry or ToolRegistry(),
        prompt_builder=PromptBuilder("You are a coding assistant."),
        max_steps=max_steps,
    )


def test_tool_schema_uses_tool_parameter_fields_and_json_schema_types():
    registry = ToolRegistry()
    registry.register(
        name="inspect",
        func=lambda path, limit=10: None,
        description="Inspect a path.",
        parameters=[
            ToolParameter("path", "Path to inspect", "string"),
            ToolParameter("limit", "Maximum results", "integer", required=False, default=10),
            ToolParameter("options", "Extra options", "dict", required=False),
        ],
    )

    schema = PromptBuilder("system").get_tool_schemas(registry)[0]
    properties = schema["function"]["parameters"]["properties"]

    assert schema["function"]["parameters"]["required"] == ["path"]
    assert properties["path"]["type"] == "string"
    assert properties["limit"] == {
        "type": "integer",
        "description": "Maximum results",
        "default": 10,
    }
    assert properties["options"]["type"] == "object"


def test_agent_succeeds_when_model_returns_final_answer():
    llm = FakeLLM([_response(content="Done.")])

    state = _executor(llm).execute("Do the task")

    assert state["status"] == "SUCCESS"
    assert state["step_count"] == 1
    assert state["messages"][-1] == {"role": "assistant", "content": "Done."}
    assert llm.calls[0]["tools"] is None


def test_agent_executes_tool_then_returns_final_answer():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo", arguments='{"text": "hello"}'),
    )
    registry = ToolRegistry()
    registry.register("echo", lambda text: text.upper(), "Uppercase text")
    llm = FakeLLM([
        _response(tool_calls=[tool_call]),
        _response(content="The tool returned HELLO."),
    ])

    state = _executor(llm, registry).execute("Echo hello")

    assert state["status"] == "SUCCESS"
    assert state["step_count"] == 2
    assert state["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert state["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "echo",
        "content": "HELLO",
    }
    assert llm.calls[0]["tools"][0]["function"]["name"] == "echo"


def test_agent_marks_llm_errors_as_failed():
    llm = FakeLLM(error=RuntimeError("service unavailable"))

    state = _executor(llm).execute("Do the task")

    assert state["status"] == "FAILED"
    assert "service unavailable" in state["messages"][-1]["content"]


def test_agent_stops_at_maximum_steps():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="continue", arguments="{}"),
    )
    registry = ToolRegistry()
    registry.register("continue", lambda: "still working", "Keep working")
    llm = FakeLLM([_response(tool_calls=[tool_call])])

    state = _executor(llm, registry, max_steps=1).execute("Do the task")

    assert state["status"] == "FAILED"
    assert state["step_count"] == 1
    assert len(llm.calls) == 1


def test_agent_retrieves_context_and_resumes_persisted_conversation(tmp_path):
    chunk = Chunk(
        chunk_id="chunk-1",
        path=Path("src/example.py"),
        language="Python",
        start_line=1,
        end_line=2,
        symbol="example",
        symbol_kind=ChunkKind.FUNCTION,
        code="def example():\n    return 42",
    )

    class FakeRetriever:
        def search(self, query, top_k):
            assert query in {"Find example", "Explain it"}
            assert top_k == 3
            return [SearchResult(chunk=chunk, score=0.9, chunk_id=chunk.chunk_id)]

    first_llm = FakeLLM([_response(content="Found the example.")])
    first_manager = ConversationManager(ConversationStore(tmp_path))
    first_executor = AgentExecutor(
        first_llm,
        ToolRegistry(),
        PromptBuilder("You are a coding assistant."),
        conversation_manager=first_manager,
        retriever=FakeRetriever(),
        context_top_k=3,
    )
    first_executor.execute("Find example", conversation_id="thread-1")

    second_llm = FakeLLM([_response(content="It returns 42.")])
    second_manager = ConversationManager(ConversationStore(tmp_path))
    second_executor = AgentExecutor(
        second_llm,
        ToolRegistry(),
        PromptBuilder("You are a coding assistant."),
        conversation_manager=second_manager,
        retriever=FakeRetriever(),
        context_top_k=3,
    )
    state = second_executor.execute("Explain it", conversation_id="thread-1")

    messages = second_llm.calls[0]["messages"]
    assert state["status"] == "SUCCESS"
    assert "def example():" in messages[0]["content"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "Find example"
    assert messages[2]["content"] == "Found the example."
    assert messages[3]["content"] == "Explain it"
    assert (tmp_path / "thread-1.json").exists()
