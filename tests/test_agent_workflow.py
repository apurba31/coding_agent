"""Tests for LangGraph plan/retrieve/tool routing."""
from types import SimpleNamespace

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.agent.workflow import LangGraphAgent
from coding_agent.planner import PlanRoute, SimplePlanner
from coding_agent.tools.models import ToolParameter
from coding_agent.tools.registry import ToolRegistry


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return []


def _graph(llm, retriever=None, registry=None):
    executor = AgentExecutor(
        llm_client=llm,
        tool_registry=registry or ToolRegistry(),
        prompt_builder=PromptBuilder("system prompt"),
        retriever=retriever,
        planner=SimplePlanner(),
    )
    return LangGraphAgent(executor)


def test_graph_direct_path_skips_retrieval_and_tools():
    llm = FakeLLM([_response(content="Recursion calls itself.")])
    retriever = FakeRetriever()
    graph = _graph(llm, retriever)

    result = graph.execute("Explain recursion simply")

    assert result["status"] == "SUCCESS"
    assert result["plan"].route == PlanRoute.DIRECT
    assert retriever.calls == []
    assert llm.calls[0]["tools"] is None


def test_graph_retrieval_path_retrieves_once_without_enabling_tools():
    llm = FakeLLM([_response(content="The class handles users.")])
    retriever = FakeRetriever()
    registry = ToolRegistry()
    registry.register("read_file", lambda path: path, "Read a file")
    graph = _graph(llm, retriever, registry)

    result = graph.execute("Explain what the UserService class does")

    assert result["status"] == "SUCCESS"
    assert result["plan"].route == PlanRoute.RETRIEVE
    assert retriever.calls == [("Explain what the UserService class does", 5)]
    assert llm.calls[0]["tools"] is None


def test_graph_tool_path_enables_tools_and_executes_tool_cycle():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="find_definition", arguments='{"symbol":"UserService"}'),
    )
    llm = FakeLLM(
        [
            _response(tool_calls=[call]),
            _response(content="UserService is defined in the repository."),
        ]
    )
    retriever = FakeRetriever()
    registry = ToolRegistry()
    registry.register(
        "find_definition",
        lambda symbol: f"Definition for {symbol}",
        "Find symbol definition",
        [ToolParameter("symbol", "Name", "string")],
    )
    graph = _graph(llm, retriever, registry)

    result = graph.execute("Find the definition of UserService")

    assert result["status"] == "SUCCESS"
    assert result["plan"].route == PlanRoute.TOOL
    assert len(retriever.calls) == 1
    assert llm.calls[0]["tools"][0]["function"]["name"] == "find_definition"
    assert result["messages"][-2]["content"] == "Definition for UserService"


def test_graph_passes_additional_context_without_duplicate_llm_calls():
    llm = FakeLLM([_response(content="Grounded answer.")])
    graph = _graph(llm)

    result = graph.execute(
        "Explain recursion",
        additional_context="Research findings: recursive call at line 3.",
    )

    assert result["status"] == "SUCCESS"
    assert len(llm.calls) == 1
    assert "recursive call at line 3" in llm.calls[0]["messages"][0]["content"]
