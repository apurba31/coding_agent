"""Tests for deterministic planner decisions and executor route enforcement."""
from types import SimpleNamespace

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.planner import PlanRoute, SimplePlanner
from coding_agent.tools.models import ToolParameter
from coding_agent.tools.registry import ToolRegistry


def test_simple_planner_routes_general_questions_directly():
    plan = SimplePlanner().plan("Explain recursion with a simple example")

    assert plan.route == PlanRoute.DIRECT
    assert not plan.needs_repository_context
    assert not plan.allow_tools


def test_simple_planner_retrieves_for_repository_context_without_tools():
    plan = SimplePlanner().plan("Explain what the UserService class does")

    assert plan.route == PlanRoute.RETRIEVE
    assert plan.needs_repository_context
    assert not plan.allow_tools


def test_simple_planner_enables_tools_for_explicit_code_lookup():
    plan = SimplePlanner().plan("Find the definition of UserService")

    assert plan.route == PlanRoute.TOOL
    assert plan.needs_repository_context
    assert plan.allow_tools


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


class CountingRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k):
        self.queries.append((query, top_k))
        return []


def test_executor_skips_retrieval_and_tool_schemas_for_direct_plan():
    llm = FakeLLM([_response(content="Recursion calls itself." )])
    retriever = CountingRetriever()
    registry = ToolRegistry()
    registry.register("inspect", lambda: "inspected", "Inspect repository")
    executor = AgentExecutor(
        llm,
        registry,
        PromptBuilder("system"),
        retriever=retriever,
        planner=SimplePlanner(),
    )

    state = executor.execute("Explain recursion")

    assert state["status"] == "SUCCESS"
    assert state["plan"].route == PlanRoute.DIRECT
    assert retriever.queries == []
    assert llm.calls[0]["tools"] is None


def test_executor_retrieves_and_enables_tools_for_explicit_lookup():
    llm = FakeLLM([_response(content="The definition is available in context.")])
    retriever = CountingRetriever()
    registry = ToolRegistry()
    registry.register(
        "find_definition",
        lambda symbol: symbol,
        "Find a definition",
        [ToolParameter("symbol", "Symbol name", "string")],
    )
    executor = AgentExecutor(
        llm,
        registry,
        PromptBuilder("system"),
        retriever=retriever,
        planner=SimplePlanner(),
    )

    state = executor.execute("Find the definition of UserService")

    assert state["status"] == "SUCCESS"
    assert state["plan"].route == PlanRoute.TOOL
    assert retriever.queries == [("Find the definition of UserService", 5)]
    assert llm.calls[0]["tools"][0]["function"]["name"] == "find_definition"


def test_executor_blocks_tool_call_that_planner_did_not_allow():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="inspect", arguments="{}"),
    )
    invoked = []
    registry = ToolRegistry()
    registry.register("inspect", lambda: invoked.append(True), "Inspect repository")
    llm = FakeLLM(
        [
            _response(tool_calls=[tool_call]),
            _response(content="Answer without tool access."),
        ]
    )
    executor = AgentExecutor(
        llm,
        registry,
        PromptBuilder("system"),
        planner=SimplePlanner(),
    )

    state = executor.execute("Explain recursion")

    assert state["status"] == "SUCCESS"
    assert invoked == []
    assert "not enabled by the task plan" in state["messages"][3]["content"]
