"""LangGraph wrapper for the plan, retrieve, tool, and answer control flow."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from coding_agent.planner import Planner, SimplePlanner, TaskPlan

from .executor import AgentExecutor
from .state import AgentState


class WorkflowState(TypedDict):
    """Data passed between graph nodes; the conversation remains in the executor."""

    goal: str
    conversation_id: NotRequired[str | None]
    additional_context: NotRequired[str | None]
    plan: NotRequired[TaskPlan]
    context: NotRequired[str | None]
    result: NotRequired[AgentState]


class LangGraphAgent:
    """Run the existing agent through an explicit LangGraph control-flow graph.

    LangGraph owns plan routing and the optional retrieval step. The executor
    continues to own the bounded LLM/tool loop and conversation persistence.
    """

    def __init__(self, executor: AgentExecutor, planner: Planner | None = None) -> None:
        self.executor = executor
        self.planner = planner or executor.planner or SimplePlanner()
        self.graph = self._build_graph()

    @property
    def conversation_manager(self):
        """Expose the underlying conversation manager to CLI lifecycle code."""
        return self.executor.conversation_manager

    def _build_graph(self):
        workflow = StateGraph(WorkflowState)
        workflow.add_node("analyze_question", self._analyze_question)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("answer_direct", self._answer_direct)
        workflow.add_node("answer_with_retrieval", self._answer_with_retrieval)
        workflow.add_node("answer_with_tools", self._answer_with_tools)
        workflow.add_edge(START, "analyze_question")
        workflow.add_conditional_edges(
            "analyze_question",
            self._after_analysis,
            {
                "direct": "answer_direct",
                "retrieve": "retrieve_context",
            },
        )
        workflow.add_conditional_edges(
            "retrieve_context",
            self._after_retrieval,
            {
                "answer": "answer_with_retrieval",
                "tools": "answer_with_tools",
            },
        )
        workflow.add_edge("answer_direct", END)
        workflow.add_edge("answer_with_retrieval", END)
        workflow.add_edge("answer_with_tools", END)
        return workflow.compile()

    def _analyze_question(self, state: WorkflowState) -> dict[str, TaskPlan]:
        return {"plan": self.planner.plan(state["goal"])}

    @staticmethod
    def _after_analysis(state: WorkflowState) -> str:
        plan = state["plan"]
        return "retrieve" if plan.needs_repository_context else "direct"

    def _retrieve_context(self, state: WorkflowState) -> dict[str, str | None]:
        context = self.executor.retrieve_context(state["goal"])
        additional_context = state.get("additional_context")
        if additional_context:
            context = "\n\n".join(item for item in (context, additional_context) if item)
        return {"context": context}

    @staticmethod
    def _after_retrieval(state: WorkflowState) -> str:
        return "tools" if state["plan"].allow_tools else "answer"

    def _answer_direct(self, state: WorkflowState) -> dict[str, AgentState]:
        return {"result": self._execute(state, tools_allowed=False)}

    def _answer_with_retrieval(self, state: WorkflowState) -> dict[str, AgentState]:
        return {"result": self._execute(state, tools_allowed=False)}

    def _answer_with_tools(self, state: WorkflowState) -> dict[str, AgentState]:
        return {"result": self._execute(state, tools_allowed=True)}

    def _execute(self, state: WorkflowState, *, tools_allowed: bool) -> AgentState:
        return self.executor.execute(
            goal=state["goal"],
            conversation_id=state.get("conversation_id"),
            additional_context=state.get("context") or state.get("additional_context"),
            plan_override=state["plan"],
            retrieve_context=False,
            tools_allowed_override=tools_allowed,
        )

    def execute(
        self,
        goal: str,
        conversation_id: str | None = None,
        additional_context: str | None = None,
    ) -> AgentState:
        """Run one request through the compiled routing graph."""
        initial_state: WorkflowState = {
            "goal": goal,
            "conversation_id": conversation_id,
            "additional_context": additional_context,
        }
        final_state = self.graph.invoke(initial_state)
        result = final_state.get("result")
        if result is None:
            raise RuntimeError("Workflow ended without an agent result")
        return result
