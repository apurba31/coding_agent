"""LangGraph wrapper for the plan, retrieve, tool, and answer control flow."""

from __future__ import annotations

import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from coding_agent.planner import Planner, SimplePlanner, TaskPlan
from coding_agent.memory.store import ConversationManager

from .executor import AgentExecutor
from .state import AgentState
from .streaming import get_token_callback, use_token_callback


class WorkflowState(TypedDict):
    """Data passed between graph nodes; the conversation remains in the executor."""

    goal: str
    conversation_id: NotRequired[str | None]
    additional_context: NotRequired[str | None]
    plan: NotRequired[TaskPlan]
    context: NotRequired[str | None]
    agent_state: NotRequired[AgentState]
    tool_schemas: NotRequired[list[dict[str, Any]]]
    tools_allowed: NotRequired[bool]
    result: NotRequired[AgentState]


class LangGraphAgent:
    """Run planning, retrieval, model calls, and tool execution as graph nodes.

    The graph checkpointer stores each completed node in a local SQLite database.
    """

    def __init__(
        self,
        executor: AgentExecutor,
        planner: Planner | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.executor = executor
        self.planner = planner or executor.planner or SimplePlanner()
        if checkpoint_path is None:
            manager = executor.conversation_manager
            checkpoint_path = (
                manager.store.storage_path / "checkpoints.sqlite"
                if manager
                else Path(".mini-agent/checkpoints.sqlite")
            )
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_context: AbstractContextManager[SqliteSaver] | None = (
            SqliteSaver.from_conn_string(str(self.checkpoint_path))
        )
        self._checkpointer = self._checkpointer_context.__enter__()
        self.graph: CompiledStateGraph[WorkflowState, None, WorkflowState, WorkflowState] = (
            self._build_graph().compile(checkpointer=self._checkpointer)
        )

    @property
    def conversation_manager(self) -> ConversationManager | None:
        """Expose the underlying conversation manager to CLI lifecycle code."""
        return self.executor.conversation_manager

    def _build_graph(self) -> StateGraph[WorkflowState, None, WorkflowState, WorkflowState]:
        workflow: StateGraph[WorkflowState, None, WorkflowState, WorkflowState] = StateGraph(
            WorkflowState
        )
        workflow.add_node("analyze_question", self._analyze_question)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("prepare_agent", self._prepare_agent)
        workflow.add_node("call_model", self._call_model)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("finish", self._finish)
        workflow.add_edge(START, "analyze_question")
        workflow.add_conditional_edges(
            "analyze_question",
            self._after_analysis,
            {
                "direct": "prepare_agent",
                "retrieve": "retrieve_context",
            },
        )
        workflow.add_conditional_edges(
            "retrieve_context",
            self._after_retrieval,
            {
                "answer": "prepare_agent",
                "tools": "prepare_agent",
            },
        )
        workflow.add_edge("prepare_agent", "call_model")
        workflow.add_conditional_edges(
            "call_model",
            self._after_model,
            {"tools": "execute_tools", "finish": "finish"},
        )
        workflow.add_conditional_edges(
            "execute_tools",
            self._after_tools,
            {"model": "call_model", "finish": "finish"},
        )
        workflow.add_edge("finish", END)
        return workflow

    def _analyze_question(self, state: WorkflowState) -> dict[str, Any]:
        return {"plan": self.planner.plan(state["goal"])}

    @staticmethod
    def _after_analysis(state: WorkflowState) -> str:
        plan = state.get("plan")
        if plan is None:
            raise RuntimeError("Question analysis did not produce a task plan")
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

    def _prepare_agent(self, state: WorkflowState) -> dict[str, Any]:
        agent_state, tool_schemas, tools_allowed = self.executor.initialize(
            goal=state["goal"],
            conversation_id=state.get("conversation_id"),
            additional_context=state.get("context") or state.get("additional_context"),
            plan_override=state["plan"],
            retrieve_context=False,
            tools_allowed_override=state["plan"].allow_tools,
        )
        return {
            "agent_state": agent_state,
            "tool_schemas": tool_schemas,
            "tools_allowed": tools_allowed,
        }

    def _call_model(self, state: WorkflowState) -> dict[str, AgentState]:
        return {
            "agent_state": self.executor.request_model(
                state["agent_state"],
                state["tool_schemas"],
                state.get("conversation_id"),
                on_token=get_token_callback(),
            )
        }

    @staticmethod
    def _after_model(state: WorkflowState) -> str:
        agent_state = state["agent_state"]
        if agent_state["status"] == "RUNNING":
            return "tools"
        return "finish"

    def _execute_tools(self, state: WorkflowState) -> dict[str, AgentState]:
        return {
            "agent_state": self.executor.execute_pending_tools(
                state["agent_state"],
                state["tools_allowed"],
                state.get("conversation_id"),
            )
        }

    @staticmethod
    def _after_tools(state: WorkflowState) -> str:
        agent_state = state["agent_state"]
        if agent_state["status"] == "RUNNING":
            return "model"
        return "finish"

    def _finish(self, state: WorkflowState) -> dict[str, AgentState]:
        return {
            "result": self.executor.finalize(
                state["agent_state"],
                state.get("conversation_id"),
            )
        }

    def _resolve_conversation_id(self, conversation_id: str | None) -> str:
        if conversation_id:
            return conversation_id
        manager = self.executor.conversation_manager
        if manager:
            conversation = manager.get_active_conversation() or manager.start_conversation()
            resolved_id = conversation.conversation_id
            if isinstance(resolved_id, str):
                return resolved_id
            raise TypeError("Conversation ID must be a string")
        return str(uuid.uuid4())

    def execute(
        self,
        goal: str,
        conversation_id: str | None = None,
        additional_context: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentState:
        """Run one request through the compiled routing graph."""
        thread_id = self._resolve_conversation_id(conversation_id)
        initial_state: WorkflowState = {
            "goal": goal,
            "conversation_id": thread_id,
            "additional_context": additional_context,
        }
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        with use_token_callback(on_token):
            final_state = self.graph.invoke(initial_state, config=config)
        return self._extract_result(final_state)

    def resume(self, conversation_id: str) -> AgentState:
        """Resume an interrupted request from its last persisted graph checkpoint."""
        config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot.next:
            raise ValueError(f"No interrupted graph execution for conversation {conversation_id!r}")
        final_state = self.graph.invoke(None, config=config)
        return self._extract_result(final_state)

    @staticmethod
    def _extract_result(state: object) -> AgentState:
        if not isinstance(state, dict):
            raise RuntimeError("Workflow returned an invalid state")
        result = state.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Workflow ended without an agent result")
        required_keys = {"goal", "messages", "step_count", "max_steps", "status", "plan"}
        if not required_keys.issubset(result):
            raise RuntimeError("Workflow returned an incomplete agent result")
        if result["status"] not in {"RUNNING", "SUCCESS", "FAILED"}:
            raise RuntimeError("Workflow returned an invalid agent status")
        return {
            "goal": result["goal"],
            "messages": result["messages"],
            "step_count": result["step_count"],
            "max_steps": result["max_steps"],
            "status": result["status"],
            "plan": result["plan"],
        }

    def close(self) -> None:
        """Release the SQLite checkpoint database connection."""
        if self._checkpointer_context is not None:
            self._checkpointer_context.__exit__(None, None, None)
            self._checkpointer_context = None

    def __enter__(self) -> LangGraphAgent:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
