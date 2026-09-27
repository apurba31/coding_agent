"""Main agent execution loop."""

import json
import logging
from typing import Any

from coding_agent.llm.client import GroqClient
from coding_agent.memory.models import Conversation, MessageRole
from coding_agent.memory.store import ConversationManager
from coding_agent.observability import MetricsCollector, get_metrics_collector
from coding_agent.planner import Planner
from coding_agent.planner.models import TaskPlan
from coding_agent.tools.models import ToolCall
from coding_agent.tools.registry import ToolExecutor, ToolRegistry

from .prompt import PromptBuilder
from .state import AgentState

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Orchestrates the execution of the agent."""

    def __init__(
        self,
        llm_client: GroqClient,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        max_steps: int = 15,
        conversation_manager: ConversationManager | None = None,
        retriever: Any | None = None,
        context_top_k: int = 5,
        history_limit: int = 10,
        planner: Planner | None = None,
        metrics: MetricsCollector | None = None,
    ):
        """Initialize the agent executor.

        Args:
            llm_client: The LLM client (GroqClient).
            tool_registry: Registry containing available tools.
            prompt_builder: Builder for prompts and tool schemas.
            max_steps: Maximum number of iterations before forcing termination.
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        self.prompt_builder = prompt_builder
        self.max_steps = max_steps
        self.conversation_manager = conversation_manager
        self.retriever = retriever
        self.context_top_k = context_top_k
        self.history_limit = history_limit
        self.planner = planner
        self.metrics = metrics or get_metrics_collector()

    def _get_conversation(self, conversation_id: str | None) -> Conversation | None:
        """Find or create the conversation used for this execution."""
        if not self.conversation_manager:
            return None

        manager = self.conversation_manager
        if conversation_id:
            conversation = manager.store.get_conversation(conversation_id)
            if conversation is None:
                conversation = manager.store.load_conversation(conversation_id)
            if conversation is None:
                return manager.start_conversation(conversation_id=conversation_id)
            manager.switch_conversation(conversation_id)
            return conversation

        conversation = manager.get_active_conversation()
        return conversation or manager.start_conversation()

    def _conversation_history(self, conversation: Conversation) -> list[dict[str, Any]]:
        """Convert persisted messages into the chat API format."""
        history = []
        for message in conversation.get_messages(limit=self.history_limit):
            item: dict[str, Any] = {"role": message.role.value, "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if message.role == MessageRole.TOOL:
                item.update(
                    tool_call_id=message.metadata.get("tool_call_id"),
                    name=message.metadata.get("name"),
                )
            history.append(item)
        return history

    def retrieve_context(self, goal: str) -> str | None:
        """Retrieve relevant code snippets for the current user turn."""
        if not self.retriever:
            return None
        try:
            results = self.retriever.search(goal, top_k=self.context_top_k)
        except Exception:
            logger.exception("Repository context retrieval failed")
            return None

        self.metrics.observe("retrieval.chunks", len(results))
        snippets = []
        for result in results:
            chunk = result.chunk
            snippets.append(
                f"[{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                f"({chunk.display_name}, score={result.score:.3f})]\n{chunk.code}"
            )
        return "\n\n".join(snippets) or None

    def execute(
        self,
        goal: str,
        conversation_id: str | None = None,
        additional_context: str | None = None,
        plan_override: TaskPlan | None = None,
        retrieve_context: bool = True,
        tools_allowed_override: bool | None = None,
    ) -> AgentState:
        """Run the agent loop to achieve the given goal.

        Args:
            goal: The task for the agent.
            conversation_id: Optional persisted conversation to resume.
            additional_context: Extra trusted context supplied by an orchestrator.
            plan_override: Precomputed plan from an outer workflow graph.
            retrieve_context: Whether this executor should perform retrieval itself.
            tools_allowed_override: Optional explicit tool-policy override.

        Returns:
            The final AgentState.
        """
        conversation = self._get_conversation(conversation_id)
        plan = plan_override or (self.planner.plan(goal) if self.planner else None)
        if plan:
            logger.info("Plan route=%s reason=%s", plan.route.value, plan.reason)
        history = None
        if conversation:
            conversation.add_message(MessageRole.USER, goal)
            history = self._conversation_history(conversation)

        retrieved_context = (
            self.retrieve_context(goal)
            if retrieve_context and (plan is None or plan.needs_repository_context)
            else None
        )
        context_parts = [part for part in (retrieved_context, additional_context) if part]
        context = "\n\n".join(context_parts) or None

        # Initialize state
        with self.metrics.measure("prompt.build"):
            initial_messages = self.prompt_builder.build_initial_messages(
                goal, history=history, context=context
            )
        state: AgentState = {
            "goal": goal,
            "messages": initial_messages,
            "step_count": 0,
            "max_steps": self.max_steps,
            "status": "RUNNING",
            "plan": plan,
        }

        # Get tool schemas for the LLM
        tools_allowed = (
            tools_allowed_override
            if tools_allowed_override is not None
            else plan is None or plan.allow_tools
        )
        tool_schemas = (
            self.prompt_builder.get_tool_schemas(self.tool_registry) if tools_allowed else []
        )

        logger.info(f"Starting agent loop for goal: {goal}")

        while state["status"] == "RUNNING" and state["step_count"] < state["max_steps"]:
            state["step_count"] += 1
            logger.info(f"--- Step {state['step_count']}/{state['max_steps']} ---")

            # Call LLM
            try:
                with self.metrics.measure("llm.request"):
                    response = self.llm_client.generate_response(
                        messages=state["messages"],
                        tools=tool_schemas if tool_schemas else None,
                    )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                state["status"] = "FAILED"
                state["messages"].append({"role": "system", "content": f"Fatal Error: {e}"})
                if conversation:
                    conversation.add_message(MessageRole.ASSISTANT, f"Fatal Error: {e}")
                break

            usage = getattr(response, "usage", None)
            if usage:
                for source_name, metric_name in (
                    ("prompt_tokens", "tokens.prompt"),
                    ("completion_tokens", "tokens.completion"),
                    ("total_tokens", "tokens.total"),
                ):
                    token_count = getattr(usage, source_name, None)
                    if token_count is not None:
                        self.metrics.increment(metric_name, int(token_count))

            response_message = response.choices[0].message
            tool_calls = getattr(response_message, "tool_calls", None)

            # Append assistant message to history
            # The Groq python client might return a pydantic model, convert it to dict for messages
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response_message.content:
                assistant_msg["content"] = response_message.content
                logger.info(f"Assistant: {response_message.content}")

            if tool_calls:
                # Format tool calls for the message history
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            state["messages"].append(assistant_msg)
            if conversation:
                conversation.add_message(
                    MessageRole.ASSISTANT,
                    response_message.content or "",
                    tool_calls=assistant_msg.get("tool_calls"),
                )

            # Process tool calls if any
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                        logger.warning(f"Failed to parse arguments for tool {tool_name}")

                    logger.info(f"Tool Call: {tool_name}({arguments})")

                    call_obj = ToolCall(tool_name=tool_name, arguments=arguments, call_id=tc.id)

                    # A model may ignore the plan's tool policy; keep execution gated.
                    if tools_allowed:
                        result = self.tool_executor.execute(call_obj)
                        if result.success:
                            tool_output = str(result.result)
                        else:
                            tool_output = f"Error: {result.error}"
                    else:
                        tool_output = (
                            "Tool execution was not enabled by the task plan. "
                            "Answer using available context instead."
                        )

                    if tools_allowed and result.success:
                        logger.info(
                            "Tool Result (%0.2fms): %s",
                            result.execution_time_ms,
                            result.result,
                        )
                    elif tools_allowed:
                        logger.warning(
                            "Tool Error (%0.2fms): %s",
                            result.execution_time_ms,
                            result.error,
                        )
                    else:
                        logger.warning("Blocked unplanned tool call: %s", tool_name)

                    # Append tool result to history
                    state["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": tool_output,
                        }
                    )
                    if conversation:
                        conversation.add_message(
                            MessageRole.TOOL,
                            tool_output,
                            metadata={"tool_call_id": tc.id, "name": tool_name},
                        )
            else:
                # If no tool calls, the agent has finished its task
                logger.info("No tool calls made. Agent considers task complete.")
                state["status"] = "SUCCESS"
                break

        if state["step_count"] >= state["max_steps"] and state["status"] == "RUNNING":
            logger.warning("Max steps reached. Forcing termination.")
            state["status"] = "FAILED"

        logger.info(f"Agent loop finished with status: {state['status']}")
        if conversation and self.conversation_manager:
            self.conversation_manager.store.save_conversation(conversation.conversation_id)
        return state
