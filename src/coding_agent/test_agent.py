import logging
import os

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.llm.client import GroqClient
from coding_agent.tools.models import ToolParameter
from coding_agent.tools.registry import ToolRegistry

logging.basicConfig(level=logging.INFO)


def example_calculator(a: float, b: float, operation: str) -> float:
    """A simple calculator tool for testing the agent."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def main():
    # Make sure to have GROQ_API_KEY in the environment
    # os.environ["GROQ_API_KEY"] = "your-api-key"
    if not os.environ.get("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY to test the agent.")
        return

    # Setup Tool Registry
    registry = ToolRegistry()
    registry.register(
        name="calculator",
        func=example_calculator,
        description="Performs basic arithmetic operations.",
        parameters=[
            ToolParameter(name="a", type="number", description="The first number", required=True),
            ToolParameter(name="b", type="number", description="The second number", required=True),
            ToolParameter(
                name="operation",
                type="string",
                description="The operation to perform",
                enum=["add", "subtract", "multiply", "divide"],
                required=True,
            ),
        ],
        return_type="number",
    )

    # Setup components
    llm_client = GroqClient(default_model="llama3-70b-8192")
    prompt_builder = PromptBuilder(
        system_prompt="You are a helpful AI assistant. You can use tools to answer questions."
    )
    executor = AgentExecutor(
        llm_client=llm_client,
        tool_registry=registry,
        prompt_builder=prompt_builder,
        max_steps=5,
    )

    # Run Agent
    goal = "What is 15.5 multiplied by 4, and then subtract 10 from the result?"
    print(f"\n--- Goal: {goal} ---\n")
    state = executor.execute(goal)

    print("\n--- Final Status ---")
    print(state["status"])
    print("\n--- Messages History ---")
    for msg in state["messages"]:
        print(f"[{msg['role'].upper()}]: {msg.get('content', 'No Content (Tool call likely)')}")
        if "tool_calls" in msg:
            print("  Tools calls:", msg["tool_calls"])


if __name__ == "__main__":
    main()
