"""Agent state models."""
from typing import Any, Literal, TypedDict

from coding_agent.planner.models import TaskPlan


class AgentState(TypedDict):
    """Represents the current state of the agent execution."""

    goal: str
    messages: list[dict[str, Any]]
    step_count: int
    max_steps: int
    status: Literal["RUNNING", "SUCCESS", "FAILED"]
    plan: TaskPlan | None
    
