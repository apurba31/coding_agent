"""Typed decisions produced by the lightweight task planner."""

from dataclasses import dataclass
from enum import StrEnum


class PlanRoute(StrEnum):
    """Control paths available to the initial deterministic planner."""

    DIRECT = "direct"
    RETRIEVE = "retrieve"
    TOOL = "tool"


@dataclass(frozen=True)
class TaskPlan:
    """A transparent plan describing the minimum steps for a user request."""

    route: PlanRoute
    reason: str

    @property
    def needs_repository_context(self) -> bool:
        return self.route in {PlanRoute.RETRIEVE, PlanRoute.TOOL}

    @property
    def allow_tools(self) -> bool:
        return self.route == PlanRoute.TOOL
