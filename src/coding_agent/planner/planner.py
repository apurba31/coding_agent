"""Small, deterministic planner for routing requests through the agent pipeline."""

import re
from abc import ABC, abstractmethod

from .models import PlanRoute, TaskPlan

_REPOSITORY_TERMS = {
    "bug",
    "class",
    "code",
    "file",
    "files",
    "definition",
    "function",
    "functions",
    "implementation",
    "implement",
    "method",
    "methods",
    "module",
    "project",
    "reference",
    "references",
    "repo",
    "repository",
    "source",
    "symbol",
    "symbols",
    "test",
    "tests",
}
_TOOL_INTENTS = {
    "definition",
    "find",
    "locate",
    "open",
    "read",
    "references",
    "search",
    "show",
    "inspect",
}
_CODE_IDENTIFIER = re.compile(
    r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+|"
    r"[A-Za-z]+_[A-Za-z0-9_]+)\b"
)


class Planner(ABC):
    """Interface for request routing decisions."""

    @abstractmethod
    def plan(self, goal: str) -> TaskPlan:
        """Choose direct response, repository retrieval, or explicit tool use."""
        raise NotImplementedError


class SimplePlanner(Planner):
    """Route requests with explainable keyword and identifier heuristics.

    This first planner deliberately avoids another model call. It opts into
    repository work for repository vocabulary, paths, or code-like identifiers,
    and exposes tools only when the user explicitly asks to inspect/find/read.
    """

    def plan(self, goal: str) -> TaskPlan:
        normalized = goal.casefold()
        words = set(re.findall(r"[a-z0-9_]+", normalized))
        has_path = bool(re.search(r"(?:[\w.-]+[/\\])+[\w.-]+|\.[a-z]{1,5}\b", normalized))
        has_identifier = bool(_CODE_IDENTIFIER.search(goal))
        needs_context = bool(words & _REPOSITORY_TERMS) or has_path or has_identifier
        if not needs_context:
            return TaskPlan(
                PlanRoute.DIRECT,
                "No repository-specific terms or identifiers detected.",
            )

        explicit_tool_intent = bool(words & _TOOL_INTENTS)
        if explicit_tool_intent:
            return TaskPlan(
                PlanRoute.TOOL,
                "Repository context is relevant and the request explicitly asks "
                "to find or inspect code.",
            )
        return TaskPlan(
            PlanRoute.RETRIEVE,
            "Repository-specific vocabulary or a code identifier was detected; "
            "retrieve context first.",
        )
