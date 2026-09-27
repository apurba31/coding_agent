"""Repository-aware prompt construction for the coding assistant."""

from __future__ import annotations

from typing import Any

from coding_agent.search.models import SearchResult
from coding_agent.tools.models import ToolDefinition
from coding_agent.tools.registry import ToolRegistry


class PromptBuilder:
    """Constructs agent messages and repository context blocks."""

    def __init__(self, system_prompt: str, token_budget: int = 1200):
        self.system_prompt = system_prompt
        self.token_budget = token_budget

    def build_initial_messages(
        self,
        goal: str,
        history: list[dict[str, Any]] | None = None,
        context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the initial system + history + user prompt for the model."""
        system_content = self.system_prompt
        if context:
            system_content += "\n\n" + context

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history)
            if history[-1].get("role") == "user" and history[-1].get("content") == goal:
                return messages

        messages.append({"role": "user", "content": goal})
        return messages

    def build_repository_context(
        self,
        results: list[SearchResult],
        token_budget: int | None = None,
    ) -> str:
        """Format repository snippets into a bounded, deterministic context block."""
        budget = self.token_budget if token_budget is None else token_budget
        if budget <= 0:
            return "<repository_context>\n</repository_context>"

        included: list[tuple[str, SearchResult]] = []
        seen_paths: set[str] = set()
        budget_used = 0

        for result in sorted(results, key=lambda item: item.score, reverse=True):
            chunk = result.chunk
            chunk_path = chunk.path.as_posix() if hasattr(chunk.path, "as_posix") else str(chunk.path)
            snippet = (
                f"FILE: {chunk_path}\n"
                f"LANGUAGE: {chunk.language}\n"
                f"SYMBOL: {chunk.display_name}\n"
                f"LINES: {chunk.start_line}-{chunk.end_line}\n\n"
                f"```{chunk.language.lower()}\n{chunk.code}\n```\n"
            )
            token_estimate = max(1, len(snippet.split()))
            if budget_used + token_estimate > budget:
                continue
            if chunk_path not in seen_paths:
                seen_paths.add(chunk_path)
            else:
                # Keep at most one snippet per file to avoid noisy duplication.
                continue
            included.append((snippet, result))
            budget_used += token_estimate

        if not included:
            return "<repository_context>\n</repository_context>"

        context = "<repository_context>\n"
        for snippet, _ in included:
            context += snippet + "\n"
        context += "</repository_context>"
        return context

    def convert_tool_to_openai_schema(self, tool_def: ToolDefinition) -> dict[str, Any]:
        """Convert a tool definition into the schema shape used by Groq/OpenAI."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        json_schema_types = {
            "string": "string",
            "integer": "integer",
            "int": "integer",
            "float": "number",
            "number": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
        }

        for param in tool_def.parameters:
            prop = {
                "type": json_schema_types.get(param.param_type, param.param_type),
                "description": param.description,
            }
            if not param.required and param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def get_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Return all tool schemas in the model-friendly format."""
        return [
            self.convert_tool_to_openai_schema(definition)
            for definition in registry.get_all_definitions().values()
        ]
