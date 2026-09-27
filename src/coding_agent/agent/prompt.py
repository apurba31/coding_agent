"""Prompt builder and tool schema conversion."""

from typing import Any

from coding_agent.tools.models import ToolDefinition
from coding_agent.tools.registry import ToolRegistry


class PromptBuilder:
    """Constructs prompts and handles tool schema conversion for Groq/OpenAI APIs."""

    def __init__(self, system_prompt: str):
        """Initialize PromptBuilder with a base system prompt.

        Args:
            system_prompt: The base system instruction for the agent.
        """
        self.system_prompt = system_prompt

    def build_initial_messages(
        self,
        goal: str,
        history: list[dict[str, Any]] | None = None,
        context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build messages with optional conversation history and retrieved code context.

        Args:
            goal: The task or goal for the agent.
            history: Prior API-compatible messages, including the current user turn if present.
            context: Relevant repository context to add to the system instruction.

        Returns:
            Initial list of messages.
        """
        system_content = self.system_prompt
        if context:
            system_content += "\n\nRelevant repository context:\n" + context

        messages = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history)
            if history[-1].get("role") == "user" and history[-1].get("content") == goal:
                return messages

        messages.append({"role": "user", "content": goal})
        return messages

    def convert_tool_to_openai_schema(self, tool_def: ToolDefinition) -> dict[str, Any]:
        """Convert a ToolDefinition to the OpenAI/Groq tool format.

        Args:
            tool_def: The internal tool definition.

        Returns:
            Dictionary formatted as an OpenAI function tool.
        """
        properties = {}
        required = []
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
        """Get all tool schemas in OpenAI/Groq format from a registry.

        Args:
            registry: The ToolRegistry instance containing registered tools.

        Returns:
            List of tool schemas.
        """
        schemas = []
        for definition in registry.get_all_definitions().values():
            schemas.append(self.convert_tool_to_openai_schema(definition))
        return schemas
