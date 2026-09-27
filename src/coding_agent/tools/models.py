"""Data models for tool calling framework."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolParameter:
    """Represents a single parameter of a tool."""
    name: str
    description: str
    param_type: str  # "string", "integer", "list", "dict", "boolean"
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ToolDefinition:
    """Metadata about a tool that can be called."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    return_type: str = "string"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.param_type,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
        }


@dataclass
class ToolCall:
    """Represents a tool invocation from the LLM."""
    tool_name: str
    arguments: dict[str, Any]
    call_id: Optional[str] = None  # For tracking multi-turn conversations


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    result: Any  # Can be string, dict, list, etc.
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    call_id: Optional[str] = None
