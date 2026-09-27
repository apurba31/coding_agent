"""Tool registry and execution framework."""

import time
import traceback
from collections.abc import Callable

from coding_agent.observability import MetricsCollector, get_metrics_collector

from .models import ToolCall, ToolDefinition, ToolParameter, ToolResult


class ToolRegistry:
    """Manages registered tools and their definitions."""

    def __init__(self, metrics: MetricsCollector | None = None):
        self._tools: dict[str, Callable] = {}
        self._definitions: dict[str, ToolDefinition] = {}
        self.metrics = metrics or get_metrics_collector()

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: list[ToolParameter] | None = None,
        return_type: str = "string",
    ) -> None:
        """Register a new tool.

        Args:
            name: Tool identifier (must be unique).
            func: Callable to execute.
            description: Human-readable description.
            parameters: List of parameter definitions.
            return_type: Type of return value.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        self._tools[name] = func
        self._definitions[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            return_type=return_type,
        )

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._definitions.get(name)

    def get_all_definitions(self) -> dict[str, ToolDefinition]:
        """Get all registered tool definitions."""
        return self._definitions.copy()

    def has_tool(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_call: ToolCall object with name and arguments.

        Returns:
            ToolResult with success status and result/error.
        """
        if not self.has_tool(tool_call.tool_name):
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_call.tool_name}' not found",
                call_id=tool_call.call_id,
            )

        tool_func = self._tools[tool_call.tool_name]
        start_time = time.time()

        try:
            with self.metrics.measure("tool.execution"):
                result = tool_func(**tool_call.arguments)
            execution_time = (time.time() - start_time) * 1000
            self.metrics.increment("tool.calls")
            self.metrics.observe("tool.execution_ms", execution_time)

            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
                call_id=tool_call.call_id,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.metrics.increment("tool.calls")
            self.metrics.increment("tool.failures")
            self.metrics.observe("tool.execution_ms", execution_time)
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=error_msg,
                execution_time_ms=execution_time,
                call_id=tool_call.call_id,
            )


class ToolExecutor:
    """Executes tool calls sequentially or in batches."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        return self.registry.call(tool_call)

    def execute_batch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls sequentially.

        Args:
            tool_calls: List of ToolCall objects.

        Returns:
            List of ToolResult objects in the same order.
        """
        results = []
        for tool_call in tool_calls:
            result = self.execute(tool_call)
            results.append(result)
        return results
