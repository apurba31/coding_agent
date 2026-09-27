"""Tests for tool calling framework."""

import pytest
import time
from coding_agent.tools import (
    ToolRegistry,
    ToolExecutor,
    ToolDefinition,
    ToolParameter,
    ToolCall,
    ToolResult,
)


# Mock tool functions
def mock_read_file(filepath: str) -> str:
    """Mock file reader tool."""
    if filepath == "/tmp/test.txt":
        return "File contents"
    raise FileNotFoundError(f"File not found: {filepath}")


def mock_calculate(x: int, y: int, operation: str = "add") -> int:
    """Mock calculator tool."""
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    else:
        raise ValueError(f"Unknown operation: {operation}")


def mock_slow_operation(delay_seconds: float = 1.0) -> str:
    """Mock slow operation."""
    time.sleep(delay_seconds)
    return f"Completed after {delay_seconds}s"


@pytest.fixture
def registry():
    """Create a tool registry with sample tools."""
    registry = ToolRegistry()

    # Register read_file tool
    registry.register(
        name="read_file",
        func=mock_read_file,
        description="Read contents of a file",
        parameters=[
            ToolParameter(
                name="filepath",
                description="Path to the file",
                param_type="string",
                required=True,
            )
        ],
        return_type="string",
    )

    # Register calculate tool
    registry.register(
        name="calculate",
        func=mock_calculate,
        description="Perform arithmetic operations",
        parameters=[
            ToolParameter(
                name="x",
                description="First operand",
                param_type="integer",
                required=True,
            ),
            ToolParameter(
                name="y",
                description="Second operand",
                param_type="integer",
                required=True,
            ),
            ToolParameter(
                name="operation",
                description="Operation to perform",
                param_type="string",
                required=False,
                default="add",
            ),
        ],
        return_type="integer",
    )

    # Register slow operation tool
    registry.register(
        name="slow_operation",
        func=mock_slow_operation,
        description="A slow operation for testing",
        parameters=[
            ToolParameter(
                name="delay_seconds",
                description="How long to delay",
                param_type="float",
                required=False,
                default=1.0,
            )
        ],
        return_type="string",
    )

    return registry


def test_tool_registry_register(registry):
    """Test registering tools."""
    assert registry.has_tool("read_file")
    assert registry.has_tool("calculate")
    assert registry.has_tool("slow_operation")


def test_tool_registry_duplicate_registration(registry):
    """Test that registering same tool twice raises error."""
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            name="read_file",
            func=lambda x: x,
            description="Duplicate",
        )


def test_tool_definition_retrieval(registry):
    """Test retrieving tool definitions."""
    definition = registry.get_definition("read_file")
    assert definition is not None
    assert definition.name == "read_file"
    assert definition.return_type == "string"
    assert len(definition.parameters) == 1
    assert definition.parameters[0].name == "filepath"


def test_tool_definition_to_dict(registry):
    """Test converting tool definition to dictionary."""
    definition = registry.get_definition("calculate")
    definition_dict = definition.to_dict()

    assert definition_dict["name"] == "calculate"
    assert definition_dict["return_type"] == "integer"
    assert len(definition_dict["parameters"]) == 3


def test_tool_execution_success(registry):
    """Test executing a tool successfully."""
    tool_call = ToolCall(
        tool_name="read_file",
        arguments={"filepath": "/tmp/test.txt"},
    )
    result = registry.call(tool_call)

    assert result.success is True
    assert result.tool_name == "read_file"
    assert result.result == "File contents"
    assert result.error is None


def test_tool_execution_with_defaults(registry):
    """Test executing a tool with default parameter values."""
    tool_call = ToolCall(
        tool_name="calculate",
        arguments={"x": 5, "y": 3},
    )
    result = registry.call(tool_call)

    assert result.success is True
    assert result.result == 8  # Default is add


def test_tool_execution_with_explicit_args(registry):
    """Test executing a tool with explicit arguments."""
    tool_call = ToolCall(
        tool_name="calculate",
        arguments={"x": 5, "y": 3, "operation": "multiply"},
    )
    result = registry.call(tool_call)

    assert result.success is True
    assert result.result == 15


def test_tool_execution_failure(registry):
    """Test tool execution with error."""
    tool_call = ToolCall(
        tool_name="read_file",
        arguments={"filepath": "/nonexistent/file.txt"},
    )
    result = registry.call(tool_call)

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "FileNotFoundError" in result.error


def test_tool_not_found(registry):
    """Test calling a non-existent tool."""
    tool_call = ToolCall(
        tool_name="nonexistent_tool",
        arguments={},
    )
    result = registry.call(tool_call)

    assert result.success is False
    assert "not found" in result.error


def test_tool_execution_timing(registry):
    """Test that execution time is measured."""
    tool_call = ToolCall(
        tool_name="slow_operation",
        arguments={"delay_seconds": 0.1},
    )
    result = registry.call(tool_call)

    assert result.success is True
    assert result.execution_time_ms >= 100  # Should be at least 100ms


def test_tool_call_id_tracking(registry):
    """Test that tool call IDs are preserved in results."""
    tool_call = ToolCall(
        tool_name="calculate",
        arguments={"x": 2, "y": 3},
        call_id="call_123",
    )
    result = registry.call(tool_call)

    assert result.call_id == "call_123"
    assert result.success is True


def test_tool_executor_single_call():
    """Test tool executor for single calls."""
    registry = ToolRegistry()
    registry.register(
        name="echo",
        func=lambda msg: f"Echo: {msg}",
        description="Echo a message",
        parameters=[
            ToolParameter(
                name="msg",
                description="Message to echo",
                param_type="string",
            )
        ],
    )

    executor = ToolExecutor(registry)
    tool_call = ToolCall(tool_name="echo", arguments={"msg": "Hello"})
    result = executor.execute(tool_call)

    assert result.success is True
    assert result.result == "Echo: Hello"


def test_tool_executor_batch():
    """Test tool executor for batch calls."""
    registry = ToolRegistry()
    registry.register(
        name="add",
        func=lambda x, y: x + y,
        description="Add two numbers",
        parameters=[
            ToolParameter(name="x", description="First number", param_type="integer"),
            ToolParameter(name="y", description="Second number", param_type="integer"),
        ],
        return_type="integer",
    )

    executor = ToolExecutor(registry)
    tool_calls = [
        ToolCall(tool_name="add", arguments={"x": 1, "y": 2}, call_id="call_1"),
        ToolCall(tool_name="add", arguments={"x": 5, "y": 10}, call_id="call_2"),
        ToolCall(tool_name="add", arguments={"x": 100, "y": 200}, call_id="call_3"),
    ]

    results = executor.execute_batch(tool_calls)

    assert len(results) == 3
    assert results[0].result == 3
    assert results[1].result == 15
    assert results[2].result == 300
    assert results[0].call_id == "call_1"


def test_tool_registry_get_all_definitions():
    """Test getting all tool definitions."""
    registry = ToolRegistry()
    registry.register(
        name="tool1",
        func=lambda: "result1",
        description="First tool",
    )
    registry.register(
        name="tool2",
        func=lambda: "result2",
        description="Second tool",
    )

    all_defs = registry.get_all_definitions()

    assert len(all_defs) == 2
    assert "tool1" in all_defs
    assert "tool2" in all_defs
