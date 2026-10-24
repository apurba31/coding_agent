"""Tests for the optional MCP tool adapter."""

import sys
from types import SimpleNamespace

import pytest

from coding_agent.agent.prompt import PromptBuilder
from coding_agent.config import AgentConfig, MCPServerConfig
from coding_agent.tools.mcp_client import MCPToolManager, register_mcp_tools
from coding_agent.tools.models import ToolCall
from coding_agent.tools.registry import ToolRegistry


def _tool(name="read_file"):
    return SimpleNamespace(
        name=name,
        description="Read a file from the configured server.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {"encoding": {"type": "string"}},
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def test_register_mcp_tool_preserves_schema_and_dispatches_call():
    registry = ToolRegistry()
    tool = _tool()
    calls = []

    def call_tool(server_name, tool_name, arguments):
        calls.append((server_name, tool_name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(text="file contents")],
            structuredContent=None,
            isError=False,
        )

    register_mcp_tools(registry, "filesystem", [tool], call_tool)

    schema = PromptBuilder("system").get_tool_schemas(registry)[0]
    assert schema["function"]["name"] == "mcp_filesystem_read_file"
    assert schema["function"]["parameters"] == tool.inputSchema

    result = registry.call(
        ToolCall(
            tool_name="mcp_filesystem_read_file",
            arguments={"path": "notes.txt", "options": {"encoding": "utf-8"}},
        )
    )

    assert result.success
    assert result.result == "file contents"
    assert calls == [
        (
            "filesystem",
            "read_file",
            {"path": "notes.txt", "options": {"encoding": "utf-8"}},
        )
    ]


def test_mcp_tool_errors_are_reported_as_tool_failures():
    registry = ToolRegistry()
    register_mcp_tools(
        registry,
        "filesystem",
        [_tool()],
        lambda *_: SimpleNamespace(
            content=[SimpleNamespace(text="permission denied")],
            structuredContent=None,
            isError=True,
        ),
    )

    result = registry.call(ToolCall("mcp_filesystem_read_file", {"path": "secret.txt"}))

    assert not result.success
    assert "permission denied" in result.error


def test_mcp_tool_name_collision_is_rejected():
    registry = ToolRegistry()
    registry.register("mcp_filesystem_read_file", lambda: None, "Existing tool")

    with pytest.raises(ValueError, match="already registered"):
        register_mcp_tools(registry, "filesystem", [_tool()], lambda *_: None)


def test_mcp_tool_schema_must_describe_an_object():
    registry = ToolRegistry()
    tool = _tool()
    tool.inputSchema = {"type": "string"}

    with pytest.raises(ValueError, match="object input schema"):
        register_mcp_tools(registry, "filesystem", [tool], lambda *_: None)


def test_mcp_configuration_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv(
        "MINI_AGENT_MCP_SERVERS",
        '{"filesystem":{"command":"npx","args":["-y","server"],"env":{"API_TOKEN":"value"}}}',
    )

    config = AgentConfig(_env_file=None)

    assert config.mcp_servers == {
        "filesystem": MCPServerConfig(
            command="npx",
            args=["-y", "server"],
            env={"API_TOKEN": "value"},
        )
    }


def test_configured_mcp_reports_missing_optional_sdk(monkeypatch):
    monkeypatch.setattr("coding_agent.tools.mcp_client.importlib.util.find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[mcp\]'"):
        MCPToolManager(
            {"filesystem": MCPServerConfig(command="npx")},
            ToolRegistry(),
        )


def test_stdio_server_tools_are_discovered_invoked_and_closed(tmp_path):
    pytest.importorskip("mcp")
    server_script = tmp_path / "server.py"
    server_script.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "server = FastMCP('test-server')\n"
        "@server.tool()\n"
        "def echo(value: str) -> str:\n"
        "    return f'echo:{value}'\n"
        "if __name__ == '__main__':\n"
        "    server.run()\n",
        encoding="utf-8",
    )
    registry = ToolRegistry()
    manager = MCPToolManager(
        {"test": MCPServerConfig(command=sys.executable, args=[str(server_script)])},
        registry,
    )

    try:
        result = registry.call(ToolCall("mcp_test_echo", {"value": "hello"}))
    finally:
        manager.close()

    assert result.success
    assert result.result == '{"result": "echo:hello"}'


def test_stdio_server_startup_failure_is_reported(tmp_path):
    pytest.importorskip("mcp")

    with pytest.raises(RuntimeError, match="Failed to start or discover tools"):
        MCPToolManager(
            {
                "missing": MCPServerConfig(
                    command=str(tmp_path / "missing-server"),
                )
            },
            ToolRegistry(),
        )
