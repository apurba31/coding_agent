"""Optional MCP stdio client and tool-registry adapter."""

import importlib.util
import json
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from contextlib import AsyncExitStack
from typing import Any

from coding_agent.config import MCPServerConfig
from coding_agent.tools.registry import ToolRegistry


def _function_name(server_name: str, tool_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp_{server_name}_{tool_name}")
    if len(name) > 64:
        raise ValueError(
            f"MCP server/tool name '{server_name}/{tool_name}' exceeds the 64-character tool limit"
        )
    return name


def _format_result(result: Any, server_name: str, tool_name: str) -> str:
    content = getattr(result, "content", [])
    text_parts = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
            continue

        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            text_parts.append(json.dumps(model_dump(mode="json"), ensure_ascii=False))
        else:
            text_parts.append(str(block))

    structured_content = getattr(
        result, "structured_content", getattr(result, "structuredContent", None)
    )
    if structured_content is not None:
        rendered = json.dumps(structured_content, ensure_ascii=False)
    elif text_parts:
        rendered = "\n".join(text_parts)
    else:
        rendered = "(MCP tool returned no content)"

    if getattr(result, "is_error", getattr(result, "isError", False)):
        raise RuntimeError(f"MCP tool {server_name}/{tool_name} failed: {rendered}")
    return rendered


def register_mcp_tools(
    registry: ToolRegistry,
    server_name: str,
    tools: Iterable[Any],
    call_tool: Callable[[str, str, dict[str, Any]], Any],
) -> None:
    """Register discovered MCP tools with their complete JSON input schemas."""
    for tool in tools:
        tool_name = tool.name
        input_schema = getattr(tool, "input_schema", getattr(tool, "inputSchema", None))
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            raise ValueError(
                f"MCP tool '{server_name}/{tool_name}' must have an object input schema"
            )

        registry_name = _function_name(server_name, tool_name)

        def invoke(
            _server_name: str = server_name,
            _tool_name: str = tool_name,
            **arguments: Any,
        ) -> str:
            result = call_tool(_server_name, _tool_name, arguments)
            return _format_result(result, _server_name, _tool_name)

        description = tool.description or f"Tool provided by MCP server '{server_name}'."
        registry.register(
            registry_name,
            invoke,
            f"[MCP server: {server_name}] {description}",
            input_schema=input_schema,
        )


class MCPToolManager:
    """Own persistent stdio MCP sessions for the lifetime of a CLI chat."""

    def __init__(
        self,
        servers: Mapping[str, MCPServerConfig],
        registry: ToolRegistry,
    ) -> None:
        self._portal_context: Any | None = None
        self._portal: Any | None = None
        self._serve_future: Future[Any] | None = None
        self._request_send: Any | None = None

        if not servers:
            return
        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError(
                "MCP servers are configured, but the MCP SDK is not installed. "
                "Install the optional dependency with `pip install -e '.[mcp]'`."
            )

        from anyio.from_thread import start_blocking_portal

        self._portal_context = start_blocking_portal()
        self._portal = self._portal_context.__enter__()
        try:
            self._serve_future, tools_by_server = self._portal.start_task(self._serve, servers)
            for server_name, tools in tools_by_server.items():
                register_mcp_tools(registry, server_name, tools, self._call_tool_sync)
        except BaseException:
            self.close()
            raise

    async def _serve(self, servers: Mapping[str, MCPServerConfig], *, task_status: Any) -> None:
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._request_send, request_receive = anyio.create_memory_object_stream[Any](0)
        tools_by_server: dict[str, list[Any]] = {}
        clients: dict[str, Any] = {}
        async with AsyncExitStack() as stack:
            for server_name, server in servers.items():
                try:
                    read_stream, write_stream = await stack.enter_async_context(
                        stdio_client(
                            StdioServerParameters(
                                command=server.command,
                                args=server.args,
                                env=server.env or None,
                            )
                        )
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                    await session.initialize()
                    clients[server_name] = session
                    result = await session.list_tools()
                    tools_by_server[server_name] = result.tools
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to start or discover tools from MCP server '{server_name}'"
                    ) from error

            task_status.started(tools_by_server)
            async with request_receive:
                async for request in request_receive:
                    if request is None:
                        break
                    server_name, tool_name, arguments, response = request
                    try:
                        result = await clients[server_name].call_tool(tool_name, arguments)
                    except Exception as error:
                        response.set_exception(error)
                    else:
                        response.set_result(result)

    async def _enqueue_tool_call(self, request: Any) -> None:
        if self._request_send is None:
            raise RuntimeError("MCP client is not connected")
        await self._request_send.send(request)

    async def _enqueue_shutdown(self) -> None:
        if self._request_send is not None:
            await self._request_send.send(None)

    def _call_tool_sync(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self._portal is None:
            raise RuntimeError("MCP client is not connected")
        response: Future[Any] = Future()
        request = (server_name, tool_name, arguments, response)
        self._portal.call(self._enqueue_tool_call, request)
        return response.result()

    async def _stop_server_task(self) -> None:
        await self._enqueue_shutdown()

    def close(self) -> None:
        """Close connected MCP sessions and stop their async runtime."""
        if self._portal_context is None:
            return

        portal_context = self._portal_context
        try:
            if self._portal is not None and self._serve_future is not None:
                self._portal.call(self._stop_server_task)
                self._serve_future.result()
        finally:
            portal_context.__exit__(None, None, None)
            self._portal_context = None
            self._portal = None
            self._request_send = None
