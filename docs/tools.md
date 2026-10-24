# Tools

Tools are registered on a `ToolRegistry` and exposed to the LLM only when the planner and CLI allow it.

## Always available (read-only chat)

- `read_file`, `list_directory`, `search_code`
- `find_definition`, `find_references`, `search_symbols`
- `git_status`, `git_diff`

## Opt-in (chat flags)

- `--allow-write` → `write_file` (repository-root confined, secrets blocked)
- `--allow-terminal` → `run_terminal`, `run_tests`

`run_tests` runs `python -m pytest <path> -q` under the same command policy as `run_terminal` (no shell metacharacters, timeout, output cap).

## Optional MCP tools

The chat CLI can connect to configured local MCP servers over stdio. MCP is an
optional dependency; install it with `pip install -e '.[mcp]'`. Configure
`MINI_AGENT_MCP_SERVERS` in `.env` as a JSON object keyed by server name:

```dotenv
MINI_AGENT_MCP_SERVERS='{"filesystem":{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/path/to/repo"]}}'
```

Each server entry accepts `command`, optional `args`, and optional `env`. MCP
server processes are started locally and can expose powerful tools, so only
configure servers you trust. The SDK gives child processes an allow-listed
environment; pass any required variables explicitly in that server's `env`.
Discovered tools are namespaced as `mcp_<server>_<tool>` and keep their JSON
input schemas. Their descriptions and results are treated as untrusted content.
Configured MCP tools are available in regular chat, not in `--multi-agent`
researchers, whose registry remains read-only.

## Security

All path and command checks go through `SecurityPolicy` in `coding_agent/security/policy.py`.
MCP servers are configured separately and do not run through that repository
command policy; their local process and tool permissions are controlled by the
server configuration and the server itself.
