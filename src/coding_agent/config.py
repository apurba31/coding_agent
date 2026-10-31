"""Central runtime configuration.

Values live here instead of being hard-coded in CLI, embedders, and tools so
experiments (model swap, token budget, DB location) stay one-layer changes.
Environment variables with prefix MINI_AGENT_ override defaults; GROQ_API_KEY
is read separately by the LLM client and is never stored on this object.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseModel):
    """Command and explicit environment for one local stdio MCP server."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseSettings):
    """Replaceable knobs for indexing, retrieval, and tool safety."""

    model_config = SettingsConfigDict(
        env_prefix="MINI_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    llm_provider: Literal["groq", "nvidia"] = "groq"
    llm_model: str | None = None
    repository_path: Path = Path(".")
    vector_db_path: Path = Path(".mini-agent/lancedb")
    memory_path: Path = Path(".mini-agent/conversations")
    top_k: int = 5
    use_reranker: bool = True
    rerank_alpha: float = 0.7
    token_budget: int = 1200
    terminal_timeout_seconds: float = 15.0
    terminal_max_output_bytes: int = 32_000
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_config() -> AgentConfig:
    """Return the process-wide config. Tests can call get_config.cache_clear()."""
    return AgentConfig()
