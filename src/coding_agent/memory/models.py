"""Data models for conversation memory."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class MessageRole(str, Enum):
    """Role of message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Optional: Tool-related fields
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_results: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization or LLM consumption."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }


@dataclass
class ConversationStats:
    """Statistics about a conversation."""
    total_messages: int = 0
    total_user_turns: int = 0
    total_assistant_turns: int = 0
    total_tool_calls: int = 0
    avg_message_length: float = 0.0
    conversation_duration_seconds: float = 0.0


@dataclass
class Conversation:
    """Represents a multi-turn conversation."""
    conversation_id: str
    title: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, etc.).
            content: Message content.
            **kwargs: Additional fields (metadata, tool_calls, tool_results).

        Returns:
            The created Message object.
        """
        message = Message(
            role=role,
            content=content,
            metadata=kwargs.get("metadata", {}),
            tool_calls=kwargs.get("tool_calls"),
            tool_results=kwargs.get("tool_results"),
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_messages(
        self, limit: Optional[int] = None, role_filter: Optional[MessageRole] = None
    ) -> list[Message]:
        """Get messages with optional filtering.

        Args:
            limit: Maximum number of recent messages to return.
            role_filter: Filter messages by role.

        Returns:
            List of messages.
        """
        filtered = self.messages
        if role_filter:
            filtered = [m for m in filtered if m.role == role_filter]
        if limit:
            filtered = filtered[-limit:]
        return filtered

    def get_context_window(
        self,
        max_messages: int = 10,
        include_summary: bool = False,
        summary_limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get recent messages as context for an LLM, optionally with a rolling summary.

        Args:
            max_messages: Maximum recent messages to include.
            include_summary: Whether to prepend a short summary of earlier turns.
            summary_limit: Maximum character budget for the summary text.

        Returns:
            List of message dictionaries suitable for LLM API.
        """
        recent = self.get_messages(limit=max_messages)
        if not include_summary:
            return [m.to_dict() for m in recent]

        if len(self.messages) <= max_messages:
            return [m.to_dict() for m in recent]

        summary_messages = self.get_messages()[:-max_messages]
        summary_parts = [f"{m.role.value}: {m.content}" for m in summary_messages]
        summary_text = " ".join(summary_parts)
        if len(summary_text) > summary_limit:
            summary_text = summary_text[:summary_limit].rstrip() + "..."

        return [
            {"role": MessageRole.SYSTEM.value, "content": f"Conversation summary: {summary_text}"},
            *[m.to_dict() for m in recent],
        ]

    def compute_stats(self) -> ConversationStats:
        """Compute statistics about the conversation."""
        stats = ConversationStats()
        stats.total_messages = len(self.messages)

        if not self.messages:
            return stats

        stats.total_user_turns = len(
            [m for m in self.messages if m.role == MessageRole.USER]
        )
        stats.total_assistant_turns = len(
            [m for m in self.messages if m.role == MessageRole.ASSISTANT]
        )

        # Count tool calls
        for message in self.messages:
            if message.tool_calls:
                stats.total_tool_calls += len(message.tool_calls)

        # Average message length
        total_length = sum(len(m.content) for m in self.messages)
        stats.avg_message_length = total_length / len(self.messages) if self.messages else 0.0

        # Conversation duration
        if len(self.messages) > 1:
            first_msg = self.messages[0]
            last_msg = self.messages[-1]
            duration = (last_msg.timestamp - first_msg.timestamp).total_seconds()
            stats.conversation_duration_seconds = max(0, duration)

        return stats

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages = []
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "stats": self.compute_stats().__dict__,
        }
