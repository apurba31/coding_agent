"""Conversation memory store and management."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Conversation, Message, MessageRole


class ConversationStore:
    """Manages conversation persistence and retrieval."""

    def __init__(self, storage_path: str | Path = ".mini-agent/conversations"):
        """Initialize conversation store.

        Args:
            storage_path: Directory to store conversation files.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._conversations: dict[str, Conversation] = {}

    def create_conversation(
        self, conversation_id: str | None = None, title: str | None = None
    ) -> Conversation:
        """Create a new conversation.

        Args:
            conversation_id: Optional custom ID. If not provided, generates UUID.
            title: Optional conversation title.

        Returns:
            New Conversation object.
        """
        conv_id = conversation_id or str(uuid.uuid4())
        if conv_id in self._conversations:
            raise ValueError(f"Conversation '{conv_id}' already exists")

        conversation = Conversation(conversation_id=conv_id, title=title)
        self._conversations[conv_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get conversation by ID."""
        return self._conversations.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: ID of conversation to delete.

        Returns:
            True if deleted, False if not found.
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            # Also delete from disk
            file_path = self.storage_path / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations with metadata."""
        result = []
        for conv_id, conv in self._conversations.items():
            stats = conv.compute_stats()
            result.append(
                {
                    "conversation_id": conv_id,
                    "title": conv.title or f"Conversation {conv_id[:8]}",
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "message_count": stats.total_messages,
                    "user_turns": stats.total_user_turns,
                    "assistant_turns": stats.total_assistant_turns,
                }
            )
        return result

    def save_conversation(self, conversation_id: str) -> bool:
        """Save conversation to disk.

        Args:
            conversation_id: ID of conversation to save.

        Returns:
            True if saved, False if not found.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        file_path = self.storage_path / f"{conversation_id}.json"
        with open(file_path, "w") as f:
            json.dump(conversation.to_dict(), f, indent=2)
        return True

    def load_conversation(self, conversation_id: str) -> Conversation | None:
        """Load conversation from disk.

        Args:
            conversation_id: ID of conversation to load.

        Returns:
            Loaded Conversation object or None if not found.
        """
        file_path = self.storage_path / f"{conversation_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            conversation = Conversation(
                conversation_id=data["conversation_id"],
                title=data.get("title"),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                metadata=data.get("metadata", {}),
            )

            # Reconstruct messages
            for msg_data in data.get("messages", []):
                message = Message(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=msg_data.get("metadata", {}),
                    tool_calls=msg_data.get("tool_calls"),
                    tool_results=msg_data.get("tool_results"),
                )
                conversation.messages.append(message)

            self._conversations[conversation_id] = conversation
            return conversation
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None

    def save_all_conversations(self) -> int:
        """Save all in-memory conversations to disk.

        Returns:
            Number of conversations saved.
        """
        count = 0
        for conv_id in self._conversations:
            if self.save_conversation(conv_id):
                count += 1
        return count


class ConversationManager:
    """High-level manager for active conversations."""

    def __init__(self, store: ConversationStore | None = None):
        """Initialize conversation manager.

        Args:
            store: ConversationStore instance. If None, creates default.
        """
        self.store = store or ConversationStore()
        self._active_conversation: str | None = None

    def start_conversation(
        self, conversation_id: str | None = None, title: str | None = None
    ) -> Conversation:
        """Start a new conversation.

        Args:
            conversation_id: Optional custom ID.
            title: Optional title.

        Returns:
            New Conversation object.
        """
        conversation = self.store.create_conversation(conversation_id, title)
        self._active_conversation = conversation.conversation_id
        return conversation

    def get_active_conversation(self) -> Conversation | None:
        """Get the currently active conversation."""
        if self._active_conversation:
            return self.store.get_conversation(self._active_conversation)
        return None

    def switch_conversation(self, conversation_id: str) -> Conversation | None:
        """Switch to a different conversation.

        Args:
            conversation_id: ID of conversation to switch to.

        Returns:
            The conversation if found, None otherwise.
        """
        conversation = self.store.get_conversation(conversation_id)
        if conversation:
            self._active_conversation = conversation_id
        return conversation

    def add_user_message(self, content: str) -> Message | None:
        """Add a user message to active conversation.

        Args:
            content: Message content.

        Returns:
            The created Message or None if no active conversation.
        """
        conv = self.get_active_conversation()
        if not conv:
            return None
        return conv.add_message(MessageRole.USER, content)

    def add_assistant_message(
        self, content: str, tool_calls: list | None = None, tool_results: list | None = None
    ) -> Message | None:
        """Add an assistant message to active conversation.

        Args:
            content: Message content.
            tool_calls: Optional list of tool calls.
            tool_results: Optional list of tool results.

        Returns:
            The created Message or None if no active conversation.
        """
        conv = self.get_active_conversation()
        if not conv:
            return None
        return conv.add_message(
            MessageRole.ASSISTANT,
            content,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    def get_context_window(self, max_messages: int = 10) -> list[dict[str, Any]]:
        """Get context window from active conversation for LLM.

        Args:
            max_messages: Maximum recent messages to include.

        Returns:
            List of message dicts or empty list if no active conversation.
        """
        conv = self.get_active_conversation()
        return conv.get_context_window(max_messages) if conv else []

    def get_conversation_summary(self, conversation_id: str | None = None) -> dict[str, Any]:
        """Get summary of a conversation.

        Args:
            conversation_id: ID of conversation. If None, uses active.

        Returns:
            Dictionary with conversation metadata and stats.
        """
        conv_id = conversation_id or self._active_conversation
        if not conv_id:
            return {}

        conv = self.store.get_conversation(conv_id)
        if not conv:
            return {}

        stats = conv.compute_stats()
        return {
            "conversation_id": conv_id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "total_messages": stats.total_messages,
            "user_turns": stats.total_user_turns,
            "assistant_turns": stats.total_assistant_turns,
            "tool_calls": stats.total_tool_calls,
            "duration_seconds": stats.conversation_duration_seconds,
        }
