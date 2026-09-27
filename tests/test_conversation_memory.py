"""Tests for conversation memory functionality."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from coding_agent.memory import (
    Message,
    MessageRole,
    Conversation,
    ConversationStats,
    ConversationStore,
    ConversationManager,
)


@pytest.fixture
def sample_conversation():
    """Create a sample conversation."""
    conv = Conversation(conversation_id="test_conv_1", title="Test Conversation")
    conv.add_message(MessageRole.USER, "Hello, how can you help?")
    conv.add_message(MessageRole.ASSISTANT, "I can help with coding tasks.")
    conv.add_message(MessageRole.USER, "Can you write a function?")
    return conv


def test_message_creation():
    """Test creating a message."""
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.timestamp is not None


def test_message_to_dict():
    """Test converting message to dictionary."""
    msg = Message(role=MessageRole.ASSISTANT, content="Response", metadata={"key": "value"})
    msg_dict = msg.to_dict()

    assert msg_dict["role"] == "assistant"
    assert msg_dict["content"] == "Response"
    assert msg_dict["metadata"] == {"key": "value"}
    assert "timestamp" in msg_dict


def test_message_with_tool_calls():
    """Test message with tool calls and results."""
    tool_calls = [{"tool": "read_file", "args": {"path": "test.py"}}]
    tool_results = [{"status": "success", "result": "file contents"}]

    msg = Message(
        role=MessageRole.ASSISTANT,
        content="Using tools...",
        tool_calls=tool_calls,
        tool_results=tool_results,
    )

    assert msg.tool_calls == tool_calls
    assert msg.tool_results == tool_results


def test_conversation_creation():
    """Test creating a conversation."""
    conv = Conversation(conversation_id="conv_1", title="My Conversation")
    assert conv.conversation_id == "conv_1"
    assert conv.title == "My Conversation"
    assert len(conv.messages) == 0
    assert conv.created_at is not None


def test_conversation_add_message():
    """Test adding messages to conversation."""
    conv = Conversation(conversation_id="conv_1")
    msg1 = conv.add_message(MessageRole.USER, "First message")
    msg2 = conv.add_message(MessageRole.ASSISTANT, "Response")

    assert len(conv.messages) == 2
    assert conv.messages[0] == msg1
    assert conv.messages[1] == msg2
    assert conv.messages[0].role == MessageRole.USER
    assert conv.messages[1].role == MessageRole.ASSISTANT


def test_conversation_get_messages():
    """Test retrieving messages from conversation."""
    conv = Conversation(conversation_id="conv_1")
    conv.add_message(MessageRole.USER, "Q1")
    conv.add_message(MessageRole.ASSISTANT, "A1")
    conv.add_message(MessageRole.USER, "Q2")
    conv.add_message(MessageRole.ASSISTANT, "A2")

    # Get all messages
    all_msgs = conv.get_messages()
    assert len(all_msgs) == 4

    # Get limited messages
    limited = conv.get_messages(limit=2)
    assert len(limited) == 2
    assert limited[0].content == "Q2"  # Most recent

    # Filter by role
    user_msgs = conv.get_messages(role_filter=MessageRole.USER)
    assert len(user_msgs) == 2
    assert all(m.role == MessageRole.USER for m in user_msgs)


def test_conversation_context_window():
    """Test getting context window for LLM."""
    conv = Conversation(conversation_id="conv_1")
    conv.add_message(MessageRole.USER, "Message 1")
    conv.add_message(MessageRole.ASSISTANT, "Response 1")
    conv.add_message(MessageRole.USER, "Message 2")

    context = conv.get_context_window(max_messages=2)
    assert len(context) == 2
    assert context[0]["content"] == "Response 1"
    assert context[1]["content"] == "Message 2"


def test_conversation_compute_stats():
    """Test computing conversation statistics."""
    conv = Conversation(conversation_id="conv_1")
    conv.add_message(MessageRole.USER, "Hello")
    conv.add_message(MessageRole.ASSISTANT, "Hi there")
    conv.add_message(MessageRole.USER, "How are you?")

    stats = conv.compute_stats()

    assert stats.total_messages == 3
    assert stats.total_user_turns == 2
    assert stats.total_assistant_turns == 1
    assert stats.avg_message_length > 0


def test_conversation_clear():
    """Test clearing conversation messages."""
    conv = Conversation(conversation_id="conv_1")
    conv.add_message(MessageRole.USER, "Message")
    assert len(conv.messages) == 1

    conv.clear()
    assert len(conv.messages) == 0


def test_conversation_to_dict():
    """Test converting conversation to dictionary."""
    conv = Conversation(conversation_id="conv_1", title="Test")
    conv.add_message(MessageRole.USER, "Hello")
    conv.add_message(MessageRole.ASSISTANT, "Hi")

    conv_dict = conv.to_dict()

    assert conv_dict["conversation_id"] == "conv_1"
    assert conv_dict["title"] == "Test"
    assert len(conv_dict["messages"]) == 2
    assert "stats" in conv_dict


def test_conversation_store_create():
    """Test creating conversations in store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)

        conv1 = store.create_conversation(conversation_id="conv_1", title="First")
        conv2 = store.create_conversation(title="Second")  # Auto-generated ID

        assert conv1.conversation_id == "conv_1"
        assert conv2.conversation_id is not None
        assert len(conv2.conversation_id) > 0


def test_conversation_store_duplicate_id():
    """Test that duplicate conversation IDs are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)

        store.create_conversation(conversation_id="conv_1")
        with pytest.raises(ValueError, match="already exists"):
            store.create_conversation(conversation_id="conv_1")


def test_conversation_store_get():
    """Test retrieving conversations from store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)

        conv = store.create_conversation(conversation_id="conv_1")
        conv.add_message(MessageRole.USER, "Test")

        retrieved = store.get_conversation("conv_1")
        assert retrieved is not None
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Test"


def test_conversation_store_delete():
    """Test deleting conversations from store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)

        store.create_conversation(conversation_id="conv_1")
        assert store.get_conversation("conv_1") is not None

        deleted = store.delete_conversation("conv_1")
        assert deleted is True
        assert store.get_conversation("conv_1") is None

        # Deleting non-existent should return False
        deleted = store.delete_conversation("conv_999")
        assert deleted is False


def test_conversation_store_list():
    """Test listing conversations in store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)

        conv1 = store.create_conversation(conversation_id="conv_1", title="First")
        conv1.add_message(MessageRole.USER, "Hi")
        conv2 = store.create_conversation(conversation_id="conv_2", title="Second")

        conversations = store.list_conversations()
        assert len(conversations) == 2
        assert any(c["conversation_id"] == "conv_1" for c in conversations)
        assert any(c["message_count"] == 1 for c in conversations)


def test_conversation_store_save_load():
    """Test saving and loading conversations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)

        conv = store.create_conversation(conversation_id="conv_1")
        conv.add_message(MessageRole.USER, "Hello")
        conv.add_message(MessageRole.ASSISTANT, "Hi there")

        # Save
        saved = store.save_conversation("conv_1")
        assert saved is True

        # Clear memory and load
        store._conversations.clear()
        loaded = store.load_conversation("conv_1")

        assert loaded is not None
        assert loaded.conversation_id == "conv_1"
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Hello"


def test_conversation_manager_start():
    """Test starting a conversation with manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)
        manager = ConversationManager(store)

        conv = manager.start_conversation(conversation_id="conv_1", title="Test")

        assert conv is not None
        assert conv.conversation_id == "conv_1"
        assert manager.get_active_conversation() == conv


def test_conversation_manager_add_messages():
    """Test adding messages via manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)
        manager = ConversationManager(store)

        manager.start_conversation(conversation_id="conv_1")
        manager.add_user_message("What is Python?")
        manager.add_assistant_message("Python is a programming language.")

        conv = manager.get_active_conversation()
        assert len(conv.messages) == 2
        assert conv.messages[0].role == MessageRole.USER


def test_conversation_manager_switch():
    """Test switching between conversations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)
        manager = ConversationManager(store)

        conv1 = manager.start_conversation(conversation_id="conv_1")
        conv1_msg = manager.add_user_message("Conv 1 message")

        conv2 = manager.start_conversation(conversation_id="conv_2")
        conv2_msg = manager.add_user_message("Conv 2 message")

        # Switch back to conv1
        manager.switch_conversation("conv_1")
        active = manager.get_active_conversation()

        assert active.conversation_id == "conv_1"
        assert len(active.messages) == 1


def test_conversation_manager_context_window():
    """Test getting context window from manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)
        manager = ConversationManager(store)

        manager.start_conversation()
        manager.add_user_message("Question 1")
        manager.add_assistant_message("Answer 1")
        manager.add_user_message("Question 2")

        context = manager.get_context_window(max_messages=2)
        assert len(context) == 2


def test_conversation_manager_summary():
    """Test getting conversation summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore(tmpdir)
        manager = ConversationManager(store)

        manager.start_conversation(conversation_id="conv_1", title="Test Conv")
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        summary = manager.get_conversation_summary()

        assert summary["conversation_id"] == "conv_1"
        assert summary["title"] == "Test Conv"
        assert summary["total_messages"] == 2
        assert summary["user_turns"] == 1
        assert summary["assistant_turns"] == 1


def test_conversation_summary_for_rolling_context():
    """Older turns should be condensed into a summary while recent turns stay visible."""
    conv = Conversation(conversation_id="conv_summary")
    conv.add_message(MessageRole.USER, "Help me build a Python API client.")
    conv.add_message(MessageRole.ASSISTANT, "I will inspect the existing client and draft the wrapper.")
    conv.add_message(MessageRole.USER, "Can you explain the auth flow?")
    conv.add_message(MessageRole.ASSISTANT, "The flow uses a token refresh mechanism.")
    conv.add_message(MessageRole.USER, "What edge cases remain?")

    context = conv.get_context_window(max_messages=2, include_summary=True)

    assert context[0]["role"] == "system"
    assert "Python API client" in context[0]["content"]
    assert context[-1]["content"] == "What edge cases remain?"
