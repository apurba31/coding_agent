# Memory

`ConversationStore` persists JSON conversations; `ConversationManager` handles active session selection and rolling history limits inside the executor.

Summarization hooks are reserved for when history exceeds configured limits (see `memory/store.py`).

Resume chats with `mini-agent chat --conversation-id <id>`.
