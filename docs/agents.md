# Agents

## Single agent

`LangGraphAgent` compiles a small graph:

```text
START → analyze_question → (direct | retrieve) → prepare_agent
      → call_model → (execute_tools → call_model)* → finish → END
```

Each model call and tool execution is a separate graph node. The executor provides
single-step model and tool operations, while LangGraph owns iteration and routing.

Graph state is checkpointed to SQLite at `checkpoints.sqlite` alongside conversation
files (or `.mini-agent/checkpoints.sqlite` when conversation persistence is disabled).
Pass the conversation ID as the graph thread ID to resume an interrupted execution:

```python
result = agent.resume(conversation_id)
```

The checkpointer connection is released with `agent.close()` or by using the agent as a
context manager.

Checkpoints are committed when a graph node finishes. If the process stops while a tool
node is running, resuming can retry that node; tools with external side effects should
therefore be safe to retry.

`LangGraphAgent.execute(..., on_token=callback)` forwards text deltas to the callback.
The interactive CLI uses this hook for streamed terminal output.

## Dynamic multi-agent delegation

`mini-agent chat --multi-agent` uses deterministic rules to split explicit lists,
semicolon-separated requests, and distinct question clauses into research tasks. Up to
five isolated researcher agents run concurrently; a synthesizer with an empty tool
registry combines their findings. Research workers remain read-only. Simple requests
stay a single research task.

The coordinator accepts a researcher factory for parallel execution. If multiple tasks
are found but no factory is supplied, it runs them sequentially on the provided researcher.

## Conversation history

`ConversationStore` persists turns under `.mini-agent/conversations/`. Resume with `--conversation-id`.
