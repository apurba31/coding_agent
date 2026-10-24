# Prompting

`PromptBuilder` formats `<repository_context>` blocks with file path, language, symbol, line range, and fenced code.

Token budget (`MINI_AGENT_TOKEN_BUDGET`) caps retrieved chunks; highest-ranked chunks win deterministically.

Agent system prompts live in `coding_agent/app.py` and expand when write/terminal tools are enabled.
