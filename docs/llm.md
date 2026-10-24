# LLM

Production chat uses `GroqClient` (`GROQ_API_KEY` in `.env`).

The client abstraction supports tool schemas and usage/token metrics recorded in `MetricsCollector`.

Additional providers can implement the same interface under `coding_agent/llm/providers/`.
