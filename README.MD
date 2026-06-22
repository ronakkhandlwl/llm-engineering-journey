# llm-engineering-journey

A Python project for learning LLM API engineering.

## Week 1

Covers calling OpenAI and Anthropic APIs, getting structured JSON outputs, and basic function calling.

## Week 2 — Multi-provider chat

A multi-turn chat REPL with a **swappable backend**. The same interface
(`LLMProvider.chat(messages) -> ChatResult`) is implemented for OpenAI,
Anthropic, and Gemini, so you switch backends with a single CLI flag.

### Features

- **Multi-turn loop** — conversation history is kept in a list of immutable
  `Message`s and resent on every turn, so the model always sees full context.
- **`--provider` flag** — `openai`, `anthropic`, or `gemini`; same code path,
  different backend.
- **Per-call token usage + cost** — input tokens, output tokens, and an
  estimated USD cost are printed after each reply, plus a running session total.

### Setup

```bash
poetry install
cp .env.example .env   # then fill in the keys for the provider(s) you use
```

Required keys (only the one for your chosen provider is needed):

| Provider  | Env var             |
|-----------|---------------------|
| openai    | `OPENAI_API_KEY`    |
| anthropic | `ANTHROPIC_API_KEY` |
| gemini    | `GEMINI_API_KEY`    |

### Run

```bash
# via the installed console script
chat --provider anthropic
chat --provider openai
chat --provider gemini

# or without installing
PYTHONPATH=src python -m llm_client.chat --provider gemini
```

Type `exit`/`quit` or press Ctrl-D to leave. Example turn:

```
you> what's the capital of France?

anthropic> Paris.
[usage] in=18 out=5 est_cost=$0.000129 | session_total=$0.000129
```

### Cost estimates

Costs come from a per-model price table in
[src/llm_client/pricing.py](src/llm_client/pricing.py) (USD per 1M tokens).
List prices change over time — treat the numbers as estimates and verify
against each provider's pricing page.

### Architecture

```
src/llm_client/
├── base.py          # Message, ChatResult, LLMProvider protocol
├── pricing.py       # price table + estimate_cost()
├── factory.py       # get_provider("openai" | "anthropic" | "gemini")
├── chat.py          # multi-turn REPL + --provider CLI
└── providers/       # one adapter per backend, all speaking the same interface
```

Adding a backend means writing one adapter and adding one line to
[factory.py](src/llm_client/factory.py) — nothing else changes.

### Tests

```bash
poetry run pytest
```

Provider adapters are exercised without network calls (the loop is tested with a
fake in-memory provider); pricing and the factory are pure unit tests.
