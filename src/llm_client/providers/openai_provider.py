"""OpenAI backend."""

from __future__ import annotations

from openai import OpenAI

from llm_client.base import ChatResult, Message

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024


class OpenAIProvider:
    name = "openai"

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # Reads OPENAI_API_KEY from the environment.
        self._client = OpenAI()

    def chat(self, messages: list[Message], system: str | None = None) -> ChatResult:
        wire_messages: list[dict] = []
        if system:
            wire_messages.append({"role": "system", "content": system})
        wire_messages.extend({"role": m.role, "content": m.content} for m in messages)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=wire_messages,
        )
        usage = response.usage
        return ChatResult(
            text=response.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
