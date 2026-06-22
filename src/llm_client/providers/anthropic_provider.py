"""Anthropic (Claude) backend."""

from __future__ import annotations

import anthropic

from llm_client.base import ChatResult, Message

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # Reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic()

    def chat(self, messages: list[Message], system: str | None = None) -> ChatResult:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        return ChatResult(
            text=text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
