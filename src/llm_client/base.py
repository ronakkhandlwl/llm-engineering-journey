"""Provider-agnostic types and the LLMProvider interface.

Every backend (OpenAI, Anthropic, Gemini) speaks the same small vocabulary:
a list of immutable Messages in, a ChatResult (text + token usage) out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Conversation roles shared across providers. Each adapter maps these to its
# own wire format (e.g. Gemini uses "model" instead of "assistant").
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """A single immutable turn in the conversation."""

    role: str
    content: str


@dataclass(frozen=True)
class ChatResult:
    """The model's reply plus the token usage reported for this call."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


@runtime_checkable
class LLMProvider(Protocol):
    """Swappable backend. Implementations are stateless across calls except
    for the underlying SDK client they hold."""

    name: str
    model: str

    def chat(self, messages: list[Message], system: str | None = None) -> ChatResult:
        """Send the full conversation history and return the next reply."""
        ...
