"""Google Gemini backend."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from llm_client.base import ROLE_ASSISTANT, ChatResult, Message

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        # google-genai requires the key explicitly; raise early if missing.
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def chat(self, messages: list[Message], system: str | None = None) -> ChatResult:
        # Gemini calls the model's role "model", not "assistant".
        contents = [
            types.Content(
                role="model" if m.role == ROLE_ASSISTANT else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]
        config = (
            types.GenerateContentConfig(system_instruction=system) if system else None
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        usage = response.usage_metadata
        return ChatResult(
            text=response.text or "",
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
