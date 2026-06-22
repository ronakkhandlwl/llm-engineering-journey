"""Map a provider name to a concrete LLMProvider instance.

Adding a backend = one entry here. Callers never import the concrete classes.
"""

from __future__ import annotations

from llm_client.base import LLMProvider

PROVIDERS = ("openai", "anthropic", "gemini")


def get_provider(name: str) -> LLMProvider:
    """Construct the provider for ``name``. Raises ValueError if unknown.

    Imports are local so picking one provider never requires the others'
    SDK clients (or API keys) to be present.
    """
    key = name.lower()
    if key == "openai":
        from llm_client.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if key == "anthropic":
        from llm_client.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if key == "gemini":
        from llm_client.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"Unknown provider {name!r}. Choose from: {', '.join(PROVIDERS)}.")
