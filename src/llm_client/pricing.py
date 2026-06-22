"""Token-usage cost estimation.

Prices are USD per 1,000,000 tokens as (input_rate, output_rate). These are
published list prices and change over time — treat the output as an *estimate*,
not a billing source of truth. Verify against each provider's pricing page:

- Anthropic: https://www.anthropic.com/pricing
- OpenAI:    https://openai.com/api/pricing
- Gemini:    https://ai.google.dev/gemini-api/docs/pricing
"""

from __future__ import annotations

_PER_MILLION = 1_000_000

# model id -> (input USD / 1M tokens, output USD / 1M tokens)
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-4-8": (15.0, 75.0),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    # Gemini
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one call. Unknown models return 0.0."""
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / _PER_MILLION
