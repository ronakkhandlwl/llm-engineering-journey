import pytest

from llm_client.pricing import PRICING, estimate_cost


@pytest.mark.unit
def test_known_model_cost_is_input_plus_output():
    # gpt-4o-mini: 0.15 in / 0.60 out per 1M tokens
    # 1_000_000 in -> 0.15, 2_000_000 out -> 1.20  => 1.35
    cost = estimate_cost("gpt-4o-mini", 1_000_000, 2_000_000)
    assert cost == pytest.approx(1.35)


@pytest.mark.unit
def test_zero_tokens_is_zero_cost():
    assert estimate_cost("claude-sonnet-4-6", 0, 0) == 0.0


@pytest.mark.unit
def test_unknown_model_returns_zero():
    assert estimate_cost("does-not-exist", 500, 500) == 0.0


@pytest.mark.unit
def test_every_pricing_entry_is_input_lte_output():
    # Output tokens are never cheaper than input across these providers.
    for model, (in_rate, out_rate) in PRICING.items():
        assert in_rate <= out_rate, model
