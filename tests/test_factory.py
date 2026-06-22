import pytest

from llm_client.factory import PROVIDERS, get_provider


@pytest.mark.unit
def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("llama")


@pytest.mark.unit
def test_providers_tuple_is_the_supported_set():
    assert set(PROVIDERS) == {"openai", "anthropic", "gemini"}


@pytest.mark.unit
def test_get_provider_is_case_insensitive(monkeypatch):
    # Avoid constructing a real SDK client / needing keys.
    import llm_client.providers.anthropic_provider as ap

    monkeypatch.setattr(ap.AnthropicProvider, "__init__", lambda self: None)
    provider = get_provider("Anthropic")
    assert isinstance(provider, ap.AnthropicProvider)
