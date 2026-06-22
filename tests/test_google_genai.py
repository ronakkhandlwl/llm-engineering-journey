import pytest
from unittest.mock import MagicMock
from utils.weather_info import get_current_temperature
from validations.google_genai_verify import WeatherResponse

# --- Test 1: JSON output has the right keys ---


@pytest.mark.unit
def test_weather_response_has_required_keys():
    raw = get_current_temperature("Tokyo")
    validated = WeatherResponse(**raw)
    dumped = validated.model_dump()

    assert "temperature" in dumped
    assert dumped["temperature"] == 20


# --- Test 2: Function calling logic routes correctly ---


@pytest.mark.unit
def test_function_call_logic_invokes_tool_and_validates():
    mock_function_call = MagicMock()
    mock_function_call.name = "get_current_temperature"
    mock_function_call.args = {"location": "Mumbai"}

    raw_result = get_current_temperature(**mock_function_call.args)
    validated = WeatherResponse(**raw_result)

    assert validated.temperature == 25
    assert isinstance(validated.model_dump(), dict)
    assert "temperature" in validated.model_dump()
