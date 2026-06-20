from pydantic import BaseModel


class WeatherResponse(BaseModel):
    """Validate get_current_temperature method response"""

    temperature: int | str
