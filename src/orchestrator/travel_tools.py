from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from orchestrator.registry import registry
from orchestrator.schemas import AgentType

OUTPUT_DIR = Path("outputs")


class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Lisbon'")
    days: int = Field(default=3, ge=1, le=7, description="Forecast days")


@registry.register(
    name="get_weather",
    description="Get the daily forecast (temperature and rain) for a city.",
    input_model=WeatherInput,
    allowed_agents=[AgentType.RESEARCH, AgentType.PLANNER],
)
def get_weather(city: str, days: int = 3) -> str:
    geo = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    )
    geo.raise_for_status()
    results = geo.json().get("results")
    if not results:
        raise ValueError(f"City not found: {city}")
    place = results[0]

    forecast = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": days,
            "timezone": "auto",
        },
        timeout=10,
    )
    forecast.raise_for_status()
    daily = forecast.json()["daily"]

    lines = [
        f"{d}: {lo}-{hi} C, rain {rain} mm"
        for d, lo, hi, rain in zip(
            daily["time"],
            daily["temperature_2m_min"],
            daily["temperature_2m_max"],
            daily["precipitation_sum"],
        )
    ]
    return f"Forecast for {place['name']}, {place.get('country', '')}:\n" + "\n".join(lines)


class WriteFileInput(BaseModel):
    filename: str = Field(description="File name only, e.g. 'lisbon_trip.md'")
    content: str = Field(description="Text to save")


@registry.register(
    name="write_file",
    description="Save text to a file in the outputs folder.",
    input_model=WriteFileInput,
    allowed_agents=[AgentType.WRITER],
)
def write_file(filename: str, content: str) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve():
        raise ValueError("Filename must not include folders")
    path.write_text(content, encoding="utf-8")
    return f"Saved {path.name} ({len(content)} chars)"