"""Travel tools registered in the ToolRegistry.

Data sources (all free, no API keys):
- Weather: Open-Meteo
- Geocoding: Open-Meteo
- Distances / travel time: OSRM public router (driving/walking/cycling)
- Currency: open.er-api.com
- Knowledge (destinations, attractions, visa info): Wikipedia
- Flights / hotels: deterministic synthetic estimates (no honest free
  no-key API exists); clearly labeled in their output and swappable for
  real APIs later without touching the agents.
"""
from __future__ import annotations

import hashlib
import math
import random

import httpx
from pydantic import BaseModel, Field

from orchestrator import config
from orchestrator.registry import registry
from orchestrator.schemas import AgentType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEO_CACHE: dict[str, tuple[float, float, str]] = {}


def _geocode(place: str) -> tuple[float, float, str]:
    """Resolve a place name to (lat, lon, display_name) via Open-Meteo."""
    key = place.strip().lower()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Place not found: {place}")
    r = results[0]
    name = r.get("name", place)
    country = r.get("country", "")
    display = f"{name}, {country}" if country else name
    _GEO_CACHE[key] = (r["latitude"], r["longitude"], display)
    return _GEO_CACHE[key]


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _seeded(*parts) -> random.Random:
    """Deterministic RNG from stable parts (so demos are reproducible)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Lisbon'")
    days: int = Field(default=3, ge=1, le=7, description="Forecast days")


@registry.register(
    name="get_weather",
    description="Get the daily forecast (temperature and rain) for a city.",
    input_model=WeatherInput,
    allowed_agents=[
        AgentType.TRAVEL_INFO,
        AgentType.ITINERARY,
        AgentType.DESTINATION,
    ],
)
def get_weather(city: str, days: int = 3) -> str:
    place = _geocode(city)
    forecast = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place[0],
            "longitude": place[1],
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
    return f"Forecast for {place[2]}:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

_RATES_CACHE: dict[str, dict[str, float]] = {}


class ConvertCurrencyInput(BaseModel):
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="ISO 4217 code, e.g. 'USD'")
    to_currency: str = Field(description="ISO 4217 code, e.g. 'INR'")


@registry.register(
    name="convert_currency",
    description="Convert an amount between two ISO currency codes.",
    input_model=ConvertCurrencyInput,
    allowed_agents=[AgentType.BUDGET, AgentType.ITINERARY, AgentType.SUPERVISOR],
)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    from_currency, to_currency = from_currency.upper(), to_currency.upper()
    if from_currency not in _RATES_CACHE:
        resp = httpx.get(
            f"https://open.er-api.com/v6/latest/{from_currency}", timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") != "success":
            raise ValueError(f"Currency lookup failed for {from_currency}")
        _RATES_CACHE[from_currency] = data["rates"]
    rates = _RATES_CACHE[from_currency]
    if to_currency not in rates:
        raise ValueError(f"Unknown currency: {to_currency}")
    converted = amount * rates[to_currency]
    return (
        f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency} "
        f"(rate {rates[to_currency]:.4f})"
    )


# ---------------------------------------------------------------------------
# Travel time (OSRM public router)
# ---------------------------------------------------------------------------


class TravelTimeInput(BaseModel):
    from_city: str = Field(description="Origin city, e.g. 'Tokyo'")
    to_city: str = Field(description="Destination city, e.g. 'Kyoto'")
    mode: str = Field(
        default="driving", description="driving / walking / cycling"
    )


@registry.register(
    name="calc_travel_time",
    description=(
        "Estimate road distance and travel time between two cities using "
        "the OSRM router. Use 'driving' for intercity estimates."
    ),
    input_model=TravelTimeInput,
    allowed_agents=[
        AgentType.TRANSPORTATION,
        AgentType.ITINERARY,
        AgentType.DESTINATION,
    ],
)
def calc_travel_time(from_city: str, to_city: str, mode: str = "driving") -> str:
    if mode not in ("driving", "walking", "cycling"):
        raise ValueError("mode must be driving, walking or cycling")
    origin = _geocode(from_city)
    destination = _geocode(to_city)
    profile = mode
    resp = httpx.get(
        f"https://router.project-osrm.org/route/v1/{profile}/"
        f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}",
        params={"overview": "false"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"No route found between {from_city} and {to_city}")
    route = data["routes"][0]
    km = route["distance"] / 1000
    hours = route["duration"] / 3600
    hours_s = f"{int(hours)}h {round((hours - int(hours)) * 60)}m"
    return f"{origin[2]} -> {destination[2]} ({mode}): {km:,.0f} km, ~{hours_s}"


# ---------------------------------------------------------------------------
# Wikipedia knowledge search
# ---------------------------------------------------------------------------


class WikipediaSearchInput(BaseModel):
    query: str = Field(description="Search query, e.g. 'Fushimi Inari Shrine'")
    max_results: int = Field(default=3, ge=1, le=5)


def _wikipedia_search(query: str, limit: int) -> list[dict]:
    """Return [{title, extract}] for a Wikipedia search."""
    search = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
        timeout=10,
    )
    search.raise_for_status()
    titles = [
        h["title"]
        for h in search.json().get("query", {}).get("search", [])
    ]
    if not titles:
        return []
    extracts = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "titles": "|".join(titles),
            "format": "json",
        },
        timeout=10,
    )
    extracts.raise_for_status()
    pages = extracts.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        text = page.get("extract", "").strip()
        if text:
            out.append({"title": page["title"], "extract": text[:900]})
    return out


@registry.register(
    name="search_wikipedia",
    description=(
        "Search English Wikipedia. Returns titles with short extracts. "
        "Use for facts, background, and destination knowledge."
    ),
    input_model=WikipediaSearchInput,
    allowed_agents=[
        AgentType.DESTINATION,
        AgentType.TRAVEL_INFO,
        AgentType.TRANSPORTATION,
        AgentType.SUPERVISOR,
    ],
)
def search_wikipedia(query: str, max_results: int = 3) -> str:
    results = _wikipedia_search(query, max_results)
    if not results:
        return f"No Wikipedia results for '{query}'. Try a simpler query."
    blocks = [f"## {r['title']}\n{r['extract']}" for r in results]
    return "\n\n".join(blocks)


class AttractionsInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Kyoto'")


@registry.register(
    name="search_attractions",
    description=(
        "Find major tourist attractions and landmarks in a city using "
        "Wikipedia. Use this to pick activities for the itinerary."
    ),
    input_model=AttractionsInput,
    allowed_agents=[AgentType.DESTINATION, AgentType.ITINERARY],
)
def search_attractions(city: str) -> str:
    results = _wikipedia_search(f"{city} tourist attractions", 4)
    if not results:
        results = _wikipedia_search(f"{city} landmarks", 4)
    if not results:
        return f"No attraction pages found for '{city}'."
    blocks = [f"## {r['title']}\n{r['extract'][:600]}" for r in results]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Visa / entry info
# ---------------------------------------------------------------------------


class VisaInfoInput(BaseModel):
    passport_country: str = Field(description="e.g. 'India'")
    destination_country: str = Field(description="e.g. 'Japan'")


@registry.register(
    name="get_visa_info",
    description=(
        "Look up visa/entry requirements for a nationality entering a "
        "country, from Wikipedia's visa-requirements pages."
    ),
    input_model=VisaInfoInput,
    allowed_agents=[AgentType.TRAVEL_INFO],
)
def get_visa_info(passport_country: str, destination_country: str) -> str:
    results = _wikipedia_search(
        f"Visa requirements for {passport_country} citizens {destination_country}",
        2,
    )
    if not results:
        results = _wikipedia_search(f"{destination_country} visa", 2)
    if not results:
        return (
            f"Could not find reliable visa information for "
            f"{passport_country} passport -> {destination_country}. "
            "Verify with the embassy/consulate before the trip."
        )
    blocks = [f"## {r['title']}\n{r['extract'][:800]}" for r in results]
    return (
        "Visa/entry reference (verify with official sources for the "
        "current rules):\n\n" + "\n\n".join(blocks)
    )


# ---------------------------------------------------------------------------
# Flight estimates (synthetic, deterministic)
# ---------------------------------------------------------------------------

_AIRLINES = [
    "Indigo", "Vistara", "Air India", "Akasa Air", "SpiceJet", "Jet Airways",
]


class FlightEstimateInput(BaseModel):
    origin: str = Field(description="Origin city, e.g. 'Delhi'")
    destination: str = Field(description="Destination city, e.g. 'Tokyo'")
    date: str = Field(description="Departure date, e.g. '2026-10-10'")
    pax: int = Field(default=1, ge=1, le=9)
    cabin: str = Field(default="economy", description="economy / premium")


@registry.register(
    name="estimate_flights",
    description=(
        "Estimate flight options between two cities. SYNTHETIC estimates "
        "derived from the real distance between cities - prices are "
        "plausible, not bookable quotes."
    ),
    input_model=FlightEstimateInput,
    allowed_agents=[AgentType.TRANSPORTATION, AgentType.BUDGET],
)
def estimate_flights(
    origin: str, destination: str, date: str, pax: int = 1, cabin: str = "economy"
) -> str:
    try:
        o = _geocode(origin)
        d = _geocode(destination)
        km = _haversine_km(o[0], o[1], d[0], d[1])
    except ValueError:
        km, o, d = 1500.0, (0, 0, origin), (0, 0, destination)
        note = f"(unknown geocodes, default distance {km:.0f} km)\n"
    else:
        note = ""

    rnd = _seeded(origin.lower(), destination.lower(), date, pax, cabin)
    cabin_mult = 2.1 if cabin == "premium" else 1.0
    base = km * 4.2 * cabin_mult + 4200 * cabin_mult  # rough INR per pax
    lines = [note] if note else []
    for i in range(3):
        airline = rnd.choice(_AIRLINES)
        price = round(base * rnd.uniform(0.82, 1.25) / 50) * 50
        dep_h = rnd.randint(5, 22)
        dur_h = max(1.5, km / 800 + rnd.uniform(0.5, 1.5))
        lines.append(
            f"{i + 1}. {airline} {origin} -> {destination}, {date}\n"
            f"   dep {dep_h:02d}:00 local, ~{dur_h:.0f}h, {cabin}\n"
            f"   est. {price:,} INR / pax (x{pax} = {price * pax:,} INR)"
        )
    lines.insert(0, "SYNTHETIC flight estimates (not bookable):\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hotel estimates (synthetic, deterministic)
# ---------------------------------------------------------------------------


class HotelEstimateInput(BaseModel):
    city: str = Field(description="City, e.g. 'Kyoto'")
    nights: int = Field(default=2, ge=1, le=30)
    pax: int = Field(default=2, ge=1, le=8)


@registry.register(
    name="estimate_hotels",
    description=(
        "Estimate hotel options for a city (budget / mid-range / upscale). "
        "SYNTHETIC estimates - plausible prices, not live availability."
    ),
    input_model=HotelEstimateInput,
    allowed_agents=[AgentType.ACCOMMODATION, AgentType.BUDGET],
)
def estimate_hotels(city: str, nights: int = 2, pax: int = 2) -> str:
    _geocode(city)  # fail early if the city is unknown
    rnd = _seeded(city.lower(), nights, pax)
    bands = [
        ("budget", 1100, 2300, "guesthouse / 3-star"),
        ("mid-range", 3000, 5500, "4-star / quality boutique"),
        ("upscale", 7000, 14000, "4-5 star / ryokan"),
    ]
    names = [
        f"{city} {n}"
        for n in ("Garden Inn", "Riverside Stay", "Heritage House",
                  "Central Boutique", "Hill View Hotel", "Lantern Suites")
    ]
    picks = rnd.sample(names, 3)
    lines = ["SYNTHETIC hotel estimates for " + city + f" ({nights} nights, {pax} guests):\n"]
    for (band, lo, hi, label), name in zip(bands, picks):
        per_night = round(rnd.uniform(lo, hi) / 100) * 100
        lines.append(
            f"- {name} [{band}, {label}]: ~{per_night:,} INR/night, "
            f"total ~{per_night * nights:,} INR for {nights} night(s)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class WriteFileInput(BaseModel):
    filename: str = Field(description="File name only, e.g. 'japan_trip.md'")
    content: str = Field(description="Text to save")


@registry.register(
    name="write_file",
    description="Save text to a file in the outputs folder.",
    input_model=WriteFileInput,
    allowed_agents=[AgentType.SUPERVISOR],
)
def write_file(filename: str, content: str) -> str:
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = (output_dir / filename).resolve()
    if path.parent != output_dir.resolve():
        raise ValueError("Filename must not include folders")
    path.write_text(content, encoding="utf-8")
    return f"Saved {path.name} ({len(content)} chars)"
