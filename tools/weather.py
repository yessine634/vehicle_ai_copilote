import requests


WMO_WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather_description(weather_code: int) -> str:
    """Translate an Open-Meteo WMO weather code into plain English."""

    return WMO_WEATHER_DESCRIPTIONS.get(weather_code, "Unknown weather condition")


def get_weather(city: str) -> dict:
    """
    Get the current weather for a city using Open-Meteo.

    The function:
    1. Converts the city name to latitude/longitude.
    2. Retrieves current weather using those coordinates.
    3. Returns clean structured data.
    """

    # -------------------------
    # 1. Geocoding
    # -------------------------

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    geo_response = requests.get(
        geocoding_url,
        params=geo_params,
        timeout=10,
    )

    geo_response.raise_for_status()

    geo_data = geo_response.json()

    if not geo_data.get("results"):
        return {
            "error": f"Location '{city}' not found."
        }

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    # -------------------------
    # 2. Weather
    # -------------------------

    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "weather_code",
            "wind_speed_10m",
        ],
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=10,
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    current = weather_data["current"]

    # -------------------------
    # 3. Clean result
    # -------------------------

    weather_code = int(current["weather_code"])

    return {
        "city": location["name"],
        "country": location.get("country"),
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": current["temperature_2m"],
        "humidity_percent": current["relative_humidity_2m"],
        "precipitation_mm": current["precipitation"],
        "rain_mm": current["rain"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "weather_code": weather_code,
        "weather_description": get_weather_description(weather_code),
    }

