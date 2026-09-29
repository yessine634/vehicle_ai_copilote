"""Routing tool for the Agentic AI Vehicle Copilot.

This module:
1. Geocodes origin and destination names with Open-Meteo.
2. Requests a car route from the public Valhalla demo server.
3. Returns only the routing information useful to the AI agent.
"""

from __future__ import annotations

from typing import Any

import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
VALHALLA_ROUTE_URL = "https://valhalla1.openstreetmap.de/route"

REQUEST_TIMEOUT = 20


def _geocode_location(location_name: str) -> dict[str, Any]:
    """Convert a place name into coordinates using Open-Meteo Geocoding."""

    candidates = [location_name.strip()]
    short_name = location_name.split(",", maxsplit=1)[0].strip()
    if short_name and short_name not in candidates:
        candidates.append(short_name)

    results = None
    for candidate in candidates:
        params = {
            "name": candidate,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        try:
            response = requests.get(
                GEOCODING_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Geocoding request failed for '{location_name}': {exc}"
            ) from exc

        data = response.json()
        results = data.get("results")
        if results:
            break

    if not results:
        raise ValueError(f"Location '{location_name}' was not found.")

    place = results[0]

    return {
        "name": place["name"],
        "country": place.get("country"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
    }


def get_route(origin: str, destination: str) -> dict[str, Any]:
    """Get a driving route between two named locations.

    Parameters
    ----------
    origin:
        Starting city/place name, for example "Sfax".
    destination:
        Destination city/place name, for example "Tunis".

    Returns
    -------
    dict
        Clean route information containing:
        - resolved origin/destination
        - coordinates
        - distance in kilometers
        - estimated duration in minutes
        - whether the route contains highways, toll roads, or ferries
    """

    origin_location = _geocode_location(origin)
    destination_location = _geocode_location(destination)

    payload = {
        "locations": [
            {
                "lat": origin_location["latitude"],
                "lon": origin_location["longitude"],
            },
            {
                "lat": destination_location["latitude"],
                "lon": destination_location["longitude"],
            },
        ],
        "costing": "auto",
        "units": "kilometers",
    }

    headers = {
        "Content-Type": "application/json",
        "X-Client-Id": "vehicle-ai-copilot",
    }

    try:
        response = requests.post(
            VALHALLA_ROUTE_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Valhalla routing request failed: {exc}") from exc

    data = response.json()

    try:
        summary = data["trip"]["summary"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Valhalla returned an unexpected response format."
        ) from exc

    return {
        "origin": {
            "name": origin_location["name"],
            "country": origin_location["country"],
            "latitude": origin_location["latitude"],
            "longitude": origin_location["longitude"],
        },
        "destination": {
            "name": destination_location["name"],
            "country": destination_location["country"],
            "latitude": destination_location["latitude"],
            "longitude": destination_location["longitude"],
        },
        "distance_km": round(float(summary["length"]), 2),
        "duration_minutes": round(float(summary["time"]) / 60, 1),
        "uses_highway": bool(summary.get("has_highway", False)),
        "has_toll": bool(summary.get("has_toll", False)),
        "has_ferry": bool(summary.get("has_ferry", False)),
    }


if __name__ == "__main__":
    route = get_route("Sfax", "Tunis")
    print(route)
