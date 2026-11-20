"""Routing tool for the Agentic AI Vehicle Copilot.

This module:
1. Resolves origin and destination names with validated OSM/Nominatim search.
2. Requests a car route from the public Valhalla demo server.
3. Uses Valhalla trace attributes to classify route sections.
4. Returns only the routing information useful to the AI agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from location.resolver import LocationResolver, ResolvedLocation

VALHALLA_ROUTE_URL = "https://valhalla1.openstreetmap.de/route"
VALHALLA_TRACE_ATTRIBUTES_URL = (
    "https://valhalla1.openstreetmap.de/trace_attributes"
)

REQUEST_TIMEOUT = 20
DEFAULT_LOCATION_RESOLVER = LocationResolver(timeout=REQUEST_TIMEOUT)


@dataclass(frozen=True)
class RoadClassificationConfig:
    """Thresholds and road classes used for deterministic segmentation."""

    urban_max_speed_kmh: float = 60.0
    highway_min_speed_kmh: float = 90.0
    highway_road_classes: frozenset[str] = frozenset({"motorway", "trunk"})
    urban_road_classes: frozenset[str] = frozenset(
        {"residential", "service_other"}
    )


DEFAULT_ROAD_CLASSIFICATION = RoadClassificationConfig()


def _optional_positive_float(value: Any) -> float | None:
    """Return a positive float, or ``None`` for absent/invalid values."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def classify_road(
    road_class: str | None,
    speed_limit: float | None,
    routing_speed: float | None,
    config: RoadClassificationConfig = DEFAULT_ROAD_CLASSIFICATION,
) -> str:
    """Classify one Valhalla edge as urban, suburban, or highway.

    The speed limit is preferred when available; otherwise Valhalla's routing
    speed is used. Missing values safely fall back to road-class hints and then
    to ``suburban``.
    """

    normalized_class = (
        str(road_class).strip().casefold().replace(" ", "_")
        if road_class
        else ""
    )
    effective_speed = _optional_positive_float(speed_limit)
    if effective_speed is None:
        effective_speed = _optional_positive_float(routing_speed)

    if (
        normalized_class in config.highway_road_classes
        or effective_speed is not None
        and effective_speed >= config.highway_min_speed_kmh
    ):
        return "highway"

    if (
        effective_speed is not None
        and effective_speed <= config.urban_max_speed_kmh
    ):
        return "urban"

    if effective_speed is not None:
        return "suburban"

    if normalized_class in config.urban_road_classes:
        return "urban"

    return "suburban"


def merge_consecutive_segments(
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge adjacent classified edges of the same road type."""

    merged: list[dict[str, Any]] = []
    speed_weights: list[float] = []
    weighted_speeds: list[float] = []
    posted_limits: list[list[float]] = []
    merge_keys: list[tuple[str, float | None, str]] = []

    for edge in edges:
        edge_type = edge.get("type")
        if edge_type not in {"urban", "suburban", "highway"}:
            continue

        distance = _optional_positive_float(edge.get("distance_km")) or 0.0
        speed = _optional_positive_float(
            edge.get("routing_speed_kmh", edge.get("average_speed_kmh"))
        )
        posted_limit = _optional_positive_float(
            edge.get("speed_limit_kmh")
        )
        edge_use = str(edge.get("use") or "").strip().casefold()
        special_use = (
            edge_use
            if edge_use in {"ramp", "turn_channel", "egress_connection"}
            else "ordinary"
        )
        merge_key = (edge_type, posted_limit, special_use)

        if not merged or merge_keys[-1] != merge_key:
            merged.append(
                {
                    "type": edge_type,
                    "distance_km": distance,
                    "average_speed_kmh": None,
                    "posted_speed_limit_kmh": None,
                    "road_classes": [],
                    "uses": [],
                    "road_names": [],
                }
            )
            speed_weights.append(0.0)
            weighted_speeds.append(0.0)
            posted_limits.append([])
            merge_keys.append(merge_key)
        else:
            merged[-1]["distance_km"] += distance

        segment = merged[-1]
        for output_key, edge_key in (
            ("road_classes", "road_class"),
            ("uses", "use"),
            ("road_names", "road_name"),
        ):
            value = edge.get(edge_key)
            if value and value not in segment[output_key]:
                segment[output_key].append(value)

        if speed is not None:
            weight = distance if distance > 0 else 1.0
            speed_weights[-1] += weight
            weighted_speeds[-1] += speed * weight

        if posted_limit is not None:
            posted_limits[-1].append(posted_limit)

    for index, segment in enumerate(merged):
        segment["distance_km"] = round(segment["distance_km"], 3)
        if speed_weights[index] > 0:
            segment["average_speed_kmh"] = round(
                weighted_speeds[index] / speed_weights[index],
                1,
            )
        if posted_limits[index]:
            # The safest limit governs a merged section that contains more
            # than one posted limit.
            segment["posted_speed_limit_kmh"] = min(posted_limits[index])

    return merged


def build_route_segments(
    edges: list[dict[str, Any]],
    config: RoadClassificationConfig = DEFAULT_ROAD_CLASSIFICATION,
) -> list[dict[str, Any]]:
    """Classify Valhalla edge records and merge consecutive types."""

    classified_edges = []
    for edge in edges:
        classified_edges.append(
            {
                **edge,
                "type": classify_road(
                    edge.get("road_class"),
                    edge.get("speed_limit_kmh"),
                    edge.get("routing_speed_kmh"),
                    config,
                ),
            }
        )
    return merge_consecutive_segments(classified_edges)


def _resolved_location_dict(location: ResolvedLocation) -> dict[str, Any]:
    """Serialize a resolved point while retaining the legacy name key."""

    return {
        "raw_input": location.raw_input,
        "canonical_name": location.canonical_name,
        "name": location.canonical_name,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "source": location.source,
        "confidence": location.confidence,
    }


def _first_road_name(names: Any) -> str | None:
    """Return the first usable Valhalla edge name."""

    if not isinstance(names, list):
        return None
    for name in names:
        if isinstance(name, str) and name.strip():
            return name.strip()
        if isinstance(name, dict):
            value = name.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _trace_route_edges(
    route_data: dict[str, Any],
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Return normalized edge attributes for all route legs.

    Detailed segmentation is optional. A missing route shape, unsupported
    attribute, or trace request failure returns an empty list so the original
    routing result remains available.
    """

    trip = route_data.get("trip")
    legs = trip.get("legs", []) if isinstance(trip, dict) else []
    if not isinstance(legs, list):
        return []

    route_edges: list[dict[str, Any]] = []
    attributes = [
        "edge.length",
        "edge.speed",
        "edge.speed_limit",
        "edge.road_class",
        "edge.use",
        "edge.names",
    ]

    for leg in legs:
        shape = leg.get("shape") if isinstance(leg, dict) else None
        if not isinstance(shape, str) or not shape:
            continue

        trace_payload = {
            "encoded_polyline": shape,
            "costing": "auto",
            "shape_match": "edge_walk",
            "units": "kilometers",
            "filters": {
                "action": "include",
                "attributes": attributes,
            },
        }

        try:
            response = requests.post(
                VALHALLA_TRACE_ATTRIBUTES_URL,
                json=trace_payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            trace_data = response.json()
        except (requests.RequestException, ValueError, TypeError):
            return []

        edges = trace_data.get("edges", [])
        if not isinstance(edges, list):
            return []

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            route_edges.append(
                {
                    "distance_km": (
                        _optional_positive_float(edge.get("length")) or 0.0
                    ),
                    "road_class": (
                        str(edge["road_class"]).strip().casefold()
                        if edge.get("road_class")
                        else None
                    ),
                    "use": (
                        str(edge["use"]).strip().casefold()
                        if edge.get("use")
                        else None
                    ),
                    "speed_limit_kmh": _optional_positive_float(
                        edge.get("speed_limit")
                    ),
                    "routing_speed_kmh": _optional_positive_float(
                        edge.get("speed")
                    ),
                    "road_name": _first_road_name(edge.get("names")),
                }
            )

    return route_edges


def get_route(
    origin: str,
    destination: str,
    *,
    location_resolver: LocationResolver | None = None,
    classification_config: RoadClassificationConfig = (
        DEFAULT_ROAD_CLASSIFICATION
    ),
) -> dict[str, Any]:
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
        - merged urban, suburban, and highway route segments when Valhalla
          detailed edge attributes are available
    """

    resolver = location_resolver or DEFAULT_LOCATION_RESOLVER
    origin_location = resolver.resolve(origin)
    destination_location = resolver.resolve(destination)

    payload = {
        "locations": [
            {
                "lat": origin_location.latitude,
                "lon": origin_location.longitude,
            },
            {
                "lat": destination_location.latitude,
                "lon": destination_location.longitude,
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

    route_edges = _trace_route_edges(data, headers)
    segments = build_route_segments(route_edges, classification_config)

    return {
        "origin": _resolved_location_dict(origin_location),
        "destination": _resolved_location_dict(destination_location),
        "distance_km": round(float(summary["length"]), 2),
        "duration_minutes": round(float(summary["time"]) / 60, 1),
        "uses_highway": bool(summary.get("has_highway", False)),
        "has_toll": bool(summary.get("has_toll", False)),
        "has_ferry": bool(summary.get("has_ferry", False)),
        "segments": segments,
    }


if __name__ == "__main__":
    route = get_route("Sfax", "Tunis")
    print(route)
