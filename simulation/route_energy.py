"""Deterministic route-aware EV energy prediction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class RouteEnergyConfig:
    """Configuration for baseline route-energy estimation."""

    urban_consumption_kwh_100km: float = 15.0
    suburban_consumption_kwh_100km: float = 17.0
    highway_consumption_kwh_100km: float = 21.0
    fallback_consumption_kwh_100km: float = 16.0
    reserve_soc_percent: float = 15.0

    def __post_init__(self) -> None:
        consumptions = (
            self.urban_consumption_kwh_100km,
            self.suburban_consumption_kwh_100km,
            self.highway_consumption_kwh_100km,
            self.fallback_consumption_kwh_100km,
        )
        if any(value <= 0 for value in consumptions):
            raise ValueError("Consumption configuration must be positive.")
        if not 0 <= self.reserve_soc_percent <= 100:
            raise ValueError("reserve_soc_percent must be between 0 and 100.")

    def baseline_for(self, driving_type: str | None) -> float:
        """Return baseline consumption for a route segment type."""

        return {
            "urban": self.urban_consumption_kwh_100km,
            "suburban": self.suburban_consumption_kwh_100km,
            "highway": self.highway_consumption_kwh_100km,
        }.get(str(driving_type).casefold(), self.fallback_consumption_kwh_100km)


DEFAULT_ROUTE_ENERGY_CONFIG = RouteEnergyConfig()


def estimate_route_energy(
    *,
    route_segments: list[dict[str, Any]],
    battery_capacity_kwh: float,
    current_energy_kwh: float,
    config: RouteEnergyConfig = DEFAULT_ROUTE_ENERGY_CONFIG,
) -> dict[str, Any]:
    """Calculate baseline route energy, arrival SOC, and feasibility."""

    capacity = float(battery_capacity_kwh)
    current_energy = float(current_energy_kwh)
    if not isfinite(capacity) or capacity <= 0:
        raise ValueError("battery_capacity_kwh must be positive and finite.")
    if not isfinite(current_energy) or current_energy < 0:
        raise ValueError("current_energy_kwh must be non-negative and finite.")

    segment_results: list[dict[str, Any]] = []
    total_distance = 0.0
    total_energy = 0.0

    for segment in route_segments:
        distance = float(segment.get("distance_km", 0.0))
        if not isfinite(distance) or distance < 0:
            raise ValueError("Route segment distance must be non-negative and finite.")
        kind = str(segment.get("type", "unknown")).casefold()
        consumption = config.baseline_for(kind)
        energy = distance * consumption / 100.0
        total_distance += distance
        total_energy += energy
        segment_results.append(
            {
                "type": kind,
                "distance_km": round(distance, 3),
                "average_speed_kmh": segment.get("average_speed_kmh"),
                "consumption_kwh_100km": round(consumption, 3),
                "energy_required_kwh": round(energy, 3),
            }
        )

    average_consumption = (
        total_energy / total_distance * 100.0 if total_distance > 0 else 0.0
    )
    raw_arrival_energy = current_energy - total_energy
    arrival_energy = max(raw_arrival_energy, 0.0)
    arrival_soc = min(max(arrival_energy / capacity * 100.0, 0.0), 100.0)
    energy_deficit = max(total_energy - current_energy, 0.0)
    reserve_energy = capacity * config.reserve_soc_percent / 100.0

    return {
        "method": "route_aware_baseline",
        "battery": {
            "capacity_kwh": round(capacity, 3),
            "current_energy_kwh": round(current_energy, 3),
            "current_soc_percent": round(
                min(max(current_energy / capacity * 100.0, 0.0), 100.0), 2
            ),
            "reserve_soc_percent": config.reserve_soc_percent,
            "reserve_energy_kwh": round(reserve_energy, 3),
        },
        "route": {
            "distance_km": round(total_distance, 3),
            "average_consumption_kwh_100km": round(average_consumption, 3),
            "required_energy_kwh": round(total_energy, 3),
            "segments": segment_results,
        },
        "arrival": {
            "expected_energy_kwh": round(arrival_energy, 3),
            "expected_soc_percent": round(arrival_soc, 2),
            "energy_deficit_kwh": round(energy_deficit, 3),
        },
        "feasibility": {
            "can_reach_destination": raw_arrival_energy >= 0,
            "can_reach_with_reserve": raw_arrival_energy >= reserve_energy,
        },
    }
