"""Expose normalized vehicle telemetry from the active trip simulator."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, Protocol


class VehicleStateProvider(Protocol):
    """Interface implemented by ``TripSimulator`` for state retrieval."""

    def get_state(self) -> dict[str, Any]:
        """Return the simulator's current combined state."""

        ...


class VehicleStateError(RuntimeError):
    """Raised when simulator state is unavailable or invalid."""


def _required_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise VehicleStateError(f"Vehicle state is missing '{key}'.")
    return value


def _required_number(
    source: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VehicleStateError(f"Vehicle state field '{key}' must be numeric.")

    normalized = float(value)
    if not isfinite(normalized):
        raise VehicleStateError(f"Vehicle state field '{key}' must be finite.")
    if minimum is not None and normalized < minimum:
        raise VehicleStateError(f"Vehicle state field '{key}' is below {minimum}.")
    if maximum is not None and normalized > maximum:
        raise VehicleStateError(f"Vehicle state field '{key}' is above {maximum}.")
    return normalized


def _optional_number(source: Mapping[str, Any], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    return _required_number(source, key, minimum=0.0)


def _normalize_ev(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy SUMO-derived electric-vehicle telemetry."""

    model = ev.get("model")
    drive = ev.get("drive")
    range_source = ev.get("range_estimate_source")
    if not isinstance(model, str) or not model:
        raise VehicleStateError("EV state is missing a valid 'model'.")
    if not isinstance(drive, str) or not drive:
        raise VehicleStateError("EV state is missing a valid 'drive'.")
    if range_source not in {"reference", "observed_consumption"}:
        raise VehicleStateError("EV state has an invalid range estimate source.")

    capacity = _required_number(ev, "battery_capacity_kwh", minimum=0.0)
    if capacity <= 0:
        raise VehicleStateError("EV battery capacity must be greater than zero.")

    actual_consumption = ev.get("actual_consumption_kwh_100km")
    if actual_consumption is not None:
        actual_consumption = _required_number(
            ev, "actual_consumption_kwh_100km", minimum=0.0
        )

    charging_station_id = ev.get("charging_station_id")
    if charging_station_id is not None and not isinstance(
        charging_station_id, str
    ):
        raise VehicleStateError("EV charging_station_id must be text or None.")

    return {
        "model": model,
        "drive": drive,
        "battery_capacity_kwh": capacity,
        "battery_energy_kwh": _required_number(
            ev, "battery_energy_kwh", minimum=0.0
        ),
        "battery_soc_percent": _required_number(
            ev, "battery_soc_percent", minimum=0.0, maximum=100.0
        ),
        "battery_low": bool(ev.get("battery_low", False)),
        "instant_consumption_wh_per_s": _required_number(
            ev, "instant_consumption_wh_per_s"
        ),
        "total_energy_consumed_kwh": _required_number(
            ev, "total_energy_consumed_kwh", minimum=0.0
        ),
        "total_energy_regenerated_kwh": _required_number(
            ev, "total_energy_regenerated_kwh", minimum=0.0
        ),
        "reference_consumption_kwh_100km": _required_number(
            ev, "reference_consumption_kwh_100km", minimum=0.0
        ),
        "reference_range_km": _required_number(
            ev, "reference_range_km", minimum=0.0
        ),
        "distance_travelled_km": _required_number(
            ev, "distance_travelled_km", minimum=0.0
        ),
        "actual_consumption_kwh_100km": actual_consumption,
        "estimated_remaining_range_km": _required_number(
            ev, "estimated_remaining_range_km", minimum=0.0
        ),
        "range_estimate_source": range_source,
        "charging_station_id": charging_station_id,
        "is_charging": bool(ev.get("is_charging", False)),
    }


def _normalize_health(health: Mapping[str, Any]) -> dict[str, Any]:
    """Validate supplemental EV-health estimates without recalculating them."""

    if health.get("source") != "supplemental_simulation":
        raise VehicleStateError("EV health has an invalid or missing source.")
    warnings = health.get("warnings")
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise VehicleStateError("EV health warnings must be a list of text values.")
    status = health.get("overall_status")
    if status not in {"normal", "warning"}:
        raise VehicleStateError("EV health has an invalid overall status.")

    tire_pressure = _required_mapping(health, "tire_pressure_psi")
    tire_temperature = _required_mapping(health, "tire_temperature_c")
    normalized_pressure = {
        wheel: _required_number(tire_pressure, wheel, minimum=0.0)
        for wheel in ("front_left", "front_right", "rear_left", "rear_right")
    }
    normalized_temperature = {
        wheel: _required_number(tire_temperature, wheel)
        for wheel in ("front_left", "front_right", "rear_left", "rear_right")
    }
    return {
        "source": "supplemental_simulation",
        "is_estimated": bool(health.get("is_estimated", True)),
        "overall_status": status,
        "warnings": list(warnings),
        "traction_battery_soh_percent": _required_number(
            health,
            "traction_battery_soh_percent",
            minimum=0.0,
            maximum=100.0,
        ),
        "battery_temperature_c": _required_number(
            health, "battery_temperature_c"
        ),
        "motor_temperature_c": _required_number(
            health, "motor_temperature_c"
        ),
        "inverter_temperature_c": _required_number(
            health, "inverter_temperature_c"
        ),
        "auxiliary_battery_voltage": _required_number(
            health, "auxiliary_battery_voltage", minimum=0.0
        ),
        "brake_pad_life_percent": _required_number(
            health, "brake_pad_life_percent", minimum=0.0, maximum=100.0
        ),
        "odometer_km": _required_number(health, "odometer_km", minimum=0.0),
        "tire_pressure_psi": normalized_pressure,
        "tire_temperature_c": normalized_temperature,
    }


def _normalize_location(location: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Validate a named trip endpoint supplied by ``TripSimulator``."""

    name = location.get("name")
    if not isinstance(name, str) or not name.strip():
        raise VehicleStateError(f"Vehicle trip {label} is missing a valid name.")

    country = location.get("country")
    if country is not None and not isinstance(country, str):
        raise VehicleStateError(
            f"Vehicle trip {label} country must be text or None."
        )

    city = location.get("city")
    if city is not None and (not isinstance(city, str) or not city.strip()):
        raise VehicleStateError(
            f"Vehicle trip {label} city must be non-empty text or None."
        )

    return {
        "name": name.strip(),
        "city": city.strip() if isinstance(city, str) else None,
        "country": country,
        "latitude": _required_number(
            location, "latitude", minimum=-90.0, maximum=90.0
        ),
        "longitude": _required_number(
            location, "longitude", minimum=-180.0, maximum=180.0
        ),
    }


def _normalize_trip(raw_state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return compact trip metadata when the simulator provides it."""

    origin = raw_state.get("origin")
    destination = raw_state.get("destination")
    route = raw_state.get("route")

    if origin is None and destination is None and route is None:
        return None
    if not isinstance(origin, Mapping) or not isinstance(destination, Mapping):
        raise VehicleStateError(
            "Vehicle trip metadata must include origin and destination mappings."
        )

    normalized: dict[str, Any] = {
        "origin": _normalize_location(origin, "origin"),
        "destination": _normalize_location(destination, "destination"),
    }

    if route is not None:
        if not isinstance(route, Mapping):
            raise VehicleStateError("Vehicle trip route must be a mapping.")
        normalized["route"] = {
            "distance_km": _required_number(route, "distance_km", minimum=0.0),
            "duration_minutes": _optional_number(route, "duration_minutes"),
            "uses_highway": bool(route.get("uses_highway", False)),
            "has_toll": bool(route.get("has_toll", False)),
            "has_ferry": bool(route.get("has_ferry", False)),
        }

    return normalized


def get_vehicle_state(simulator: VehicleStateProvider) -> dict[str, Any]:
    """Return a validated, agent-friendly view of a started trip simulator."""

    if simulator is None:
        raise ValueError("A TripSimulator instance is required.")
    get_state = getattr(simulator, "get_state", None)
    if not callable(get_state):
        raise TypeError("simulator must provide a callable get_state() method.")

    try:
        raw_state = get_state()
    except RuntimeError as exc:
        raise VehicleStateError(
            "Vehicle state is unavailable; start the TripSimulator first."
        ) from exc

    if not isinstance(raw_state, Mapping):
        raise VehicleStateError("TripSimulator.get_state() must return a dictionary.")

    movement = _required_mapping(raw_state, "movement")
    position = _required_mapping(movement, "position")
    ev = _required_mapping(raw_state, "ev")
    health = _required_mapping(raw_state, "health")
    traffic = raw_state.get("traffic")

    status = raw_state.get("status")
    if not isinstance(status, str) or not status:
        raise VehicleStateError("Vehicle state is missing a valid 'status'.")

    current_road = movement.get("current_road")
    if not isinstance(current_road, str) or not current_road:
        raise VehicleStateError("Vehicle state is missing a valid 'current_road'.")

    normalized_motion = {
        "speed_kmh": _required_number(movement, "speed_kph", minimum=0.0),
        "latitude": _required_number(
            position, "latitude", minimum=-90.0, maximum=90.0
        ),
        "longitude": _required_number(
            position, "longitude", minimum=-180.0, maximum=180.0
        ),
        "current_road": current_road,
        "trip_progress_percent": _required_number(
            movement,
            "trip_progress_percent",
            minimum=0.0,
            maximum=100.0,
        ),
        "elapsed_time_seconds": _required_number(
            movement, "elapsed_seconds", minimum=0.0
        ),
        "travelled_distance_km": _required_number(
            movement, "travelled_distance_km", minimum=0.0
        ),
    }
    for source_key, output_key in (
        ("acceleration_mps2", "acceleration_mps2"),
        ("lane_position_m", "lane_position_m"),
        ("angle_deg", "angle_deg"),
        ("waiting_time_s", "waiting_time_s"),
        ("simulation_time_s", "simulation_time_s"),
    ):
        if movement.get(source_key) is not None:
            normalized_motion[output_key] = _required_number(
                movement, source_key
            )
    for coordinate in ("x", "y"):
        if position.get(coordinate) is not None:
            normalized_motion[coordinate] = _required_number(position, coordinate)
    lane_id = movement.get("lane_id")
    if lane_id is not None:
        if not isinstance(lane_id, str):
            raise VehicleStateError("Vehicle state field 'lane_id' must be text.")
        normalized_motion["lane_id"] = lane_id
    for source_key in ("road_id", "road_name"):
        value = movement.get(source_key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise VehicleStateError(
                    f"Vehicle state field '{source_key}' must be non-empty text."
                )
            normalized_motion[source_key] = value

    normalized_state = {
        "status": status,
        "motion": normalized_motion,
        "ev": _normalize_ev(ev),
        "health": _normalize_health(health),
    }
    trip = _normalize_trip(raw_state)
    if trip is not None:
        normalized_state["trip"] = trip
    if traffic is not None:
        if not isinstance(traffic, Mapping):
            raise VehicleStateError("Vehicle state field 'traffic' must be a mapping.")
        scenario = traffic.get("scenario")
        traffic_level = traffic.get("traffic_level")
        if not isinstance(scenario, str) or not scenario:
            raise VehicleStateError("Traffic state is missing a valid 'scenario'.")
        if not isinstance(traffic_level, str) or not traffic_level:
            raise VehicleStateError("Traffic state is missing a valid 'traffic_level'.")
        normalized_state["traffic"] = {
            "scenario": scenario,
            "level": traffic_level,
            "is_congested": bool(traffic.get("is_congested", False)),
            "average_speed_kmh": _optional_number(traffic, "average_speed_kph"),
            "average_delay_seconds": _optional_number(
                traffic, "average_delay_seconds"
            ),
            "average_travel_time_seconds": _optional_number(
                traffic, "average_travel_time_seconds"
            ),
            "congestion_index_percent": _optional_number(
                traffic, "congestion_index_percent"
            ),
            "completed_trips": _optional_number(traffic, "completed_trips"),
            "unfinished_trips": _optional_number(traffic, "unfinished_trips"),
            "completion_percent": _optional_number(traffic, "completion_percent"),
            "active_vehicles": _optional_number(traffic, "active_vehicles"),
            "halted_vehicles": _optional_number(traffic, "halted_vehicles"),
            "current_road_vehicle_count": _optional_number(
                traffic, "current_road_vehicle_count"
            ),
            "current_road_mean_speed_kmh": _optional_number(
                traffic, "current_road_mean_speed_kph"
            ),
        }
    return normalized_state
