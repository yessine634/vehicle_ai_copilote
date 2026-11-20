"""Segment-aware ML trip-energy estimation implemented as a LangGraph subgraph.

The subgraph is calculation-only state.  It deliberately does not share the
copilot's persistent message history, and it fetches route, weather, and
vehicle values again on every invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypedDict

import joblib
import pandas as pd
from langgraph.graph import END, START, StateGraph

from simulation.ev_config import BMW_I3_120_AH, SUMO_EV_MODEL_APPROXIMATIONS
from tools.routing import get_route
from tools.vehicle import VehicleStateProvider, get_vehicle_state
from tools.weather import get_weather, get_weather_by_coordinates


FEATURES = [
    "motor_power_kW",
    "vehicle_weight_kg",
    "drag_coefficient_Cd",
    "frontal_area_m2",
    "rolling_resistance_Cr",
    "motor_efficiency_pct",
    "avg_driving_speed_kmh",
    "ambient_temperature_C",
    "ac_usage",
    "terrain_type",
    "traffic_condition",
    "driving_style",
    "passengers",
    "battery_age_years",
    "tyre_pressure_deviation_psi",
]

CATEGORICAL_FEATURES = {
    "ac_usage",
    "terrain_type",
    "traffic_condition",
    "driving_style",
}
INDIA_MODEL_TARGET = "energy_consumption_kWh_per_km"
TRAFFIC_CONDITION_CODES = {"low": "0", "normal": "1", "heavy": "2"}
REFERENCE_TIRE_PRESSURE_PSI = 32.0
DEFAULT_MODEL_ASSUMPTIONS_WARNING = (
    "The India model uses scenario assumptions where live inputs are not "
    "available: 92% motor efficiency, AC off, terrain category 0, normal "
    "driving style, one occupant, one-year battery age, and a 32 psi "
    "tire-pressure reference. SUMO traffic levels are mapped to the model's "
    "0/1/2 traffic categories."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "EV energy consumption"
    / "ev_energy_consumption_india_15_features.joblib"
)


class EnergyTripState(TypedDict, total=False):
    """Temporary state used only during one energy-estimation request."""

    origin: str
    destination: str
    reserve_soc_percent: float
    assumed_slope_percent: float | None
    assumed_acceleration_mps2: float | None
    driver_max_speed_kmh: float | None
    route: dict[str, Any]
    weather: dict[str, Any]
    vehicle: dict[str, Any]
    route_segments: list[dict[str, Any]]
    total_distance_km: float
    total_predicted_energy_kwh: float
    available_energy_kwh: float
    reserve_energy_kwh: float
    energy_available_after_reserve_kwh: float
    energy_after_trip_kwh: float
    projected_arrival_soc_percent: float
    feasible: bool
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True)
class EnergyEstimationConfig:
    """Auditable legal, physics, and validation constants."""

    default_average_slope_percent: float = 0.0
    default_average_acceleration_mps2: float = 0.0
    urban_limit_kmh: float = 50.0
    rural_limit_kmh: float = 90.0
    highway_limit_kmh: float = 110.0
    rainy_highway_limit_kmh: float = 90.0
    fog_highway_limit_kmh: float = 50.0
    fog_visibility_threshold_m: float = 50.0
    motorway_ramp_limit_kmh: float = 70.0
    gravity_mps2: float = 9.81
    rolling_resistance_coefficient: float = 0.012
    route_distance_tolerance_percent: float = 2.0
    route_distance_tolerance_min_km: float = 1.0
    model_training_max_distance_km: float = 50.0


DEFAULT_ENERGY_CONFIG = EnergyEstimationConfig()


class SegmentEnergyPredictor(Protocol):
    """Small interface that keeps model inference mockable in unit tests."""

    @property
    def target(self) -> str:
        """Return the saved model target name."""

        ...

    def predict(self, features: Mapping[str, Any]) -> float:
        """Predict energy consumption in kWh/km for one route segment."""

        ...


class SavedEnergyModel:
    """Lazy, validated adapter around the saved sklearn pipeline."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self._model: Any | None = None
        self._target = INDIA_MODEL_TARGET

    @property
    def target(self) -> str:
        self._load()
        return self._target

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Energy model was not found at '{self.model_path}'."
            )
        model = joblib.load(self.model_path)
        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features != FEATURES:
            raise RuntimeError(
                "Saved sklearn pipeline feature order does not match FEATURES."
            )
        self._model = model

    def predict(self, features: Mapping[str, Any]) -> float:
        self._load()
        missing = [name for name in FEATURES if name not in features]
        if missing:
            raise ValueError(
                "Energy feature vector is missing: " + ", ".join(missing)
            )
        ordered = {
            name: (
                str(features[name])
                if name in CATEGORICAL_FEATURES
                else float(features[name])
            )
            for name in FEATURES
        }
        frame = pd.DataFrame([ordered], columns=FEATURES)
        prediction = float(self._model.predict(frame)[0])
        if not isfinite(prediction) or prediction <= 0:
            raise RuntimeError(
                "Energy model returned a non-finite or non-positive "
                "kWh/km prediction."
            )
        return prediction


def _number(value: Any, label: str) -> float:
    """Return one finite numeric value or raise a clear validation error."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    return _number(value, "Optional value")


def _nested_number(
    source: Mapping[str, Any],
    section: str,
    key: str,
) -> float:
    nested = source.get(section)
    if not isinstance(nested, Mapping):
        raise ValueError(f"Vehicle state is missing '{section}'.")
    return _number(nested.get(key), f"Vehicle {section}.{key}")


def _append_warnings(
    state: Mapping[str, Any],
    *new_warnings: str,
) -> list[str]:
    warnings = list(state.get("warnings", []))
    for warning in new_warnings:
        if warning and warning not in warnings:
            warnings.append(warning)
    return warnings


def _error(state: Mapping[str, Any], message: str) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    errors.append(message)
    return {"errors": errors}


def _has_errors(state: Mapping[str, Any]) -> bool:
    return bool(state.get("errors"))


def _normalize_road_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace(" ", "_")
    aliases = {
        "urban": "urban",
        "residential": "urban",
        "rural": "rural",
        "suburban": "rural",
        "outside_urban": "rural",
        "highway": "highway",
        "motorway": "highway",
        "trunk": "highway",
    }
    return aliases.get(normalized)


def _is_raining(weather: Mapping[str, Any]) -> bool:
    rain = _optional_number(weather.get("rain_mm")) or 0.0
    precipitation = _optional_number(weather.get("precipitation_mm")) or 0.0
    code = weather.get("weather_code")
    rain_codes = {
        51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
        80, 81, 82, 95, 96, 99,
    }
    return rain > 0 or precipitation > 0 or code in rain_codes


def get_legal_speed_limit(
    road_type: str,
    posted_limit: float | None,
    weather: Mapping[str, Any],
    visibility: float | None = None,
    driver_profile: Mapping[str, Any] | None = None,
    *,
    uses: list[str] | None = None,
    config: EnergyEstimationConfig = DEFAULT_ENERGY_CONFIG,
) -> float:
    """Return the strictest known Tunisian speed limit in km/h.

    General limits are 50 km/h urban, 90 km/h outside urban areas, and
    110 km/h on motorways.  Tunisie Autoroutes documents 90 km/h on a
    motorway during rain, 50 km/h when fog visibility is below 50 metres,
    and 70/50 km/h on ramps and toll approaches.  A lower posted or known
    driver-specific limit always wins.
    """

    normalized_type = _normalize_road_type(road_type)
    if normalized_type is None:
        raise ValueError(f"Unsupported road classification: {road_type!r}.")

    default_limits = {
        "urban": config.urban_limit_kmh,
        "rural": config.rural_limit_kmh,
        "highway": config.highway_limit_kmh,
    }
    limit = default_limits[normalized_type]

    if posted_limit is not None:
        posted = _number(posted_limit, "Posted speed limit")
        if posted <= 0:
            raise ValueError("Posted speed limit must be greater than zero.")
        limit = min(limit, posted)

    normalized_uses = {
        str(item).strip().casefold().replace("-", "_")
        for item in (uses or [])
    }
    if normalized_type == "highway" and normalized_uses.intersection(
        {"ramp", "turn_channel", "egress_connection"}
    ):
        limit = min(limit, config.motorway_ramp_limit_kmh)

    if normalized_type == "highway" and _is_raining(weather):
        limit = min(limit, config.rainy_highway_limit_kmh)

    actual_visibility = visibility
    if actual_visibility is None:
        actual_visibility = _optional_number(weather.get("visibility_m"))
    if (
        normalized_type == "highway"
        and actual_visibility is not None
        and actual_visibility < config.fog_visibility_threshold_m
    ):
        limit = min(limit, config.fog_highway_limit_kmh)

    if driver_profile:
        driver_limit = _optional_number(
            driver_profile.get("max_speed_kmh")
        )
        if driver_limit is not None:
            if driver_limit <= 0:
                raise ValueError(
                    "Driver-specific speed limit must be greater than zero."
                )
            limit = min(limit, driver_limit)

    return float(limit)


def build_ml_features(
    *,
    vehicle: Mapping[str, Any],
    weather: Mapping[str, Any],
    segment: Mapping[str, Any],
    config: EnergyEstimationConfig = DEFAULT_ENERGY_CONFIG,
) -> dict[str, Any]:
    """Build the saved model's 15 inputs using their training units.

    Continuous inputs use kW, kg, m², km/h, °C, percent, years, and psi as
    indicated by their feature names. Drag and rolling resistance are
    dimensionless coefficients. Categorical inputs are string codes. Tire
    pressure is the mean tire pressure in psi minus the 32 psi reference.
    """

    ev = vehicle.get("ev")
    health = vehicle.get("health")
    if not isinstance(ev, Mapping) or not isinstance(health, Mapping):
        raise ValueError("Vehicle state must contain EV and health sections.")

    model_name = ev.get("model")
    if not isinstance(model_name, str) or "bmw i3" not in model_name.casefold():
        raise ValueError(
            "The saved estimator is configured only for the BMW i3 120 Ah."
        )

    tire_pressures = health.get("tire_pressure_psi")
    if not isinstance(tire_pressures, Mapping):
        raise ValueError("Vehicle health is missing tire pressures.")
    pressure_values = [
        _number(tire_pressures.get(wheel), f"Tire pressure {wheel}")
        for wheel in (
            "front_left",
            "front_right",
            "rear_left",
            "rear_right",
        )
    ]

    speed_kmh = _number(
        segment.get("estimated_speed_kmh"), "Segment estimated speed"
    )
    mean_tire_pressure = sum(pressure_values) / len(pressure_values)

    traffic = vehicle.get("traffic")
    traffic_level = (
        # get_vehicle_state() normalizes SUMO's ``traffic_level`` field to
        # ``level`` before the subgraph receives the vehicle state.
        traffic.get("level", "normal")
        if isinstance(traffic, Mapping)
        else "normal"
    )
    traffic_code = TRAFFIC_CONDITION_CODES.get(
        str(traffic_level).strip().casefold(),
        TRAFFIC_CONDITION_CODES["normal"],
    )

    features: dict[str, Any] = {
        "motor_power_kW": BMW_I3_120_AH.max_motor_power_kw,
        "vehicle_weight_kg": BMW_I3_120_AH.vehicle_mass_kg,
        "drag_coefficient_Cd": SUMO_EV_MODEL_APPROXIMATIONS[
            "air_drag_coefficient"
        ],
        "frontal_area_m2": SUMO_EV_MODEL_APPROXIMATIONS[
            "front_surface_area_m2"
        ],
        "rolling_resistance_Cr": config.rolling_resistance_coefficient,
        # The India notebook's BMW i3 examples use 92% as an explicit
        # inference assumption; SUMO's propulsion efficiency is a different
        # simulation parameter and is not treated as motor efficiency here.
        "motor_efficiency_pct": 92.0,
        "avg_driving_speed_kmh": speed_kmh,
        "ambient_temperature_C": _number(
            weather.get("temperature_c"), "Weather temperature"
        ),
        # The training notebook's BMW examples use these category codes.
        "ac_usage": "0",
        "terrain_type": "0",
        "traffic_condition": traffic_code,
        "driving_style": "1",
        "passengers": 1.0,
        "battery_age_years": 1.0,
        "tyre_pressure_deviation_psi": (
            mean_tire_pressure - REFERENCE_TIRE_PRESSURE_PSI
        ),
    }
    return {name: features[name] for name in FEATURES}


RouteProvider = Callable[[str, str], dict[str, Any]]
WeatherProvider = Callable[[str], dict[str, Any]]
CoordinateWeatherProvider = Callable[..., dict[str, Any]]
VehicleProvider = Callable[[VehicleStateProvider], dict[str, Any]]


def build_energy_estimation_graph(
    simulator: VehicleStateProvider,
    *,
    route_provider: RouteProvider = get_route,
    weather_provider: WeatherProvider = get_weather,
    coordinate_weather_provider: CoordinateWeatherProvider | None = None,
    vehicle_provider: VehicleProvider = get_vehicle_state,
    predictor: SegmentEnergyPredictor | None = None,
    config: EnergyEstimationConfig = DEFAULT_ENERGY_CONFIG,
):
    """Compile the fresh, deterministic trip-energy calculation subgraph."""

    model = predictor or SavedEnergyModel()

    def fetch_fresh_context(state: EnergyTripState) -> dict[str, Any]:
        try:
            origin = state.get("origin")
            destination = state.get("destination")
            if not isinstance(origin, str) or not origin.strip():
                raise ValueError("A route origin is required.")
            if not isinstance(destination, str) or not destination.strip():
                raise ValueError("A route destination is required.")

            reserve = _number(
                state.get("reserve_soc_percent", 15.0),
                "Safety reserve SOC",
            )
            if not 0 <= reserve <= 100:
                raise ValueError(
                    "Safety reserve SOC must be between 0 and 100 percent."
                )

            route = route_provider(origin.strip(), destination.strip())
            if not isinstance(route, dict):
                raise ValueError("Routing provider did not return a dictionary.")

            route_origin = route.get("origin")
            route_destination = route.get("destination")
            resolved_origin = (
                str(
                    route_origin.get("canonical_name")
                    or route_origin.get("name")
                    or origin
                )
                if isinstance(route_origin, Mapping)
                else origin.strip()
            )
            resolved_destination = (
                str(
                    route_destination.get("canonical_name")
                    or route_destination.get("name")
                    or destination
                )
                if isinstance(route_destination, Mapping)
                else destination.strip()
            )

            coordinates_provider = coordinate_weather_provider
            if coordinates_provider is None and weather_provider is get_weather:
                coordinates_provider = get_weather_by_coordinates

            if coordinates_provider is not None:
                if not isinstance(route_destination, Mapping):
                    raise ValueError(
                        "Resolved route destination coordinates are unavailable."
                    )
                destination_latitude = _number(
                    route_destination.get("latitude"),
                    "Resolved destination latitude",
                )
                destination_longitude = _number(
                    route_destination.get("longitude"),
                    "Resolved destination longitude",
                )
                weather = coordinates_provider(
                    destination_latitude,
                    destination_longitude,
                    location_name=resolved_destination,
                )
            else:
                # Retain compatibility with deterministic tests and custom
                # providers that intentionally accept a location name.
                weather = weather_provider(destination.strip())

            vehicle = vehicle_provider(simulator)
            if not isinstance(vehicle, dict):
                raise ValueError("Vehicle provider did not return a dictionary.")
            if not isinstance(weather, dict):
                raise ValueError("Weather provider did not return a dictionary.")
            if weather.get("error"):
                raise ValueError(f"Weather unavailable: {weather['error']}")

            warnings = _append_warnings(
                state,
                "Weather is sampled at the destination and applied to all "
                "segments; conditions may vary along the route.",
                "Temporary roadworks and novice-driver restrictions are "
                "applied only when a lower posted or driver-specific limit "
                "is available.",
            )
            if weather.get("visibility_m") is None:
                warnings = _append_warnings(
                    {"warnings": warnings},
                    "Visibility is unavailable, so the motorway fog limit "
                    "cannot be evaluated.",
                )

            return {
                "origin": resolved_origin,
                "destination": resolved_destination,
                "reserve_soc_percent": reserve,
                "route": route,
                "vehicle": vehicle,
                "weather": weather,
                "warnings": warnings,
                "errors": [],
            }
        except Exception as exc:
            return _error(state, f"Fresh trip data could not be retrieved: {exc}")

    def segment_route(state: EnergyTripState) -> dict[str, Any]:
        if _has_errors(state):
            return {}
        try:
            route = state["route"]
            raw_segments = route.get("segments")
            if not isinstance(raw_segments, list) or not raw_segments:
                raise ValueError(
                    "Valhalla did not provide classified route segments."
                )

            assumed_slope = _optional_number(
                state.get("assumed_slope_percent")
            )
            if assumed_slope is None:
                assumed_slope = config.default_average_slope_percent
            assumed_acceleration = _optional_number(
                state.get("assumed_acceleration_mps2")
            )
            if assumed_acceleration is None:
                assumed_acceleration = (
                    config.default_average_acceleration_mps2
                )
            normalized_segments: list[dict[str, Any]] = []
            warnings = list(state.get("warnings", []))

            for index, raw in enumerate(raw_segments, start=1):
                if not isinstance(raw, Mapping):
                    raise ValueError(f"Route segment {index} is not a mapping.")
                road_type = _normalize_road_type(
                    raw.get("road_type", raw.get("type"))
                )
                if road_type is None:
                    raise ValueError(
                        f"Route segment {index} has no supported road "
                        "classification."
                    )
                distance = _number(
                    raw.get("distance_km"), f"Route segment {index} distance"
                )
                if distance <= 0:
                    raise ValueError(
                        f"Route segment {index} distance must be positive."
                    )

                slope = _optional_number(
                    raw.get("slope_percent", raw.get("average_slope_percent"))
                )
                if slope is None:
                    slope = assumed_slope
                    warnings = _append_warnings(
                        {"warnings": warnings},
                        f"The configured {assumed_slope:.2f}% average slope "
                        "default is used where route grade is missing.",
                    )

                acceleration = _optional_number(raw.get("acceleration_mps2"))
                if acceleration is None:
                    acceleration = assumed_acceleration
                    warnings = _append_warnings(
                        {"warnings": warnings},
                        f"The configured {assumed_acceleration:.2f} m/s² "
                        "average acceleration default is used for segments "
                        "without a driving profile (0 means steady speed).",
                    )

                normalized_segments.append(
                    {
                        "segment_id": index,
                        "road_type": road_type,
                        "distance_km": distance,
                        "posted_speed_limit_kmh": _optional_number(
                            raw.get(
                                "posted_speed_limit_kmh",
                                raw.get("speed_limit_kmh"),
                            )
                        ),
                        "route_speed_kmh": _optional_number(
                            raw.get(
                                "average_speed_kmh",
                                raw.get("routing_speed_kmh"),
                            )
                        ),
                        "slope_percent": slope,
                        "acceleration_mps2": acceleration,
                        "road_names": (
                            list(raw["road_names"])
                            if isinstance(raw.get("road_names"), list)
                            else []
                        ),
                        "uses": (
                            list(raw["uses"])
                            if isinstance(raw.get("uses"), list)
                            else []
                        ),
                    }
                )

                if distance > config.model_training_max_distance_km:
                    warnings = _append_warnings(
                        {"warnings": warnings},
                        f"Segment {index} distance ({distance:.1f} km) is "
                        "outside the model training distance range (up to "
                        f"about {config.model_training_max_distance_km:.0f} "
                        "km); this is extrapolation.",
                    )

            total_distance = _number(
                route.get("distance_km"), "Total route distance"
            )
            segment_distance = sum(
                item["distance_km"] for item in normalized_segments
            )
            tolerance = max(
                config.route_distance_tolerance_min_km,
                total_distance
                * config.route_distance_tolerance_percent
                / 100.0,
            )
            if abs(segment_distance - total_distance) > tolerance:
                raise ValueError(
                    "Segment distances do not match the route total: "
                    f"{segment_distance:.2f} km versus {total_distance:.2f} "
                    f"km (tolerance {tolerance:.2f} km)."
                )

            warnings = _append_warnings(
                {"warnings": warnings},
                "The model predicts an energy-consumption rate in kWh/km. "
                "Each segment rate is multiplied by that segment's distance "
                "in km to estimate energy in kWh. Segment-level additivity "
                "was not validated during training, so summing segment "
                "estimates is experimental, not a guarantee.",
                "Future segment SOC, traction voltage, and battery "
                "temperature are not invented; current measurements and the "
                "published nominal traction voltage are reused as model "
                "context.",
            )
            return {
                "route_segments": normalized_segments,
                "total_distance_km": total_distance,
                "warnings": warnings,
            }
        except Exception as exc:
            return _error(state, f"Route segmentation failed: {exc}")

    def apply_speed_limits(state: EnergyTripState) -> dict[str, Any]:
        if _has_errors(state):
            return {}
        try:
            weather = state["weather"]
            driver_limit = _optional_number(state.get("driver_max_speed_kmh"))
            driver_profile = (
                {"max_speed_kmh": driver_limit}
                if driver_limit is not None
                else None
            )
            segments: list[dict[str, Any]] = []
            warnings = list(state.get("warnings", []))
            for segment in state["route_segments"]:
                legal = get_legal_speed_limit(
                    segment["road_type"],
                    segment["posted_speed_limit_kmh"],
                    weather,
                    weather.get("visibility_m"),
                    driver_profile,
                    uses=segment.get("uses"),
                    config=config,
                )
                route_speed = segment.get("route_speed_kmh")
                if route_speed is None:
                    raise ValueError(
                        f"Segment {segment['segment_id']} has no Valhalla "
                        "routing-speed estimate; a legal maximum is not an "
                        "average driving speed."
                    )
                if route_speed <= 0:
                    raise ValueError("Routing speed must be greater than zero.")
                estimated = min(float(route_speed), legal)
                if route_speed > legal:
                    warnings = _append_warnings(
                        {"warnings": warnings},
                        f"Segment {segment['segment_id']} routing speed was "
                        f"capped at the applicable {legal:.0f} km/h limit.",
                    )
                segments.append(
                    {
                        **segment,
                        "legal_speed_limit_kmh": legal,
                        "estimated_speed_kmh": estimated,
                    }
                )
            return {"route_segments": segments, "warnings": warnings}
        except Exception as exc:
            return _error(state, f"Speed policy could not be applied: {exc}")

    def prepare_features(state: EnergyTripState) -> dict[str, Any]:
        if _has_errors(state):
            return {}
        try:
            segments = []
            for segment in state["route_segments"]:
                features = build_ml_features(
                    vehicle=state["vehicle"],
                    weather=state["weather"],
                    segment=segment,
                    config=config,
                )
                segments.append({**segment, "ml_features": features})
            return {
                "route_segments": segments,
                "warnings": _append_warnings(
                    state, DEFAULT_MODEL_ASSUMPTIONS_WARNING
                ),
            }
        except Exception as exc:
            return _error(state, f"ML feature preparation failed: {exc}")

    def predict_segments(state: EnergyTripState) -> dict[str, Any]:
        if _has_errors(state):
            return {}
        try:
            segments = []
            for segment in state["route_segments"]:
                consumption_rate = _number(
                    model.predict(segment["ml_features"]),
                    "Predicted segment consumption (kWh/km)",
                )
                if consumption_rate <= 0:
                    raise ValueError(
                        "Predicted segment consumption must be positive."
                    )
                distance_km = _number(
                    segment.get("distance_km"), "Segment distance"
                )
                segment_energy_kwh = consumption_rate * distance_km
                segments.append(
                    {
                        **segment,
                        "predicted_energy_kwh": segment_energy_kwh,
                    }
                )
            return {"route_segments": segments}
        except Exception as exc:
            return _error(state, f"ML segment prediction failed: {exc}")

    def aggregate_energy(state: EnergyTripState) -> dict[str, Any]:
        if _has_errors(state):
            return {}
        try:
            total = sum(
                _number(
                    segment.get("predicted_energy_kwh"),
                    "Predicted segment energy",
                )
                for segment in state["route_segments"]
            )
            return {"total_predicted_energy_kwh": total}
        except Exception as exc:
            return _error(state, f"Energy aggregation failed: {exc}")

    def check_feasibility(state: EnergyTripState) -> dict[str, Any]:
        if _has_errors(state):
            return {}
        try:
            vehicle = state["vehicle"]
            capacity = _nested_number(
                vehicle, "ev", "battery_capacity_kwh"
            )
            soc = _nested_number(vehicle, "ev", "battery_soc_percent")
            if capacity <= 0 or not 0 <= soc <= 100:
                raise ValueError("Battery capacity/SOC is outside valid bounds.")

            ev = vehicle["ev"]
            raw_available = ev.get("battery_energy_kwh")
            warnings = list(state.get("warnings", []))
            if raw_available is None:
                available = capacity * soc / 100.0
                warnings = _append_warnings(
                    {"warnings": warnings},
                    "Available battery energy was calculated from configured "
                    "capacity and current SOC because no direct energy value "
                    "was available.",
                )
            else:
                available = _number(
                    raw_available, "Available battery energy"
                )

            reserve = (
                capacity * state.get("reserve_soc_percent", 15.0) / 100.0
            )
            usable_after_reserve = max(available - reserve, 0.0)
            total = state["total_predicted_energy_kwh"]
            after_trip = available - total
            feasible = total <= usable_after_reserve
            projected_soc = max(after_trip, 0.0) / capacity * 100.0
            return {
                "available_energy_kwh": available,
                "reserve_energy_kwh": reserve,
                "energy_available_after_reserve_kwh": usable_after_reserve,
                "energy_after_trip_kwh": after_trip,
                "projected_arrival_soc_percent": projected_soc,
                "feasible": feasible,
                "warnings": warnings,
            }
        except Exception as exc:
            return _error(state, f"Battery feasibility check failed: {exc}")

    builder = StateGraph(EnergyTripState)
    builder.add_node("fetch_fresh_context", fetch_fresh_context)
    builder.add_node("segment_route", segment_route)
    builder.add_node("apply_speed_limits", apply_speed_limits)
    builder.add_node("prepare_features", prepare_features)
    builder.add_node("predict_segments", predict_segments)
    builder.add_node("aggregate_energy", aggregate_energy)
    builder.add_node("check_feasibility", check_feasibility)
    builder.add_edge(START, "fetch_fresh_context")
    builder.add_edge("fetch_fresh_context", "segment_route")
    builder.add_edge("segment_route", "apply_speed_limits")
    builder.add_edge("apply_speed_limits", "prepare_features")
    builder.add_edge("prepare_features", "predict_segments")
    builder.add_edge("predict_segments", "aggregate_energy")
    builder.add_edge("aggregate_energy", "check_feasibility")
    builder.add_edge("check_feasibility", END)
    return builder.compile()


def energy_result(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the compact trip-energy result needed by the LLM.

    Route payloads, weather observations, raw vehicle telemetry, per-segment
    feature vectors, and per-segment predictions remain inside the subgraph
    State.  Keeping them out of this return value prevents oversized ReAct
    tool messages while still exposing the values needed for a driver answer.
    """

    errors = list(state.get("errors", []))
    all_warnings = list(state.get("warnings", []))
    public_warnings = all_warnings[:8]
    base: dict[str, Any] = {
        "status": "error" if errors else "ok",
        "origin": state.get("origin"),
        "destination": state.get("destination"),
        "warnings": public_warnings,
        "additional_warning_count": max(
            len(all_warnings) - len(public_warnings),
            0,
        ),
    }
    if errors:
        base["errors"] = errors
        return base

    base.update(
        {
            "total_distance_km": round(state["total_distance_km"], 2),
            "estimated_energy_consumption_kwh": round(
                state["total_predicted_energy_kwh"], 3
            ),
            "available_energy_kwh": round(
                state["available_energy_kwh"], 3
            ),
            "reserve_soc_percent": round(
                state.get("reserve_soc_percent", 15.0), 2
            ),
            "reserve_energy_kwh": round(state["reserve_energy_kwh"], 3),
            "energy_available_after_reserve_kwh": round(
                state["energy_available_after_reserve_kwh"], 3
            ),
            "energy_after_trip_kwh": round(
                state["energy_after_trip_kwh"], 3
            ),
            "projected_arrival_soc_percent": round(
                state["projected_arrival_soc_percent"], 2
            ),
            "feasible": state["feasible"],
        }
    )
    return base


__all__ = [
    "DEFAULT_ENERGY_CONFIG",
    "EnergyEstimationConfig",
    "EnergyTripState",
    "FEATURES",
    "SavedEnergyModel",
    "build_energy_estimation_graph",
    "build_ml_features",
    "energy_result",
    "get_legal_speed_limit",
]
