"""Reference configuration for the SUMO electric ego vehicle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EVConfig:
    """Published EV references used to configure and describe a SUMO model."""

    model_name: str
    drive: str
    battery_capacity_kwh: float
    max_motor_power_kw: float
    max_motor_torque_nm: float
    max_speed_kmh: float
    reference_consumption_kwh_100km: float
    reference_range_km: float


IONIQ5_LONG_RANGE_RWD = EVConfig(
    model_name="Hyundai IONIQ 5 Long Range RWD",
    drive="RWD",
    battery_capacity_kwh=84.0,
    max_motor_power_kw=168.0,
    max_motor_torque_nm=350.0,
    max_speed_kmh=185.0,
    reference_consumption_kwh_100km=16.0,
    reference_range_km=570.0,
)

DEFAULT_INITIAL_SOC_PERCENT = 80.0
MIN_DISTANCE_FOR_OBSERVED_RANGE_KM = 1.0

# These parameters are engineering defaults for the SUMO energy calculation,
# not verified Hyundai specifications. They remain explicit so they can be
# calibrated later instead of being mistaken for manufacturer values.
SUMO_EV_MODEL_APPROXIMATIONS = {
    "accel_mps2": 2.6,
    "decel_mps2": 4.5,
    "vehicle_length_m": 5.0,
    "mass_kg": 1830.0,
    "front_surface_area_m2": 2.6,
    "air_drag_coefficient": 0.35,
    "rotating_mass_kg": 40.0,
    "radial_drag_coefficient": 0.1,
    "roll_drag_coefficient": 0.01,
    "constant_power_intake_w": 100.0,
    "propulsion_efficiency": 0.98,
    "recuperation_efficiency": 0.96,
}
