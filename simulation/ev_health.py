"""Supplemental EV-health estimates for values not provided by SUMO."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from simulation.ev_config import IONIQ5_LONG_RANGE_RWD


@dataclass(frozen=True)
class EVHealthConfig:
    """Configurable engineering approximations, not manufacturer telemetry."""

    ambient_temperature_c: float = 25.0
    initial_battery_temperature_c: float = 25.0
    initial_motor_temperature_c: float = 25.0
    initial_inverter_temperature_c: float = 25.0
    initial_battery_soh_percent: float = 100.0
    initial_tire_pressure_psi: float = 32.0
    initial_tire_temperature_c: float = 25.0
    initial_auxiliary_battery_voltage: float = 12.6
    initial_brake_pad_life_percent: float = 100.0
    initial_odometer_km: float = 0.0
    battery_thermal_time_constant_s: float = 900.0
    motor_thermal_time_constant_s: float = 420.0
    inverter_thermal_time_constant_s: float = 360.0
    battery_fade_percent_per_equivalent_cycle: float = 0.02


class EVHealthSimulator:
    """Estimate non-SUMO EV health from live motion and energy telemetry."""

    def __init__(self, config: EVHealthConfig | None = None) -> None:
        self.config = config or EVHealthConfig()
        if not 0 <= self.config.initial_battery_soh_percent <= 100:
            raise ValueError("initial_battery_soh_percent must be between 0 and 100.")
        if self.config.initial_tire_pressure_psi <= 0:
            raise ValueError("initial_tire_pressure_psi must be greater than zero.")
        self.reset()

    def reset(self) -> None:
        """Restore configured initial values and telemetry counters."""

        config = self.config
        self.battery_temperature_c = config.initial_battery_temperature_c
        self.motor_temperature_c = config.initial_motor_temperature_c
        self.inverter_temperature_c = config.initial_inverter_temperature_c
        self.battery_soh_percent = config.initial_battery_soh_percent
        self.tire_temperature_c = config.initial_tire_temperature_c
        self.auxiliary_battery_voltage = config.initial_auxiliary_battery_voltage
        self.brake_pad_life_percent = config.initial_brake_pad_life_percent
        self.odometer_km = config.initial_odometer_km
        self._last_consumed_kwh = 0.0
        self._last_regenerated_kwh = 0.0
        self._battery_low = False

    @staticmethod
    def _approach(current: float, target: float, seconds: float, tau: float) -> float:
        if seconds <= 0:
            return current
        fraction = 1.0 - exp(-seconds / tau)
        return current + (target - current) * fraction

    def update(
        self,
        *,
        speed_kmh: float,
        acceleration_mps2: float,
        elapsed_seconds: float,
        distance_delta_km: float,
        instant_consumption_wh_per_s: float,
        total_energy_consumed_kwh: float,
        total_energy_regenerated_kwh: float,
        battery_low: bool,
    ) -> dict[str, Any]:
        """Advance estimated health using one interval of SUMO telemetry."""

        values = (
            elapsed_seconds,
            distance_delta_km,
            total_energy_consumed_kwh,
            total_energy_regenerated_kwh,
        )
        if any(value < 0 for value in values):
            raise ValueError("Elapsed time, distance, and energy totals cannot be negative.")

        speed_kmh = max(float(speed_kmh), 0.0)
        acceleration_mps2 = float(acceleration_mps2)
        power_kw = abs(float(instant_consumption_wh_per_s)) * 3.6
        power_load = min(
            power_kw / IONIQ5_LONG_RANGE_RWD.max_motor_power_kw,
            1.0,
        )
        acceleration_load = min(abs(acceleration_mps2) / 4.5, 1.0)

        ambient = self.config.ambient_temperature_c
        battery_target = ambient + 4.0 + 20.0 * power_load + 4.0 * acceleration_load
        motor_target = ambient + 8.0 + 75.0 * power_load + 12.0 * acceleration_load
        inverter_target = ambient + 6.0 + 55.0 * power_load + 8.0 * acceleration_load
        self.battery_temperature_c = self._approach(
            self.battery_temperature_c,
            battery_target,
            elapsed_seconds,
            self.config.battery_thermal_time_constant_s,
        )
        self.motor_temperature_c = self._approach(
            self.motor_temperature_c,
            motor_target,
            elapsed_seconds,
            self.config.motor_thermal_time_constant_s,
        )
        self.inverter_temperature_c = self._approach(
            self.inverter_temperature_c,
            inverter_target,
            elapsed_seconds,
            self.config.inverter_thermal_time_constant_s,
        )

        is_initial_sync = elapsed_seconds == 0 and distance_delta_km == 0
        consumed_delta = 0.0 if is_initial_sync else max(
            total_energy_consumed_kwh - self._last_consumed_kwh, 0.0
        )
        regenerated_delta = 0.0 if is_initial_sync else max(
            total_energy_regenerated_kwh - self._last_regenerated_kwh, 0.0
        )
        throughput_kwh = consumed_delta + regenerated_delta
        equivalent_cycles = throughput_kwh / (
            2.0 * IONIQ5_LONG_RANGE_RWD.battery_capacity_kwh
        )
        self.battery_soh_percent = max(
            0.0,
            self.battery_soh_percent
            - equivalent_cycles
            * self.config.battery_fade_percent_per_equivalent_cycle,
        )
        self._last_consumed_kwh = total_energy_consumed_kwh
        self._last_regenerated_kwh = total_energy_regenerated_kwh

        tire_target = ambient + min(speed_kmh / 120.0, 1.0) * 18.0
        self.tire_temperature_c = self._approach(
            self.tire_temperature_c,
            tire_target,
            elapsed_seconds,
            600.0,
        )
        braking_load = min(max(-acceleration_mps2, 0.0) / 4.5, 1.0)
        friction_braking_share = 0.25
        self.brake_pad_life_percent = max(
            0.0,
            self.brake_pad_life_percent
            - distance_delta_km * braking_load * friction_braking_share * 0.002,
        )
        self.odometer_km += distance_delta_km
        voltage_target = 14.2 if speed_kmh > 1.0 else 12.6
        self.auxiliary_battery_voltage = self._approach(
            self.auxiliary_battery_voltage,
            voltage_target,
            elapsed_seconds,
            20.0,
        )
        self._battery_low = bool(battery_low)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return the current estimated health state with provenance metadata."""

        pressure = round(self.config.initial_tire_pressure_psi * (
            (self.tire_temperature_c + 273.15)
            / (self.config.initial_tire_temperature_c + 273.15)
        ), 2)
        warnings: list[str] = []
        if self._battery_low:
            warnings.append("traction_battery_low")
        if self.battery_temperature_c >= 45.0:
            warnings.append("traction_battery_temperature_high")
        if self.motor_temperature_c >= 100.0:
            warnings.append("motor_temperature_high")
        if pressure < 28.0:
            warnings.append("tire_pressure_low")
        if self.auxiliary_battery_voltage < 11.8:
            warnings.append("auxiliary_battery_voltage_low")

        return {
            "source": "supplemental_simulation",
            "is_estimated": True,
            "overall_status": "warning" if warnings else "normal",
            "warnings": warnings,
            "traction_battery_soh_percent": round(self.battery_soh_percent, 4),
            "battery_temperature_c": round(self.battery_temperature_c, 2),
            "motor_temperature_c": round(self.motor_temperature_c, 2),
            "inverter_temperature_c": round(self.inverter_temperature_c, 2),
            "auxiliary_battery_voltage": round(
                self.auxiliary_battery_voltage, 2
            ),
            "brake_pad_life_percent": round(self.brake_pad_life_percent, 4),
            "odometer_km": round(self.odometer_km, 3),
            "tire_pressure_psi": {
                "front_left": pressure,
                "front_right": pressure,
                "rear_left": pressure,
                "rear_right": pressure,
            },
            "tire_temperature_c": {
                "front_left": round(self.tire_temperature_c, 2),
                "front_right": round(self.tire_temperature_c, 2),
                "rear_left": round(self.tire_temperature_c, 2),
                "rear_right": round(self.tire_temperature_c, 2),
            },
        }
