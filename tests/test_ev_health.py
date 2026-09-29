"""Tests for supplemental electric-vehicle health simulation."""

import unittest

from simulation.ev_health import EVHealthConfig, EVHealthSimulator


class EVHealthSimulatorTests(unittest.TestCase):
    def test_update_uses_sumo_telemetry_and_returns_estimates(self) -> None:
        simulator = EVHealthSimulator()

        state = simulator.update(
            speed_kmh=80.0,
            acceleration_mps2=2.0,
            elapsed_seconds=120.0,
            distance_delta_km=2.0,
            instant_consumption_wh_per_s=35.0,
            total_energy_consumed_kwh=1.2,
            total_energy_regenerated_kwh=0.1,
            battery_low=False,
        )

        self.assertEqual(state["source"], "supplemental_simulation")
        self.assertTrue(state["is_estimated"])
        self.assertGreater(state["battery_temperature_c"], 25.0)
        self.assertGreater(state["motor_temperature_c"], 25.0)
        self.assertGreater(state["inverter_temperature_c"], 25.0)
        self.assertLess(state["traction_battery_soh_percent"], 100.0)
        self.assertEqual(state["odometer_km"], 2.0)
        self.assertIn("front_left", state["tire_pressure_psi"])

    def test_low_sumo_battery_creates_health_warning(self) -> None:
        simulator = EVHealthSimulator()

        state = simulator.update(
            speed_kmh=0,
            acceleration_mps2=0,
            elapsed_seconds=1,
            distance_delta_km=0,
            instant_consumption_wh_per_s=0,
            total_energy_consumed_kwh=0,
            total_energy_regenerated_kwh=0,
            battery_low=True,
        )

        self.assertEqual(state["overall_status"], "warning")
        self.assertIn("traction_battery_low", state["warnings"])

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EVHealthSimulator(EVHealthConfig(initial_tire_pressure_psi=0))

    def test_negative_telemetry_is_rejected(self) -> None:
        simulator = EVHealthSimulator()
        with self.assertRaises(ValueError):
            simulator.update(
                speed_kmh=0,
                acceleration_mps2=0,
                elapsed_seconds=-1,
                distance_delta_km=0,
                instant_consumption_wh_per_s=0,
                total_energy_consumed_kwh=0,
                total_energy_regenerated_kwh=0,
                battery_low=False,
            )


if __name__ == "__main__":
    unittest.main()
