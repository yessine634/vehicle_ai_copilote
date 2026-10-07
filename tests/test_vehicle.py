"""Tests for the SUMO electric-vehicle state adapter."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from tools.vehicle import VehicleStateError, get_vehicle_state


SIMULATOR_STATE = {
    "status": "running",
    "movement": {
        "speed_kph": 42.5,
        "acceleration_mps2": 0.7,
        "position": {"latitude": 34.76, "longitude": 10.74},
        "current_road": "Route El Ain",
        "road_name": "Route El Ain",
        "road_id": "729752355#3",
        "lane_id": "729752355#3_0",
        "elapsed_seconds": 120.0,
        "travelled_distance_km": 1.42,
        "trip_progress_percent": 18.5,
    },
    "ev": {
        "model": "Hyundai IONIQ 5 Long Range RWD",
        "drive": "RWD",
        "battery_capacity_kwh": 84.0,
        "battery_energy_kwh": 66.8,
        "battery_soc_percent": 79.52,
        "battery_low": False,
        "instant_consumption_wh_per_s": 3.6,
        "total_energy_consumed_kwh": 0.42,
        "total_energy_regenerated_kwh": 0.02,
        "reference_consumption_kwh_100km": 16.0,
        "reference_range_km": 570.0,
        "distance_travelled_km": 1.42,
        "actual_consumption_kwh_100km": 29.58,
        "estimated_remaining_range_km": 225.83,
        "range_estimate_source": "observed_consumption",
        "charging_station_id": None,
        "is_charging": False,
    },
    "health": {
        "source": "supplemental_simulation",
        "is_estimated": True,
        "overall_status": "normal",
        "warnings": [],
        "traction_battery_soh_percent": 99.999,
        "battery_temperature_c": 28.4,
        "motor_temperature_c": 37.2,
        "inverter_temperature_c": 34.1,
        "auxiliary_battery_voltage": 14.1,
        "brake_pad_life_percent": 99.99,
        "odometer_km": 1.42,
        "tire_pressure_psi": {
            "front_left": 32.2,
            "front_right": 32.2,
            "rear_left": 32.2,
            "rear_right": 32.2,
        },
        "tire_temperature_c": {
            "front_left": 27.0,
            "front_right": 27.0,
            "rear_left": 27.0,
            "rear_right": 27.0,
        },
    },
    "traffic": {
        "completed": False,
        "scenario": "normal",
        "traffic_level": "normal",
        "is_congested": False,
        "active_vehicles": 12,
        "halted_vehicles": 1,
        "average_speed_kph": 35.0,
        "average_delay_seconds": 2.0,
        "congestion_index_percent": 8.33,
    },
}


class VehicleToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = Mock()
        self.simulator.get_state.return_value = SIMULATOR_STATE

    def test_get_vehicle_state_returns_motion_and_ev(self) -> None:
        state = get_vehicle_state(self.simulator)

        self.assertIsInstance(state, dict)
        self.assertIn("motion", state)
        self.assertIn("ev", state)
        self.assertIn("health", state)
        self.assertIsInstance(state["motion"]["speed_kmh"], float)
        self.assertEqual(state["motion"]["current_road"], "Route El Ain")
        self.assertEqual(state["motion"]["road_id"], "729752355#3")
        self.assertEqual(state["ev"]["battery_capacity_kwh"], 84.0)
        self.assertGreaterEqual(state["ev"]["battery_soc_percent"], 0)
        self.assertLessEqual(state["ev"]["battery_soc_percent"], 100)
        self.assertGreaterEqual(state["ev"]["total_energy_consumed_kwh"], 0)

    def test_electric_health_is_exposed_without_mechanical_fields(self) -> None:
        state = get_vehicle_state(self.simulator)

        self.assertEqual(state["health"]["source"], "supplemental_simulation")
        self.assertTrue(state["health"]["is_estimated"])
        self.assertNotIn("fuel_percent", state["ev"])
        self.assertNotIn("engine_temp_c", state["ev"])
        self.assertNotIn("oil_life_percent", state["ev"])
        self.assertIn("battery_temperature_c", state["health"])
        self.assertIn("motor_temperature_c", state["health"])

    def test_trip_metadata_is_exposed_when_supported(self) -> None:
        self.simulator.get_state.return_value = {
            **SIMULATOR_STATE,
            "origin": {
                "name": "Route El Ain, central Sfax",
                "city": "Sfax",
                "country": "Tunisia",
                "latitude": 34.7518952,
                "longitude": 10.7296079,
            },
            "destination": {
                "name": "North-east central Sfax",
                "city": "Sfax",
                "country": "Tunisia",
                "latitude": 34.762,
                "longitude": 10.746,
            },
            "route": {
                "distance_km": 1.824,
                "duration_minutes": None,
                "uses_highway": False,
                "has_toll": False,
                "has_ferry": False,
            },
        }

        state = get_vehicle_state(self.simulator)

        self.assertEqual(
            state["trip"]["origin"]["name"],
            "Route El Ain, central Sfax",
        )
        self.assertEqual(
            state["trip"]["destination"]["name"],
            "North-east central Sfax",
        )
        self.assertEqual(state["trip"]["origin"]["city"], "Sfax")
        self.assertEqual(state["trip"]["route"]["distance_km"], 1.824)

    def test_missing_simulator_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "TripSimulator instance"):
            get_vehicle_state(None)  # type: ignore[arg-type]

    def test_unstarted_simulator_has_clear_error(self) -> None:
        self.simulator.get_state.side_effect = RuntimeError(
            "Call start() before requesting simulation state."
        )
        with self.assertRaisesRegex(VehicleStateError, "start the TripSimulator"):
            get_vehicle_state(self.simulator)

    def test_missing_ev_state_is_rejected(self) -> None:
        self.simulator.get_state.return_value = {
            "status": "running",
            "movement": SIMULATOR_STATE["movement"],
        }
        with self.assertRaisesRegex(VehicleStateError, "ev"):
            get_vehicle_state(self.simulator)

    def test_out_of_range_battery_soc_is_rejected(self) -> None:
        self.simulator.get_state.return_value = {
            **SIMULATOR_STATE,
            "ev": {**SIMULATOR_STATE["ev"], "battery_soc_percent": 101.0},
        }
        with self.assertRaisesRegex(VehicleStateError, "battery_soc_percent"):
            get_vehicle_state(self.simulator)

    def test_non_positive_battery_capacity_is_rejected(self) -> None:
        self.simulator.get_state.return_value = {
            **SIMULATOR_STATE,
            "ev": {**SIMULATOR_STATE["ev"], "battery_capacity_kwh": 0.0},
        }
        with self.assertRaisesRegex(VehicleStateError, "greater than zero"):
            get_vehicle_state(self.simulator)


class SumoVehicleToolIntegrationTests(unittest.TestCase):
    def test_vehicle_tool_adapts_live_ioniq5_state(self) -> None:
        from simulation.trip_simulator import TripSimulator

        simulator = TripSimulator(
            "Central Sfax origin",
            "Central Sfax destination",
            traffic_level="normal",
            initial_soc_percent=65.0,
        )
        try:
            simulator.start(use_gui=False)
            simulator.step(10)
            state = get_vehicle_state(simulator)

            self.assertEqual(
                state["ev"]["model"], "Hyundai IONIQ 5 Long Range RWD"
            )
            self.assertAlmostEqual(state["ev"]["battery_capacity_kwh"], 84.0)
            self.assertGreater(state["motion"]["speed_kmh"], 0)
            self.assertGreaterEqual(state["ev"]["battery_soc_percent"], 0)
            self.assertLessEqual(state["ev"]["battery_soc_percent"], 100)
            self.assertGreaterEqual(
                state["ev"]["total_energy_regenerated_kwh"], 0
            )
            self.assertIn("health", state)
            self.assertEqual(
                state["health"]["source"], "supplemental_simulation"
            )
            self.assertEqual(state["traffic"]["level"], "normal")
        finally:
            simulator.stop()


if __name__ == "__main__":
    unittest.main()
