"""Integration tests for the real TraCI-controlled SUMO demo."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from simulation.sumo_runner import SumoTrafficSimulator
from simulation.sumo_sfax import SCENARIO_DIRECTORY, SFAX_CONFIGS


class SumoTrafficSimulatorTests(unittest.TestCase):
    """Launch the non-GUI SUMO executable and inspect a moving vehicle."""

    def test_real_sumo_simulation_exposes_live_vehicle_state(self) -> None:
        simulator = SumoTrafficSimulator()
        vehicle_was_seen = False

        try:
            initial_time = simulator.start(use_gui=False)
            self.assertTrue(simulator.is_running)

            current_time = simulator.step()
            self.assertGreater(current_time, initial_time)

            for _ in range(10):
                vehicle_ids = simulator.get_vehicle_ids()
                if vehicle_ids:
                    vehicle_was_seen = True
                    state = simulator.get_vehicle_state(vehicle_ids[0])
                    motion = state["motion"]
                    self.assertIsInstance(state, dict)
                    self.assertIsInstance(motion["speed_mps"], (int, float))
                    self.assertIsInstance(motion["speed_kmh"], (int, float))
                    self.assertIn("road_id", motion)
                    self.assertIsInstance(motion["road_id"], str)
                    self.assertIsInstance(motion["road_name"], str)
                    self.assertFalse(motion["road_name"].startswith(":"))
                    break
                simulator.step()

            self.assertTrue(vehicle_was_seen, "No vehicle appeared in the SUMO demo.")
        finally:
            simulator.stop()

        self.assertFalse(simulator.is_running)

    def test_ioniq5_route_definition_enables_84_kwh_battery(self) -> None:
        route_file = Path(SCENARIO_DIRECTORY) / "sfax_ego.rou.xml"
        root = ET.parse(route_file).getroot()
        vehicle_type = root.find("vType")
        vehicle = root.find("vehicle")

        self.assertIsNotNone(vehicle_type)
        self.assertEqual(vehicle_type.get("id"), "ioniq5_long_range_rwd")
        self.assertEqual(vehicle_type.get("emissionClass"), "Energy/unknown")
        self.assertAlmostEqual(float(vehicle_type.get("maxSpeed")), 51.39)
        parameters = {
            item.get("key"): item.get("value")
            for item in vehicle_type.findall("param")
        }
        self.assertEqual(parameters["has.battery.device"], "true")
        self.assertEqual(float(parameters["device.battery.capacity"]), 84_000)
        self.assertEqual(float(parameters["maximumPower"]), 168_000)
        self.assertEqual(vehicle.get("type"), "ioniq5_long_range_rwd")

    def test_sfax_ioniq5_exposes_live_energy_and_configurable_soc(self) -> None:
        simulator = SumoTrafficSimulator(
            SFAX_CONFIGS["normal"],
            traffic_level="normal",
            initial_soc_percent=55.0,
        )
        try:
            simulator.start(use_gui=False)
            for _ in range(10):
                simulator.step()
                if "ego_vehicle" in simulator.get_vehicle_ids():
                    break

            self.assertIn("ego_vehicle", simulator.get_vehicle_ids())
            initial_state = simulator.get_vehicle_state("ego_vehicle")
            traffic = simulator.get_traffic_state("ego_vehicle")
            initial_energy = initial_state["ev"]["battery_energy_kwh"]

            state = initial_state
            for _ in range(200):
                simulator.step()
                if "ego_vehicle" not in simulator.get_vehicle_ids():
                    break
                state = simulator.get_vehicle_state("ego_vehicle")
                if state["ev"]["distance_travelled_km"] >= 1.0:
                    break

            self.assertIsInstance(state["motion"]["speed_kmh"], float)
            self.assertTrue(state["motion"]["road_id"])
            self.assertTrue(state["motion"]["road_name"])
            self.assertFalse(state["motion"]["road_name"].startswith(":"))
            self.assertTrue(state["motion"]["lane_id"])
            self.assertIn("position_geo", state["motion"])
            self.assertIn("ev", state)
            self.assertEqual(state["ev"]["model"], "Hyundai IONIQ 5 Long Range RWD")
            self.assertEqual(state["ev"]["battery_capacity_kwh"], 84.0)
            self.assertAlmostEqual(
                initial_state["ev"]["battery_soc_percent"], 55.0, places=5
            )
            self.assertNotEqual(state["ev"]["battery_energy_kwh"], initial_energy)
            self.assertIsInstance(
                state["ev"]["instant_consumption_wh_per_s"], float
            )
            self.assertGreaterEqual(state["ev"]["battery_soc_percent"], 0)
            self.assertLessEqual(state["ev"]["battery_soc_percent"], 100)
            self.assertGreaterEqual(state["ev"]["total_energy_consumed_kwh"], 0)
            self.assertGreaterEqual(
                state["ev"]["total_energy_regenerated_kwh"], 0
            )
            self.assertGreaterEqual(
                state["ev"]["estimated_remaining_range_km"], 0
            )
            self.assertIsNone(state["ev"]["charging_station_id"])
            self.assertFalse(state["ev"]["is_charging"])
            self.assertGreaterEqual(state["ev"]["distance_travelled_km"], 1.0)
            self.assertEqual(
                state["ev"]["range_estimate_source"], "observed_consumption"
            )
            self.assertEqual(traffic["traffic_level"], "normal")
            self.assertGreaterEqual(traffic["active_vehicles"], 1)
            self.assertIsInstance(traffic["average_speed_kph"], float)
        finally:
            simulator.stop()

        self.assertFalse(simulator.is_running)


@unittest.skipUnless(
    os.getenv("RUN_SUMO_GUI_DEMO") == "1",
    "Set RUN_SUMO_GUI_DEMO=1 to run the visible SUMO GUI integration test.",
)
class SumoGuiDemoTests(unittest.TestCase):
    """Optionally verify that the GUI executable accepts TraCI commands."""

    def test_gui_starts_steps_and_stops(self) -> None:
        simulator = SumoTrafficSimulator()
        try:
            initial_time = simulator.start(use_gui=True)
            self.assertGreater(simulator.step(), initial_time)
        finally:
            simulator.stop()


if __name__ == "__main__":
    unittest.main()
