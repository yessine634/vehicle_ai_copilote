"""Tests for the SUMO-only combined trip simulator."""

import unittest

from simulation.trip_simulator import TripSimulator


class TestTripSimulator(unittest.TestCase):
    def test_start_and_step_combine_sumo_ev_health_and_traffic(self) -> None:
        simulator = TripSimulator(
            "Central Sfax origin",
            "Central Sfax destination",
            traffic_level="normal",
        )
        try:
            initial_state = simulator.start(use_gui=False)
            moving_state = simulator.step(10)

            self.assertEqual(initial_state["simulation_backend"], "sumo")
            self.assertGreater(moving_state["movement"]["simulation_time_s"], 0)
            self.assertTrue(moving_state["movement"]["road_id"])
            self.assertTrue(moving_state["movement"]["road_name"])
            self.assertFalse(
                moving_state["movement"]["current_road"].startswith(":")
            )
            self.assertTrue(moving_state["movement"]["lane_id"])
            self.assertEqual(
                moving_state["ev"]["model"],
                "Hyundai IONIQ 5 Long Range RWD",
            )
            self.assertGreaterEqual(moving_state["ev"]["battery_soc_percent"], 0)
            self.assertLessEqual(moving_state["ev"]["battery_soc_percent"], 100)
            self.assertEqual(
                moving_state["health"]["source"],
                "supplemental_simulation",
            )
            self.assertEqual(moving_state["traffic"]["scenario"], "normal")
            self.assertNotIn("vehicle_health", moving_state)
        finally:
            simulator.stop()

        self.assertFalse(simulator.sumo.is_running)

    def test_invalid_sumo_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TripSimulator("origin", "destination", traffic_level="invalid")
        with self.assertRaises(ValueError):
            TripSimulator("origin", "destination", initial_soc_percent=101)


if __name__ == "__main__":
    unittest.main()
