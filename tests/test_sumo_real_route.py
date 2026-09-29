"""Integration test for the real El Ain-to-Teniour SUMO route."""

from __future__ import annotations

import unittest

from simulation.sumo_real_route import REAL_ROUTE_CONFIG
from simulation.sumo_runner import SumoTrafficSimulator


class SumoRealRouteTests(unittest.TestCase):
    def test_real_sfax_vehicle_enters_osm_road_network(self) -> None:
        simulator = SumoTrafficSimulator(REAL_ROUTE_CONFIG)
        try:
            initial_time = simulator.start(use_gui=False)
            vehicle_ids: list[str] = []
            for _ in range(10):
                simulator.step()
                vehicle_ids = simulator.get_vehicle_ids()
                if vehicle_ids:
                    break

            self.assertGreater(simulator.get_simulation_time(), initial_time)
            self.assertIn("ego_vehicle", vehicle_ids)

            state = simulator.get_vehicle_state("ego_vehicle")
            motion = state["motion"]
            self.assertEqual(state["vehicle_id"], "ego_vehicle")
            self.assertIsInstance(motion["speed_kmh"], float)
            self.assertTrue(motion["road_id"])
            self.assertTrue(motion["lane_id"])
        finally:
            simulator.stop()

        self.assertFalse(simulator.is_running)


if __name__ == "__main__":
    unittest.main()
