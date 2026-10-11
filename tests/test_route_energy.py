"""Tests for deterministic baseline route-energy estimation."""

import unittest

from simulation.route_energy import estimate_route_energy


def _segment(kind: str, distance_km: float) -> dict:
    return {
        "type": kind,
        "distance_km": distance_km,
        "average_speed_kmh": None,
    }


class RouteEnergyTests(unittest.TestCase):
    def test_each_road_type_uses_its_baseline(self) -> None:
        result = estimate_route_energy(
            route_segments=[
                _segment("urban", 10),
                _segment("suburban", 20),
                _segment("highway", 100),
            ],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )

        segments = result["route"]["segments"]
        self.assertEqual(segments[0]["consumption_kwh_100km"], 15.0)
        self.assertEqual(segments[1]["consumption_kwh_100km"], 17.0)
        self.assertEqual(segments[2]["consumption_kwh_100km"], 21.0)
        self.assertAlmostEqual(result["route"]["required_energy_kwh"], 25.9)

    def test_arrival_soc_uses_route_energy(self) -> None:
        result = estimate_route_energy(
            route_segments=[_segment("urban", 100)],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )

        self.assertAlmostEqual(
            result["arrival"]["expected_soc_percent"],
            (67 - 15) / 84 * 100,
            places=2,
        )

    def test_reserve_and_destination_checks_are_consistent(self) -> None:
        with_reserve = estimate_route_energy(
            route_segments=[_segment("highway", 250)],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )
        without_reserve = estimate_route_energy(
            route_segments=[_segment("highway", 260)],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )
        cannot_reach = estimate_route_energy(
            route_segments=[_segment("highway", 400)],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )

        self.assertTrue(with_reserve["feasibility"]["can_reach_with_reserve"])
        self.assertTrue(without_reserve["feasibility"]["can_reach_destination"])
        self.assertFalse(without_reserve["feasibility"]["can_reach_with_reserve"])
        self.assertFalse(cannot_reach["feasibility"]["can_reach_destination"])

    def test_unknown_segment_uses_fallback_consumption(self) -> None:
        result = estimate_route_energy(
            route_segments=[_segment("unknown", 100)],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )
        self.assertEqual(
            result["route"]["segments"][0]["consumption_kwh_100km"],
            16.0,
        )

    def test_zero_distance_route_is_safe(self) -> None:
        result = estimate_route_energy(
            route_segments=[],
            battery_capacity_kwh=84,
            current_energy_kwh=67,
        )
        self.assertEqual(result["route"]["required_energy_kwh"], 0.0)
        self.assertTrue(result["feasibility"]["can_reach_with_reserve"])


if __name__ == "__main__":
    unittest.main()
