"""Tests for road-network preparation."""

import unittest

from simulation.osm_network import (
    build_road_network,
    distance_km,
    get_route_nodes,
    simplify_route,
)


class TestOsmNetwork(unittest.TestCase):
    def test_distance_km_uses_latitude_and_longitude(self) -> None:
        distance = distance_km((0.0, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(distance, 111.2, places=1)

    def test_simplify_route_preserves_endpoints(self) -> None:
        points = [(34.75, 10.72), (34.751, 10.721), (34.76, 10.73)]
        simplified = simplify_route(points, spacing_km=0.5)
        self.assertEqual(simplified[0], points[0])
        self.assertEqual(simplified[-1], points[-1])

    def test_build_road_network_creates_nodes_and_links(self) -> None:
        points = [(34.75, 10.72), (34.755, 10.725), (34.76, 10.73)]
        graph = build_road_network(
            points,
            route_distance_km=2.0,
            duration_minutes=5.0,
            spacing_km=0.1,
            lanes=2,
        )

        route_nodes = get_route_nodes(graph)
        self.assertGreaterEqual(len(route_nodes), 2)
        self.assertEqual(graph.number_of_edges(), len(route_nodes) - 1)
        edge = graph.get_edge_data(route_nodes[0], route_nodes[1])[0]
        self.assertEqual(edge["lanes"], 2)
        self.assertGreater(edge["length"], 0)
        self.assertGreater(edge["speed_kph"], 0)

    def test_build_road_network_validates_inputs(self) -> None:
        with self.assertRaises(ValueError):
            build_road_network(
                [(34.75, 10.72)],
                route_distance_km=1.0,
                duration_minutes=5.0,
            )


if __name__ == "__main__":
    unittest.main()
