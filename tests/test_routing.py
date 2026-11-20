"""Tests for the Open-Meteo and Valhalla routing tool."""

import os
import unittest
from unittest.mock import Mock, patch

import requests

from location.resolver import (
    LocationNotFoundError,
    LocationResolver,
    ResolvedLocation,
)
from tools.routing import (
    DEFAULT_ROAD_CLASSIFICATION,
    REQUEST_TIMEOUT,
    VALHALLA_ROUTE_URL,
    VALHALLA_TRACE_ATTRIBUTES_URL,
    build_route_segments,
    classify_road,
    get_route,
    merge_consecutive_segments,
)


def _json_response(data: dict) -> Mock:
    """Build a successful mocked HTTP response containing JSON data."""

    response = Mock()
    response.json.return_value = data
    return response


def _resolved(
    raw_input: str,
    canonical_name: str,
    latitude: float,
    longitude: float,
) -> ResolvedLocation:
    return ResolvedLocation(
        raw_input=raw_input,
        canonical_name=canonical_name,
        latitude=latitude,
        longitude=longitude,
        source="nominatim",
        confidence=0.95,
    )


def _resolver(*locations: ResolvedLocation) -> Mock:
    resolver = Mock(spec=LocationResolver)
    resolver.resolve.side_effect = list(locations)
    return resolver


class TestGetRoute(unittest.TestCase):
    """Verify route lookup behavior without requiring internet access."""

    @patch("tools.routing.requests.post")
    def test_get_route_returns_clean_route_data(self, mock_post: Mock) -> None:
        resolver = _resolver(
            _resolved("Sfax", "Sfax, Tunisia", 34.7406, 10.7603),
            _resolved("Tunis", "Tunis, Tunisia", 36.8065, 10.1815),
        )
        route_response = _json_response(
            {
                "trip": {
                    "summary": {
                        "length": 269.347,
                        "time": 10_260,
                        "has_highway": True,
                        "has_toll": False,
                        "has_ferry": False,
                    }
                }
            }
        )
        mock_post.return_value = route_response

        result = get_route("Sfax", "Tunis", location_resolver=resolver)

        self.assertEqual(
            result,
            {
                "origin": {
                    "raw_input": "Sfax",
                    "canonical_name": "Sfax, Tunisia",
                    "name": "Sfax, Tunisia",
                    "latitude": 34.7406,
                    "longitude": 10.7603,
                    "source": "nominatim",
                    "confidence": 0.95,
                },
                "destination": {
                    "raw_input": "Tunis",
                    "canonical_name": "Tunis, Tunisia",
                    "name": "Tunis, Tunisia",
                    "latitude": 36.8065,
                    "longitude": 10.1815,
                    "source": "nominatim",
                    "confidence": 0.95,
                },
                "distance_km": 269.35,
                "duration_minutes": 171.0,
                "uses_highway": True,
                "has_toll": False,
                "has_ferry": False,
                "segments": [],
            },
        )

        self.assertEqual(
            [call.args[0] for call in resolver.resolve.call_args_list],
            ["Sfax", "Tunis"],
        )
        route_response.raise_for_status.assert_called_once_with()

        route_call = mock_post.call_args
        self.assertEqual(route_call.args[0], VALHALLA_ROUTE_URL)
        self.assertEqual(route_call.kwargs["timeout"], REQUEST_TIMEOUT)
        self.assertEqual(route_call.kwargs["json"]["costing"], "auto")
        self.assertEqual(route_call.kwargs["json"]["units"], "kilometers")
        self.assertEqual(
            route_call.kwargs["json"]["locations"],
            [
                {"lat": 34.7406, "lon": 10.7603},
                {"lat": 36.8065, "lon": 10.1815},
            ],
        )

    @patch("tools.routing.requests.post")
    def test_get_route_rejects_unknown_origin(self, mock_post: Mock) -> None:
        resolver = Mock(spec=LocationResolver)
        resolver.resolve.side_effect = LocationNotFoundError(
            "Location 'Unknown Place' was not found."
        )

        with self.assertRaisesRegex(
            LocationNotFoundError, "Location 'Unknown Place' was not found"
        ):
            get_route(
                "Unknown Place",
                "Tunis",
                location_resolver=resolver,
            )

        resolver.resolve.assert_called_once_with("Unknown Place")
        mock_post.assert_not_called()

    @patch("tools.routing.requests.post")
    def test_get_route_preserves_full_comma_qualified_place(
        self, mock_post: Mock
    ) -> None:
        destination_query = "Houmt Souk, Djerba, Tunisia"
        resolver = _resolver(
            _resolved("Tunis", "Tunis, Tunisia", 36.8065, 10.1815),
            _resolved(
                destination_query,
                "Houmt Souk, Djerba, Tunisia",
                33.8758,
                10.8575,
            ),
        )
        mock_post.return_value = _json_response(
            {
                "trip": {
                    "summary": {
                        "length": 520.4,
                        "time": 21_600,
                        "has_highway": True,
                        "has_toll": True,
                        "has_ferry": True,
                    }
                }
            }
        )

        result = get_route(
            "Tunis",
            destination_query,
            location_resolver=resolver,
        )

        self.assertEqual(
            result["destination"]["canonical_name"],
            "Houmt Souk, Djerba, Tunisia",
        )
        self.assertEqual(
            resolver.resolve.call_args_list[1].args[0],
            destination_query,
        )

    @patch("tools.routing.requests.post")
    def test_get_route_converts_routing_request_errors(
        self, mock_post: Mock
    ) -> None:
        resolver = _resolver(
            _resolved("Sfax", "Sfax, Tunisia", 34.7406, 10.7603),
            _resolved("Tunis", "Tunis, Tunisia", 36.8065, 10.1815),
        )
        mock_post.side_effect = requests.Timeout("request timed out")

        with self.assertRaisesRegex(
            RuntimeError, "Valhalla routing request failed"
        ):
            get_route("Sfax", "Tunis", location_resolver=resolver)

    @patch("tools.routing.requests.post")
    def test_get_route_rejects_unexpected_response(
        self, mock_post: Mock
    ) -> None:
        resolver = _resolver(
            _resolved("Sfax", "Sfax, Tunisia", 34.7406, 10.7603),
            _resolved("Tunis", "Tunis, Tunisia", 36.8065, 10.1815),
        )
        mock_post.return_value = _json_response({"status": "ok"})

        with self.assertRaisesRegex(
            RuntimeError, "Valhalla returned an unexpected response format"
        ):
            get_route("Sfax", "Tunis", location_resolver=resolver)

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_ROUTING_TEST") == "1",
        "Set RUN_LIVE_ROUTING_TEST=1 to call the live APIs.",
    )
    def test_live_route_from_sfax_to_tunis(self) -> None:
        route = get_route("Sfax", "Tunis")

        self.assertIn("Sfax", route["origin"]["canonical_name"])
        self.assertIn("Tunis", route["destination"]["canonical_name"])
        self.assertGreater(route["distance_km"], 0)
        self.assertGreater(route["duration_minutes"], 0)
        self.assertTrue(route["segments"])
        self.assertIn(
            "highway",
            {segment["type"] for segment in route["segments"]},
        )
        self.assertAlmostEqual(
            sum(segment["distance_km"] for segment in route["segments"]),
            route["distance_km"],
            delta=1.0,
        )


class TestRouteSegmentation(unittest.TestCase):
    """Verify deterministic road classification and segment merging."""

    def test_classifies_urban_edge(self) -> None:
        self.assertEqual(
            classify_road("residential", 50, 45),
            "urban",
        )

    def test_classifies_suburban_edge(self) -> None:
        self.assertEqual(
            classify_road("primary", 70, 65),
            "suburban",
        )

    def test_classifies_highway_by_class_or_speed(self) -> None:
        self.assertEqual(classify_road("motorway", 70, 70), "highway")
        self.assertEqual(classify_road("primary", 100, 90), "highway")

    def test_missing_speed_values_are_safe(self) -> None:
        self.assertEqual(classify_road("residential", None, None), "urban")
        self.assertEqual(classify_road("primary", None, None), "suburban")
        self.assertEqual(classify_road("motorway", None, None), "highway")
        self.assertEqual(classify_road(None, None, None), "suburban")

    def test_merges_consecutive_segments(self) -> None:
        merged = merge_consecutive_segments(
            [
                {"distance_km": 3, "type": "urban"},
                {"distance_km": 5, "type": "suburban"},
                {"distance_km": 120, "type": "highway"},
                {"distance_km": 80, "type": "highway"},
                {"distance_km": 10, "type": "urban"},
            ]
        )

        self.assertEqual(
            [(segment["type"], segment["distance_km"]) for segment in merged],
            [
                ("urban", 3.0),
                ("suburban", 5.0),
                ("highway", 200.0),
                ("urban", 10.0),
            ],
        )

    def test_does_not_merge_incompatible_posted_limits(self) -> None:
        merged = merge_consecutive_segments(
            [
                {
                    "distance_km": 5,
                    "type": "highway",
                    "speed_limit_kmh": 110,
                    "use": "road",
                },
                {
                    "distance_km": 2,
                    "type": "highway",
                    "speed_limit_kmh": 70,
                    "use": "ramp",
                },
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["posted_speed_limit_kmh"], 110.0)
        self.assertEqual(merged[1]["posted_speed_limit_kmh"], 70.0)

    def test_builds_mixed_urban_highway_route(self) -> None:
        segments = build_route_segments(
            [
                {
                    "distance_km": 4,
                    "road_class": "residential",
                    "speed_limit_kmh": 50,
                    "routing_speed_kmh": 45,
                    "road_name": "Urban Road",
                    "use": "road",
                },
                {
                    "distance_km": 100,
                    "road_class": "motorway",
                    "speed_limit_kmh": 110,
                    "routing_speed_kmh": 100,
                    "road_name": "A1",
                    "use": "road",
                },
            ]
        )

        self.assertEqual([item["type"] for item in segments], ["urban", "highway"])
        self.assertEqual(segments[0]["average_speed_kmh"], 45.0)
        self.assertEqual(segments[1]["road_names"], ["A1"])
        self.assertEqual(segments[1]["posted_speed_limit_kmh"], 110.0)

    @patch("tools.routing.requests.post")
    def test_get_route_adds_segments_from_trace_attributes(
        self,
        mock_post: Mock,
    ) -> None:
        resolver = _resolver(
            _resolved("Sfax", "Sfax, Tunisia", 34.7406, 10.7603),
            _resolved("Tunis", "Tunis, Tunisia", 36.8065, 10.1815),
        )
        route_response = _json_response(
            {
                "trip": {
                    "summary": {
                        "length": 218,
                        "time": 9_000,
                        "has_highway": True,
                        "has_toll": True,
                        "has_ferry": False,
                    },
                    "legs": [{"shape": "encoded-route-shape"}],
                }
            }
        )
        trace_response = _json_response(
            {
                "edges": [
                    {
                        "length": 3,
                        "road_class": "residential",
                        "use": "road",
                        "speed_limit": 50,
                        "speed": 45,
                        "names": ["Sfax Street"],
                    },
                    {
                        "length": 5,
                        "road_class": "primary",
                        "use": "road",
                        "speed_limit": 70,
                        "speed": 65,
                        "names": ["Regional Road"],
                    },
                    {
                        "length": 120,
                        "road_class": "motorway",
                        "use": "road",
                        "speed_limit": 110,
                        "speed": 100,
                        "names": ["A1"],
                    },
                    {
                        "length": 80,
                        "road_class": "trunk",
                        "use": "road",
                        "speed_limit": 110,
                        "speed": 110,
                        "names": ["A1"],
                    },
                    {
                        "length": 10,
                        "road_class": "residential",
                        "use": "road",
                        "speed": 50,
                        "names": ["Tunis Street"],
                    },
                ]
            }
        )
        mock_post.side_effect = [route_response, trace_response]

        result = get_route(
            "Sfax",
            "Tunis",
            location_resolver=resolver,
        )

        self.assertEqual(
            [
                (segment["type"], segment["distance_km"])
                for segment in result["segments"]
            ],
            [
                ("urban", 3.0),
                ("suburban", 5.0),
                ("highway", 200.0),
                ("urban", 10.0),
            ],
        )
        self.assertEqual(result["segments"][2]["average_speed_kmh"], 104.0)
        self.assertEqual(result["segments"][2]["road_names"], ["A1"])
        self.assertEqual(mock_post.call_count, 2)

        trace_call = mock_post.call_args_list[1]
        self.assertEqual(trace_call.args[0], VALHALLA_TRACE_ATTRIBUTES_URL)
        self.assertEqual(
            trace_call.kwargs["json"]["encoded_polyline"],
            "encoded-route-shape",
        )
        self.assertEqual(trace_call.kwargs["json"]["shape_match"], "edge_walk")
        self.assertIn(
            "edge.speed_limit",
            trace_call.kwargs["json"]["filters"]["attributes"],
        )

    def test_thresholds_are_configurable(self) -> None:
        custom = type(DEFAULT_ROAD_CLASSIFICATION)(
            urban_max_speed_kmh=40,
            highway_min_speed_kmh=80,
        )
        self.assertEqual(classify_road("primary", 50, 50, custom), "suburban")
        self.assertEqual(classify_road("primary", 80, 80, custom), "highway")


if __name__ == "__main__":
    unittest.main()
