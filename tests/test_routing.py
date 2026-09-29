"""Tests for the Open-Meteo and Valhalla routing tool."""

import os
import unittest
from unittest.mock import Mock, patch

import requests

from tools.routing import (
    GEOCODING_URL,
    REQUEST_TIMEOUT,
    VALHALLA_ROUTE_URL,
    get_route,
)


def _json_response(data: dict) -> Mock:
    """Build a successful mocked HTTP response containing JSON data."""

    response = Mock()
    response.json.return_value = data
    return response


class TestGetRoute(unittest.TestCase):
    """Verify route lookup behavior without requiring internet access."""

    @patch("tools.routing.requests.post")
    @patch("tools.routing.requests.get")
    def test_get_route_returns_clean_route_data(
        self, mock_get: Mock, mock_post: Mock
    ) -> None:
        origin_response = _json_response(
            {
                "results": [
                    {
                        "name": "Sfax",
                        "country": "Tunisia",
                        "latitude": 34.7406,
                        "longitude": 10.7603,
                    }
                ]
            }
        )
        destination_response = _json_response(
            {
                "results": [
                    {
                        "name": "Tunis",
                        "country": "Tunisia",
                        "latitude": 36.8065,
                        "longitude": 10.1815,
                    }
                ]
            }
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
        mock_get.side_effect = [origin_response, destination_response]
        mock_post.return_value = route_response

        result = get_route("Sfax", "Tunis")

        self.assertEqual(
            result,
            {
                "origin": {
                    "name": "Sfax",
                    "country": "Tunisia",
                    "latitude": 34.7406,
                    "longitude": 10.7603,
                },
                "destination": {
                    "name": "Tunis",
                    "country": "Tunisia",
                    "latitude": 36.8065,
                    "longitude": 10.1815,
                },
                "distance_km": 269.35,
                "duration_minutes": 171.0,
                "uses_highway": True,
                "has_toll": False,
                "has_ferry": False,
            },
        )

        self.assertEqual(mock_get.call_count, 2)
        for response in (origin_response, destination_response, route_response):
            response.raise_for_status.assert_called_once_with()

        origin_call, destination_call = mock_get.call_args_list
        self.assertEqual(origin_call.args[0], GEOCODING_URL)
        self.assertEqual(origin_call.kwargs["params"]["name"], "Sfax")
        self.assertEqual(destination_call.kwargs["params"]["name"], "Tunis")
        self.assertEqual(origin_call.kwargs["timeout"], REQUEST_TIMEOUT)

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
    @patch("tools.routing.requests.get")
    def test_get_route_rejects_unknown_origin(
        self, mock_get: Mock, mock_post: Mock
    ) -> None:
        mock_get.return_value = _json_response({"results": []})

        with self.assertRaisesRegex(
            ValueError, "Location 'Unknown Place' was not found"
        ):
            get_route("Unknown Place", "Tunis")

        mock_get.assert_called_once()
        mock_post.assert_not_called()

    @patch("tools.routing.requests.post")
    @patch("tools.routing.requests.get")
    def test_get_route_retries_comma_qualified_place_with_short_name(
        self, mock_get: Mock, mock_post: Mock
    ) -> None:
        mock_get.side_effect = [
            _json_response(
                {
                    "results": [
                        {
                            "name": "Tunis",
                            "country": "Tunisia",
                            "latitude": 36.8065,
                            "longitude": 10.1815,
                        }
                    ]
                }
            ),
            _json_response({"results": []}),
            _json_response(
                {
                    "results": [
                        {
                            "name": "Houmt Souk",
                            "country": "Tunisia",
                            "latitude": 33.8758,
                            "longitude": 10.8575,
                        }
                    ]
                }
            ),
        ]
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

        result = get_route("Tunis", "Houmt Souk, Djerba, Tunisia")

        self.assertEqual(result["destination"]["name"], "Houmt Souk")
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(
            mock_get.call_args_list[1].kwargs["params"]["name"],
            "Houmt Souk, Djerba, Tunisia",
        )
        self.assertEqual(
            mock_get.call_args_list[2].kwargs["params"]["name"],
            "Houmt Souk",
        )

    @patch("tools.routing.requests.post")
    @patch("tools.routing.requests.get")
    def test_get_route_converts_routing_request_errors(
        self, mock_get: Mock, mock_post: Mock
    ) -> None:
        mock_get.side_effect = [
            _json_response(
                {
                    "results": [
                        {
                            "name": "Sfax",
                            "country": "Tunisia",
                            "latitude": 34.7406,
                            "longitude": 10.7603,
                        }
                    ]
                }
            ),
            _json_response(
                {
                    "results": [
                        {
                            "name": "Tunis",
                            "country": "Tunisia",
                            "latitude": 36.8065,
                            "longitude": 10.1815,
                        }
                    ]
                }
            ),
        ]
        mock_post.side_effect = requests.Timeout("request timed out")

        with self.assertRaisesRegex(
            RuntimeError, "Valhalla routing request failed"
        ):
            get_route("Sfax", "Tunis")

    @patch("tools.routing.requests.post")
    @patch("tools.routing.requests.get")
    def test_get_route_rejects_unexpected_response(
        self, mock_get: Mock, mock_post: Mock
    ) -> None:
        mock_get.side_effect = [
            _json_response(
                {
                    "results": [
                        {
                            "name": "Sfax",
                            "country": "Tunisia",
                            "latitude": 34.7406,
                            "longitude": 10.7603,
                        }
                    ]
                }
            ),
            _json_response(
                {
                    "results": [
                        {
                            "name": "Tunis",
                            "country": "Tunisia",
                            "latitude": 36.8065,
                            "longitude": 10.1815,
                        }
                    ]
                }
            ),
        ]
        mock_post.return_value = _json_response({"status": "ok"})

        with self.assertRaisesRegex(
            RuntimeError, "Valhalla returned an unexpected response format"
        ):
            get_route("Sfax", "Tunis")

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_ROUTING_TEST") == "1",
        "Set RUN_LIVE_ROUTING_TEST=1 to call the live APIs.",
    )
    def test_live_route_from_sfax_to_tunis(self) -> None:
        route = get_route("Sfax", "Tunis")

        self.assertEqual(route["origin"]["name"], "Sfax")
        self.assertEqual(route["destination"]["name"], "Tunis")
        self.assertGreater(route["distance_km"], 0)
        self.assertGreater(route["duration_minutes"], 0)


if __name__ == "__main__":
    unittest.main()
