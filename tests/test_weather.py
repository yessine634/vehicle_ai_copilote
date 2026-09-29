"""Tests for the Open-Meteo weather tool."""

import os
import unittest
from unittest.mock import Mock, patch

from tools.weather import get_weather, get_weather_description


class TestGetWeather(unittest.TestCase):
    """Verify geocoding and weather responses are handled correctly."""

    @patch("tools.weather.requests.get")
    def test_get_weather_returns_clean_data(self, mock_get: Mock) -> None:
        geocoding_response = Mock()
        geocoding_response.json.return_value = {
            "results": [
                {
                    "name": "Tunis",
                    "country": "Tunisia",
                    "latitude": 36.8065,
                    "longitude": 10.1815,
                }
            ]
        }

        weather_response = Mock()
        weather_response.json.return_value = {
            "current": {
                "temperature_2m": 24.5,
                "relative_humidity_2m": 55,
                "precipitation": 0.0,
                "rain": 0.0,
                "wind_speed_10m": 12.3,
                "weather_code": 1,
            }
        }

        mock_get.side_effect = [geocoding_response, weather_response]

        result = get_weather("Tunis")

        self.assertEqual(
            result,
            {
                "city": "Tunis",
                "country": "Tunisia",
                "latitude": 36.8065,
                "longitude": 10.1815,
                "temperature_c": 24.5,
                "humidity_percent": 55,
                "precipitation_mm": 0.0,
                "rain_mm": 0.0,
                "wind_speed_kmh": 12.3,
                "weather_code": 1,
                "weather_description": "Mainly clear",
            },
        )
        self.assertEqual(mock_get.call_count, 2)
        geocoding_response.raise_for_status.assert_called_once_with()
        weather_response.raise_for_status.assert_called_once_with()

        geocoding_call, weather_call = mock_get.call_args_list
        self.assertEqual(
            geocoding_call.args[0],
            "https://geocoding-api.open-meteo.com/v1/search",
        )
        self.assertEqual(geocoding_call.kwargs["params"]["name"], "Tunis")
        self.assertEqual(geocoding_call.kwargs["timeout"], 10)
        self.assertEqual(
            weather_call.args[0],
            "https://api.open-meteo.com/v1/forecast",
        )
        self.assertEqual(weather_call.kwargs["params"]["latitude"], 36.8065)
        self.assertEqual(weather_call.kwargs["params"]["longitude"], 10.1815)
        self.assertEqual(weather_call.kwargs["timeout"], 10)

    def test_weather_code_is_translated_to_plain_english(self) -> None:
        self.assertEqual(get_weather_description(3), "Overcast")
        self.assertEqual(get_weather_description(63), "Moderate rain")
        self.assertEqual(get_weather_description(95), "Thunderstorm")
        self.assertEqual(
            get_weather_description(999),
            "Unknown weather condition",
        )

    @patch("tools.weather.requests.get")
    def test_get_weather_returns_error_when_city_is_not_found(
        self, mock_get: Mock
    ) -> None:
        geocoding_response = Mock()
        geocoding_response.json.return_value = {"results": []}
        mock_get.return_value = geocoding_response

        result = get_weather("NotARealCity")

        self.assertEqual(result, {"error": "Location 'NotARealCity' not found."})
        mock_get.assert_called_once()
        geocoding_response.raise_for_status.assert_called_once_with()

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_WEATHER_TEST") == "1",
        "Set RUN_LIVE_WEATHER_TEST=1 to call the live Open-Meteo API.",
    )
    def test_live_weather_api(self) -> None:
        result = get_weather("Tunis")

        self.assertNotIn("error", result)
        self.assertEqual(result["city"], "Tunis")
        self.assertIsInstance(result["temperature_c"], (int, float))
        self.assertIsInstance(result["humidity_percent"], (int, float))
        self.assertIsInstance(result["wind_speed_kmh"], (int, float))
        self.assertIsInstance(result["weather_description"], str)


if __name__ == "__main__":
    unittest.main()
