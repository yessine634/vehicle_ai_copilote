"""Tests for validated OSM/Nominatim location resolution."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock

from location.resolver import (
    AmbiguousLocationError,
    InvalidCoordinatesError,
    LocationNotFoundError,
    LocationResolver,
)


def _candidate(
    display_name: str,
    latitude: float,
    longitude: float,
    *,
    country_code: str = "tn",
    address: dict | None = None,
    category: str | None = None,
    place_type: str | None = None,
) -> dict:
    candidate = {
        "display_name": display_name,
        "lat": str(latitude),
        "lon": str(longitude),
        "address": {"country_code": country_code, **(address or {})},
        "namedetails": {"name": display_name.split(",", maxsplit=1)[0]},
    }
    if category is not None:
        candidate["category"] = category
    if place_type is not None:
        candidate["type"] = place_type
    return candidate


def _http_client(payload) -> tuple[Mock, Mock]:
    response = Mock()
    response.json.return_value = payload
    client = Mock()
    client.get.return_value = response
    return client, response


def _resolver_for(payload, **kwargs) -> tuple[LocationResolver, Mock, Mock]:
    client, response = _http_client(payload)
    resolver = LocationResolver(
        http_client=client,
        minimum_request_interval_seconds=0,
        **kwargs,
    )
    return resolver, client, response


class LocationResolverTests(unittest.TestCase):
    def test_tunis_city_center_rejects_hammamet_business(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("Tunis City Center, Hammamet, Nabeul, Tunisia", 36.40, 10.62,
                       address={"city": "Hammamet", "state": "Nabeul"},
                       category="shop", place_type="mall"),
        ])
        with self.assertRaises(LocationNotFoundError):
            resolver.resolve("Tunis city center")

    def test_tunis_city_selects_geographic_candidate(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("Tunis City Centre, Hammamet, Tunisia", 36.40, 10.62,
                       address={"city": "Hammamet", "state": "Nabeul"},
                       category="shop", place_type="mall"),
            _candidate("Tunis, Tunis, Tunisia", 36.80, 10.18,
                       address={"city": "Tunis", "state": "Tunis"},
                       category="place", place_type="city"),
        ])
        self.assertEqual(resolver.resolve("Tunis").latitude, 36.80)

    def test_tunis_city_centre_uses_city_not_same_city_shop(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("Tunis City Centre Shop, Tunis, Tunisia", 36.81, 10.19,
                       address={"city": "Tunis", "state": "Tunis"},
                       category="shop", place_type="mall"),
            _candidate("Tunis, Wilaya Tunis, Tunisia", 36.80, 10.18,
                       address={"city": "تونس", "state": "ولاية تونس"},
                       category="place", place_type="city"),
        ])
        self.assertEqual(resolver.resolve("Tunis city center").latitude, 36.80)

    def test_other_city_centre_is_not_limited_to_known_aliases(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("Sousse, Tunisia", 35.82, 10.64,
                       address={"city": "Sousse", "state": "Sousse"},
                       category="place", place_type="city"),
        ])
        self.assertEqual(resolver.resolve("Sousse city centre").latitude, 35.82)

    def test_requested_hammamet_business_remains_valid(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("City Centre Mall, Hammamet, Nabeul, Tunisia", 36.40, 10.62,
                       address={"city": "Hammamet", "state": "Nabeul"},
                       category="shop", place_type="mall"),
        ])
        self.assertEqual(resolver.resolve("City Centre Mall, Hammamet").source, "nominatim")

    def test_explicit_city_rejects_conflicting_administration(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("Route El Ain, Hammamet, Tunisia", 36.40, 10.62,
                       address={"city": "Hammamet", "state": "Nabeul"},
                       category="highway", place_type="residential"),
        ])
        with self.assertRaises(LocationNotFoundError):
            resolver.resolve("Route El Ain, Sfax")

    def test_city_centre_without_city_is_not_guessed(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("City Centre Mall, Hammamet, Tunisia", 36.40, 10.62,
                       address={"city": "Hammamet"}, category="shop", place_type="mall")
        ])
        with self.assertRaises(LocationNotFoundError):
            resolver.resolve("city center")

    def test_nearby_subdivision_with_requested_region_is_valid(self) -> None:
        resolver, _, _ = _resolver_for([
            _candidate("ENETCom, Sakiet Ezzit, Sfax, Tunisia", 34.84, 10.75,
                       address={"town": "Sakiet Ezzit", "state": "Sfax"},
                       category="amenity", place_type="university")
        ])
        self.assertEqual(resolver.resolve("ENETCom, Sfax").source, "trusted_location")
        self.assertEqual(resolver.resolve("ENETCom campus, Sfax").source, "nominatim")

    def test_nominatim_request_throttling_is_preserved(self) -> None:
        client, _ = _http_client([
            _candidate("Gremda, Sfax, Tunisia", 34.75, 10.72)
        ])
        times = iter([10.0, 10.0, 10.2, 11.0])
        sleeper = Mock()
        resolver = LocationResolver(http_client=client, clock=lambda: next(times),
                                    sleeper=sleeper, minimum_request_interval_seconds=1.0)
        resolver.resolve("Gremda, Sfax")
        resolver.resolve("Gremda, Sfax, Tunisia")
        sleeper.assert_called_once()
        self.assertAlmostEqual(sleeper.call_args.args[0], 0.8)

    def test_route_el_ain_km4_uses_trusted_project_coordinates(self) -> None:
        resolver, client, _ = _resolver_for([])

        result = resolver.resolve(
            "Route El Ain Km 4, Sfax, Tunisia"
        )

        self.assertEqual(result.source, "trusted_location")
        self.assertEqual(result.latitude, 34.7518952)
        self.assertEqual(result.longitude, 10.7296079)
        client.get.assert_not_called()

    def test_enetcom_alias_uses_same_coordinates_as_sumo(self) -> None:
        resolver, client, _ = _resolver_for([])

        result = resolver.resolve(
            "ENET'Com, Cité El Ons, Sakiet Ezzit, Sfax, Tunisia"
        )

        self.assertEqual(result.source, "trusted_location")
        self.assertEqual(result.latitude, 34.839620)
        self.assertEqual(result.longitude, 10.756662)
        client.get.assert_not_called()

    def test_resolves_direct_coordinates_without_network(self) -> None:
        resolver, client, _ = _resolver_for([])

        result = resolver.resolve("34.7518952, 10.7296079")

        self.assertEqual(result.source, "coordinates")
        self.assertEqual(result.latitude, 34.7518952)
        self.assertEqual(result.longitude, 10.7296079)
        self.assertEqual(result.confidence, 1.0)
        client.get.assert_not_called()

    def test_rejects_malformed_or_out_of_range_coordinates(self) -> None:
        resolver, client, _ = _resolver_for([])

        with self.assertRaises(InvalidCoordinatesError):
            resolver.resolve("34.7518952, not-a-longitude")
        with self.assertRaises(InvalidCoordinatesError):
            resolver.resolve("91, 10.7296079")

        client.get.assert_not_called()

    def test_resolves_local_sfax_examples(self) -> None:
        cases = [
            (
                "Gremda, Sfax, Tunisia",
                "Gremda, Sfax, Tunisia",
                34.75,
                10.72,
            ),
            (
                "Route El Ain, Sfax, Tunisia",
                "Route El Aïn, Sfax, Tunisia",
                34.760682,
                10.718643,
            ),
        ]

        for query, canonical, latitude, longitude in cases:
            with self.subTest(query=query):
                resolver, client, response = _resolver_for(
                    [_candidate(canonical, latitude, longitude)]
                )
                result = resolver.resolve(query)

                self.assertEqual(result.raw_input, query)
                self.assertEqual(result.canonical_name, canonical)
                self.assertEqual(result.source, "nominatim")
                self.assertGreater(result.confidence or 0, 0)
                request = client.get.call_args
                self.assertEqual(request.kwargs["params"]["q"], query)
                self.assertEqual(request.kwargs["params"]["limit"], 5)
                self.assertEqual(
                    request.kwargs["params"]["countrycodes"], "tn"
                )
                self.assertEqual(
                    request.kwargs["headers"]["User-Agent"],
                    "vehicle-ai-copilot/1.0",
                )
                response.raise_for_status.assert_called_once_with()

    def test_unknown_location_raises_clear_error(self) -> None:
        resolver, _, _ = _resolver_for([])

        with self.assertRaisesRegex(
            LocationNotFoundError,
            "was not found",
        ):
            resolver.resolve("Unknown Place, Sfax, Tunisia")

    def test_rejects_candidate_outside_sfax_area(self) -> None:
        resolver, _, _ = _resolver_for(
            [_candidate("Wrong Place, Tunis, Tunisia", 36.8065, 10.1815)]
        )

        with self.assertRaises(LocationNotFoundError):
            resolver.resolve("Local Place, Sfax, Tunisia")

    def test_rejects_non_tunisian_candidate(self) -> None:
        resolver, _, _ = _resolver_for(
            [
                _candidate(
                    "Sfax, Somewhere Else",
                    34.7406,
                    10.7603,
                    country_code="fr",
                )
            ]
        )

        with self.assertRaises(LocationNotFoundError):
            resolver.resolve("Sfax")

    def test_ambiguous_candidates_are_not_silently_guessed(self) -> None:
        resolver, _, _ = _resolver_for(
            [
                _candidate("Test Place A, Sfax, Tunisia", 34.75, 10.75),
                _candidate("Test Place B, Sfax, Tunisia", 34.75, 10.75),
            ],
            ambiguity_threshold=0.05,
        )

        with self.assertRaises(AmbiguousLocationError) as context:
            resolver.resolve("Test Place, Sfax")

        self.assertEqual(len(context.exception.candidates), 2)
        self.assertIn("Test Place A", str(context.exception))
        self.assertIn("Test Place B", str(context.exception))

    def test_repeated_normalized_query_uses_memory_cache(self) -> None:
        resolver, client, _ = _resolver_for(
            [_candidate("Gremda, Sfax, Tunisia", 34.75, 10.72)]
        )

        first = resolver.resolve("Gremda,   Sfax")
        second = resolver.resolve("  gremda, sfax  ")

        self.assertEqual(client.get.call_count, 1)
        self.assertEqual(first.latitude, second.latitude)
        self.assertEqual(second.raw_input, "gremda, sfax")


@unittest.skipUnless(
    os.getenv("RUN_LIVE_LOCATION_TEST") == "1",
    "Set RUN_LIVE_LOCATION_TEST=1 to call public Nominatim.",
)
class LiveLocationResolverTests(unittest.TestCase):
    def test_print_resolved_locations(self) -> None:
        resolver = LocationResolver()
        for query in (
            "Gremda, Sfax, Tunisia",
            "ENET'Com, Sfax, Tunisia",
            "Route El Ain, Sfax, Tunisia",
            "34.7518952, 10.7296079",
        ):
            result = resolver.resolve(query)
            print(
                f"{query} -> {result.canonical_name}: "
                f"{result.latitude:.7f}, {result.longitude:.7f}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
