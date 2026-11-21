"""Deterministic tests for the segment-aware EV energy subgraph."""

from __future__ import annotations

from copy import deepcopy
import unittest

from simulation.ev_config import BMW_I3_120_AH, SUMO_EV_MODEL_APPROXIMATIONS
from simulation.energy_estimation import (
    DEFAULT_MODEL_PATH,
    FEATURES,
    INDIA_MODEL_TARGET,
    TRAFFIC_CONDITION_CODES,
    SavedEnergyModel,
    build_energy_estimation_graph,
    build_ml_features,
    energy_result,
    get_legal_speed_limit,
)


def vehicle_state(
    *,
    soc_percent: float = 80.0,
    energy_kwh: float | None = 33.76,
    traffic_level: str | None = "normal",
) -> dict:
    ev = {
        "model": "BMW i3 120 Ah",
        "battery_capacity_kwh": 42.2,
        "battery_soc_percent": soc_percent,
    }
    if energy_kwh is not None:
        ev["battery_energy_kwh"] = energy_kwh
    state = {
        "ev": ev,
        "health": {
            "battery_temperature_c": 28.0,
            "tire_pressure_psi": {
                "front_left": 32.0,
                "front_right": 32.0,
                "rear_left": 31.0,
                "rear_right": 31.0,
            },
        },
    }
    if traffic_level is not None:
        # This is the normalized shape returned by tools.vehicle.get_vehicle_state.
        state["traffic"] = {"level": traffic_level}
    return state


def weather_state(**updates) -> dict:
    weather = {
        "temperature_c": 24.0,
        "humidity_percent": 55.0,
        "wind_speed_kmh": 18.0,
        "precipitation_mm": 0.0,
        "rain_mm": 0.0,
        "weather_code": 1,
        "visibility_m": 10_000.0,
    }
    weather.update(updates)
    return weather


def route_state(segments: list[dict], distance_km: float | None = None) -> dict:
    total = (
        sum(float(item["distance_km"]) for item in segments)
        if distance_km is None
        else distance_km
    )
    return {
        "origin": {"name": "Sfax"},
        "destination": {"name": "Tunis"},
        "distance_km": total,
        "duration_minutes": 120.0,
        "segments": segments,
    }


def segment(
    road_type: str,
    distance_km: float,
    speed_kmh: float,
    *,
    posted_limit: float | None = None,
    slope_percent: float = 0.0,
    acceleration_mps2: float = 0.0,
) -> dict:
    return {
        "type": road_type,
        "distance_km": distance_km,
        "average_speed_kmh": speed_kmh,
        "posted_speed_limit_kmh": posted_limit,
        "slope_percent": slope_percent,
        "acceleration_mps2": acceleration_mps2,
        "road_names": [],
        "uses": [],
    }


class DistancePredictor:
    """Predict a fixed 0.2 kWh/km and retain each feature vector."""

    target = INDIA_MODEL_TARGET

    def __init__(self, rate_kwh_per_km: float = 0.2) -> None:
        self.rate_kwh_per_km = rate_kwh_per_km
        self.received: list[dict[str, object]] = []

    def predict(self, features) -> float:
        copied = dict(features)
        self.received.append(copied)
        return self.rate_kwh_per_km


class SequencePredictor:
    """Return one configured kWh/km rate for each successive segment."""

    target = INDIA_MODEL_TARGET

    def __init__(self, rates_kwh_per_km: list[float]) -> None:
        self.rates = iter(rates_kwh_per_km)
        self.received: list[dict[str, object]] = []

    def predict(self, features) -> float:
        self.received.append(dict(features))
        return next(self.rates)


class RaisingProvider:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("provider offline")


class EnergySubgraphTests(unittest.TestCase):
    def invoke(
        self,
        route: dict,
        *,
        weather: dict | None = None,
        vehicle: dict | None = None,
        reserve: float = 15.0,
        predictor: DistancePredictor | None = None,
    ) -> tuple[dict, DistancePredictor]:
        selected_predictor = predictor or DistancePredictor()
        graph = build_energy_estimation_graph(
            object(),
            route_provider=lambda origin, destination: deepcopy(route),
            weather_provider=lambda city: deepcopy(
                weather if weather is not None else weather_state()
            ),
            vehicle_provider=lambda simulator: deepcopy(
                vehicle if vehicle is not None else vehicle_state()
            ),
            predictor=selected_predictor,
        )
        state = graph.invoke(
            {
                "origin": "Sfax",
                "destination": "Tunis",
                "reserve_soc_percent": reserve,
                "warnings": [],
                "errors": [],
            }
        )
        self.last_state = state
        return energy_result(state), selected_predictor

    def test_entire_urban_rural_and_highway_routes(self) -> None:
        cases = [
            ("urban", 45.0, 50.0),
            ("suburban", 75.0, 90.0),
            ("highway", 100.0, 110.0),
        ]
        for road_type, speed, expected_limit in cases:
            with self.subTest(road_type=road_type):
                result, _ = self.invoke(
                    route_state([segment(road_type, 10.0, speed)])
                )
                self.assertEqual(result["status"], "ok")
                self.assertEqual(
                    self.last_state["route_segments"][0][
                        "legal_speed_limit_kmh"
                    ],
                    expected_limit,
                )
                self.assertAlmostEqual(
                    result["estimated_energy_consumption_kwh"], 2.0
                )

    def test_mixed_route_predicts_each_segment_and_sums_kwh(self) -> None:
        route = route_state(
            [
                segment("urban", 10.0, 40.0),
                segment("highway", 80.0, 100.0),
                segment("urban", 15.0, 45.0),
            ]
        )
        result, predictor = self.invoke(route)

        self.assertEqual(len(self.last_state["route_segments"]), 3)
        self.assertEqual(len(predictor.received), 3)
        self.assertAlmostEqual(result["total_distance_km"], 105.0)
        self.assertAlmostEqual(
            result["estimated_energy_consumption_kwh"], 21.0
        )
        self.assertEqual(
            [
                item["road_type"]
                for item in self.last_state["route_segments"]
            ],
            ["urban", "highway", "urban"],
        )

    def test_segment_rates_are_multiplied_by_segment_distance(self) -> None:
        predictor = SequencePredictor([0.12, 0.15])
        result, _ = self.invoke(
            route_state(
                [segment("urban", 5.0, 40.0), segment("highway", 10.0, 90.0)]
            ),
            predictor=predictor,
        )

        self.assertEqual(
            [
                item["predicted_energy_kwh"]
                for item in self.last_state["route_segments"]
            ],
            [0.6, 1.5],
        )
        self.assertAlmostEqual(result["estimated_energy_consumption_kwh"], 2.1)

    def test_public_result_excludes_internal_subgraph_state(self) -> None:
        result, _ = self.invoke(
            route_state([segment("urban", 10.0, 40.0)])
        )

        for internal_key in (
            "route",
            "weather",
            "vehicle",
            "route_segments",
            "segments",
            "ml_features",
        ):
            self.assertNotIn(internal_key, result)
        self.assertEqual(
            result["estimated_energy_consumption_kwh"], 2.0
        )
        self.assertIn("available_energy_kwh", result)
        self.assertIn("feasible", result)

    def test_weather_reuses_resolved_route_destination_coordinates(self) -> None:
        route = route_state([segment("urban", 12.2, 40.0)])
        route["origin"] = {
            "canonical_name": "Route El Ain Km 4, Sfax, Tunisia",
            "name": "Route El Ain Km 4, Sfax, Tunisia",
            "latitude": 34.7518952,
            "longitude": 10.7296079,
        }
        route["destination"] = {
            "canonical_name": (
                "ENET'Com, Technopole de Sfax, Sakiet Ezzit, Tunisia"
            ),
            "name": "ENET'Com, Technopole de Sfax, Sakiet Ezzit, Tunisia",
            "latitude": 34.839620,
            "longitude": 10.756662,
        }
        coordinate_calls = []

        def coordinate_weather(latitude, longitude, *, location_name=None):
            coordinate_calls.append((latitude, longitude, location_name))
            return weather_state()

        graph = build_energy_estimation_graph(
            object(),
            route_provider=lambda origin, destination: deepcopy(route),
            weather_provider=RaisingProvider(),
            coordinate_weather_provider=coordinate_weather,
            vehicle_provider=lambda simulator: vehicle_state(),
            predictor=DistancePredictor(),
        )
        state = graph.invoke(
            {
                "origin": "Route El Ain Km 4, Sfax",
                "destination": "ENET'Com, Cite El Ons, Sfax",
                "warnings": [],
                "errors": [],
            }
        )
        result = energy_result(state)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            coordinate_calls,
            [
                (
                    34.839620,
                    10.756662,
                    "ENET'Com, Technopole de Sfax, Sakiet Ezzit, Tunisia",
                )
            ],
        )
        self.assertEqual(
            result["origin"], "Route El Ain Km 4, Sfax, Tunisia"
        )
        self.assertEqual(
            result["destination"],
            "ENET'Com, Technopole de Sfax, Sakiet Ezzit, Tunisia",
        )

    def test_segment_distances_must_match_route_total(self) -> None:
        result, _ = self.invoke(
            route_state(
                [segment("urban", 10.0, 40.0)],
                distance_km=50.0,
            )
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("do not match", result["errors"][0])

    def test_exact_india_feature_mapping_and_bmw_reference_values(self) -> None:
        vehicle = vehicle_state()
        weather = weather_state()
        raw_segment = {
            "distance_km": 12.0,
            "estimated_speed_kmh": 70.0,
            "slope_percent": 2.0,
            "acceleration_mps2": 0.5,
        }

        features = build_ml_features(
            vehicle=vehicle,
            weather=weather,
            segment=raw_segment,
        )

        self.assertEqual(list(features), FEATURES)
        self.assertEqual(features["motor_power_kW"], 125.0)
        self.assertEqual(features["vehicle_weight_kg"], 1270.0)
        self.assertEqual(features["drag_coefficient_Cd"], 0.29)
        self.assertEqual(features["frontal_area_m2"], 2.38)
        self.assertEqual(features["rolling_resistance_Cr"], 0.012)
        self.assertEqual(features["motor_efficiency_pct"], 92.0)
        self.assertEqual(features["avg_driving_speed_kmh"], 70.0)
        self.assertEqual(features["ambient_temperature_C"], 24.0)
        self.assertEqual(features["ac_usage"], "0")
        self.assertEqual(features["terrain_type"], "0")
        self.assertEqual(features["traffic_condition"], "1")
        self.assertEqual(features["driving_style"], "1")
        self.assertEqual(features["passengers"], 1.0)
        self.assertEqual(features["battery_age_years"], 1.0)
        self.assertEqual(features["tyre_pressure_deviation_psi"], -0.5)
        for name in ("ac_usage", "terrain_type", "traffic_condition", "driving_style"):
            self.assertIsInstance(features[name], str)
        self.assertEqual(BMW_I3_120_AH.battery_capacity_kwh, 42.2)
        self.assertEqual(BMW_I3_120_AH.max_motor_power_kw, 125.0)
        self.assertEqual(BMW_I3_120_AH.vehicle_mass_kg, 1270.0)
        self.assertEqual(SUMO_EV_MODEL_APPROXIMATIONS["air_drag_coefficient"], 0.29)
        self.assertEqual(SUMO_EV_MODEL_APPROXIMATIONS["front_surface_area_m2"], 2.38)

    def test_energy_warning_uses_rate_and_segment_energy_units(self) -> None:
        self.invoke(route_state([segment("urban", 5.0, 40.0)]))

        unit_warning = next(
            warning
            for warning in self.last_state["warnings"]
            if "model predicts" in warning
        )
        self.assertIn("kWh/km", unit_warning)
        self.assertIn("distance in km", unit_warning)
        self.assertIn("energy in kWh", unit_warning)
        self.assertNotIn("total energy per training row", unit_warning)

    def test_traffic_condition_uses_normalized_vehicle_level_and_defaults_if_missing(self) -> None:
        expected = {"low": "0", "normal": "1", "heavy": "2", None: "1"}
        self.assertEqual(TRAFFIC_CONDITION_CODES, {"low": "0", "normal": "1", "heavy": "2"})
        for level, expected_code in expected.items():
            with self.subTest(normalized_traffic_level=level):
                vehicle = vehicle_state(traffic_level=level)
                features = build_ml_features(
                    vehicle=vehicle,
                    weather=weather_state(),
                    segment={"estimated_speed_kmh": 40.0},
                )
                self.assertEqual(features["traffic_condition"], expected_code)
                self.assertIsInstance(features["traffic_condition"], str)

    @unittest.skipUnless(
        DEFAULT_MODEL_PATH.is_file(),
        "The saved ML artifact is local and intentionally excluded from Git.",
    )
    def test_saved_model_artifact_has_exact_target_and_feature_order(self) -> None:
        model = SavedEnergyModel()
        self.assertEqual(model.target, "energy_consumption_kWh_per_km")

    def test_speed_policy_handles_rain_fog_and_posted_limits(self) -> None:
        clear = weather_state()
        rainy = weather_state(rain_mm=1.0, weather_code=61)
        foggy = weather_state(visibility_m=40.0, weather_code=45)

        self.assertEqual(
            get_legal_speed_limit("highway", None, clear), 110.0
        )
        self.assertEqual(
            get_legal_speed_limit("highway", None, rainy), 90.0
        )
        self.assertEqual(
            get_legal_speed_limit("highway", None, foggy), 50.0
        )
        self.assertEqual(
            get_legal_speed_limit("rural", 70.0, clear), 70.0
        )
        self.assertEqual(
            get_legal_speed_limit(
                "highway", None, clear, uses=["ramp"]
            ),
            70.0,
        )

    def test_speed_used_by_model_is_capped_not_legal_limit_as_average(self) -> None:
        result, predictor = self.invoke(
            route_state([segment("highway", 10.0, 105.0)]),
            weather=weather_state(rain_mm=1.0, weather_code=61),
        )

        internal_segment = self.last_state["route_segments"][0]
        self.assertEqual(internal_segment["legal_speed_limit_kmh"], 90.0)
        self.assertEqual(internal_segment["estimated_speed_kmh"], 90.0)
        self.assertEqual(predictor.received[0]["avg_driving_speed_kmh"], 90.0)

    def test_sufficient_insufficient_and_reserve_handling(self) -> None:
        route = route_state([segment("urban", 100.0, 40.0)])
        sufficient, _ = self.invoke(route, vehicle=vehicle_state(energy_kwh=33.0))
        insufficient, _ = self.invoke(route, vehicle=vehicle_state(energy_kwh=24.0))
        high_reserve, _ = self.invoke(
            route,
            vehicle=vehicle_state(energy_kwh=33.0),
            reserve=35.0,
        )

        self.assertTrue(sufficient["feasible"])
        self.assertFalse(insufficient["feasible"])
        self.assertFalse(high_reserve["feasible"])
        self.assertAlmostEqual(
            sufficient["reserve_energy_kwh"], 42.2 * 0.15, places=3
        )

    def test_available_energy_falls_back_to_capacity_times_soc(self) -> None:
        result, _ = self.invoke(
            route_state([segment("urban", 10.0, 40.0)]),
            vehicle=vehicle_state(soc_percent=50.0, energy_kwh=None),
        )

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["available_energy_kwh"], 21.1)
        self.assertTrue(
            any("calculated" in item for item in result["warnings"])
        )

    def test_missing_battery_or_weather_measurement_is_explicit_error(self) -> None:
        no_soc = vehicle_state()
        del no_soc["ev"]["battery_soc_percent"]
        missing_battery, _ = self.invoke(
            route_state([segment("urban", 10.0, 40.0)]),
            vehicle=no_soc,
        )
        no_temperature = weather_state()
        del no_temperature["temperature_c"]
        missing_weather, _ = self.invoke(
            route_state([segment("urban", 10.0, 40.0)]),
            weather=no_temperature,
        )

        self.assertEqual(missing_battery["status"], "error")
        self.assertIn("battery_soc_percent", missing_battery["errors"][0])
        self.assertEqual(missing_weather["status"], "error")
        self.assertIn("Weather temperature", missing_weather["errors"][0])

    def test_missing_road_classification_is_not_fabricated(self) -> None:
        invalid_segment = segment("urban", 10.0, 40.0)
        invalid_segment["type"] = None
        result, _ = self.invoke(route_state([invalid_segment]))

        self.assertEqual(result["status"], "error")
        self.assertIn("classification", result["errors"][0])

    def test_missing_slope_and_acceleration_use_configured_defaults(self) -> None:
        raw = segment("urban", 10.0, 40.0)
        del raw["slope_percent"]
        del raw["acceleration_mps2"]
        result, _ = self.invoke(route_state([raw]))

        self.assertEqual(result["status"], "ok")
        features = self.last_state["route_segments"][0]["ml_features"]
        self.assertEqual(list(features), FEATURES)
        self.assertNotIn("slope_force_proxy_N", features)
        self.assertNotIn("inertial_force_proxy_N", features)
        self.assertTrue(
            any("0.00% average slope" in item for item in result["warnings"])
        )
        self.assertTrue(
            any("0.00 m/s²" in item for item in result["warnings"])
        )

    def test_explicit_slope_and_acceleration_assumptions_are_supported(self) -> None:
        raw = segment("urban", 10.0, 40.0)
        del raw["slope_percent"]
        del raw["acceleration_mps2"]
        predictor = DistancePredictor()
        graph = build_energy_estimation_graph(
            object(),
            route_provider=lambda origin, destination: route_state([raw]),
            weather_provider=lambda city: weather_state(),
            vehicle_provider=lambda simulator: vehicle_state(),
            predictor=predictor,
        )

        internal_state = graph.invoke(
            {
                "origin": "Sfax",
                "destination": "Tunis",
                "assumed_slope_percent": 0.0,
                "assumed_acceleration_mps2": 0.0,
                "warnings": [],
                "errors": [],
            }
        )
        result = energy_result(internal_state)

        self.assertEqual(result["status"], "ok")
        internal_features = internal_state["route_segments"][0][
            "ml_features"
        ]
        self.assertEqual(list(internal_features), FEATURES)
        self.assertNotIn("slope_force_proxy_N", internal_features)
        self.assertNotIn("inertial_force_proxy_N", internal_features)

    def test_external_api_failure_is_returned_without_prediction(self) -> None:
        predictor = DistancePredictor()
        graph = build_energy_estimation_graph(
            object(),
            route_provider=RaisingProvider(),
            weather_provider=lambda city: weather_state(),
            vehicle_provider=lambda simulator: vehicle_state(),
            predictor=predictor,
        )
        result = energy_result(
            graph.invoke(
                {
                    "origin": "Sfax",
                    "destination": "Tunis",
                    "warnings": [],
                    "errors": [],
                }
            )
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("provider offline", result["errors"][0])
        self.assertEqual(predictor.received, [])

    def test_repeated_invocations_fetch_fresh_measurements(self) -> None:
        calls = {"route": 0, "weather": 0, "vehicle": 0}
        energy_values = iter([33.0, 20.0])
        rain_values = iter([0.0, 1.0])

        def route_provider(origin, destination):
            calls["route"] += 1
            return route_state([segment("highway", 10.0, 105.0)])

        def weather_provider(city):
            calls["weather"] += 1
            rain = next(rain_values)
            return weather_state(rain_mm=rain, weather_code=61 if rain else 1)

        def vehicle_provider(simulator):
            calls["vehicle"] += 1
            return vehicle_state(energy_kwh=next(energy_values))

        graph = build_energy_estimation_graph(
            object(),
            route_provider=route_provider,
            weather_provider=weather_provider,
            vehicle_provider=vehicle_provider,
            predictor=DistancePredictor(),
        )
        inputs = {
            "origin": "Sfax",
            "destination": "Tunis",
            "warnings": [],
            "errors": [],
        }
        first_state = graph.invoke(inputs)
        first = energy_result(first_state)
        second_state = graph.invoke(inputs)
        second = energy_result(second_state)

        self.assertEqual(calls, {"route": 2, "weather": 2, "vehicle": 2})
        self.assertEqual(first["available_energy_kwh"], 33.0)
        self.assertEqual(second["available_energy_kwh"], 20.0)
        self.assertEqual(
            first_state["route_segments"][0]["estimated_speed_kmh"],
            105.0,
        )
        self.assertEqual(
            second_state["route_segments"][0]["estimated_speed_kmh"],
            90.0,
        )


if __name__ == "__main__":
    unittest.main()
