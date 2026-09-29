"""Coordinate the SUMO trip, native EV energy, and estimated EV health."""

from __future__ import annotations

from typing import Any

from simulation.ev_config import DEFAULT_INITIAL_SOC_PERCENT
from simulation.ev_health import EVHealthConfig, EVHealthSimulator


Coordinate = tuple[float, float]


class TripSimulator:
    """Run the compact Sfax SUMO simulation for one electric ego vehicle."""

    def __init__(
        self,
        origin: str,
        destination: str,
        *,
        origin_coordinates: Coordinate | None = None,
        destination_coordinates: Coordinate | None = None,
        traffic_level: str = "normal",
        initial_soc_percent: float = DEFAULT_INITIAL_SOC_PERCENT,
        ambient_temperature_c: float = 25.0,
    ) -> None:
        if (origin_coordinates is None) != (destination_coordinates is None):
            raise ValueError("Provide both origin and destination coordinates, or neither.")
        if traffic_level not in {"low", "normal", "heavy"}:
            raise ValueError("traffic_level must be 'low', 'normal', or 'heavy'.")
        if not 0 <= initial_soc_percent <= 100:
            raise ValueError("initial_soc_percent must be between 0 and 100.")

        self.origin = origin
        self.destination = destination
        self.origin_coordinates = origin_coordinates
        self.destination_coordinates = destination_coordinates
        self.traffic_level = traffic_level
        self.initial_soc_percent = float(initial_soc_percent)
        self.ambient_temperature_c = float(ambient_temperature_c)
        self.ev_health = EVHealthSimulator(
            EVHealthConfig(ambient_temperature_c=self.ambient_temperature_c)
        )
        self.route_info: dict[str, Any] | None = None
        self.sumo: Any | None = None
        self.ego_vehicle_id = "ego_vehicle"
        self._sumo_last_state: dict[str, Any] | None = None
        self._sumo_vehicle_seen = False
        self.elapsed_seconds = 0.0
        self.travelled_distance_km = 0.0
        self._started = False

    def start(self, *, use_gui: bool = False) -> dict[str, Any]:
        """Start the Sfax SUMO scenario and return the initial combined state."""

        if self._started:
            raise RuntimeError("The trip simulation is already running.")

        from simulation.sumo_runner import SumoTrafficSimulator
        from simulation.sumo_sfax import (
            SFAX_CONFIGS,
            SFAX_DESTINATION,
            SFAX_ORIGIN,
            SFAX_ROUTE_DISTANCE_KM,
        )

        origin = dict(SFAX_ORIGIN)
        destination = dict(SFAX_DESTINATION)
        if self.origin_coordinates is not None:
            origin.update(
                {
                    "name": self.origin,
                    "latitude": self.origin_coordinates[0],
                    "longitude": self.origin_coordinates[1],
                }
            )
            destination.update(
                {
                    "name": self.destination,
                    "latitude": self.destination_coordinates[0],
                    "longitude": self.destination_coordinates[1],
                }
            )

        self.route_info = {
            "origin": origin,
            "destination": destination,
            "distance_km": SFAX_ROUTE_DISTANCE_KM,
            "duration_minutes": None,
            "uses_highway": False,
            "has_toll": False,
            "has_ferry": False,
        }
        self.elapsed_seconds = 0.0
        self.travelled_distance_km = 0.0
        self.ev_health.reset()
        self._sumo_last_state = None
        self._sumo_vehicle_seen = False
        self.sumo = SumoTrafficSimulator(
            SFAX_CONFIGS[self.traffic_level],
            traffic_level=self.traffic_level,
            initial_soc_percent=self.initial_soc_percent,
        )

        try:
            self.sumo.start(use_gui=use_gui)
            for _ in range(10):
                self.sumo.step()
                if self.ego_vehicle_id in self.sumo.get_vehicle_ids():
                    self._sumo_last_state = self.sumo.get_vehicle_state(
                        self.ego_vehicle_id
                    )
                    self._sumo_vehicle_seen = True
                    break
            if self._sumo_last_state is None:
                raise RuntimeError("ego_vehicle did not enter the SUMO simulation.")
            if use_gui:
                self.sumo.track_vehicle(self.ego_vehicle_id, zoom=3000.0)
        except Exception:
            self.sumo.stop()
            self.sumo = None
            raise

        self.elapsed_seconds = self._sumo_last_state["motion"][
            "simulation_time_s"
        ]
        self._update_ev_health(
            self._sumo_last_state,
            elapsed_seconds=0.0,
            distance_delta_km=0.0,
        )
        self._started = True
        return self.get_state()

    def _update_ev_health(
        self,
        state: dict[str, Any],
        *,
        elapsed_seconds: float,
        distance_delta_km: float,
    ) -> None:
        """Feed native SUMO telemetry into supplemental EV-health estimates."""

        motion = state["motion"]
        ev = state["ev"]
        self.ev_health.update(
            speed_kmh=motion["speed_kmh"],
            acceleration_mps2=motion["acceleration_mps2"],
            elapsed_seconds=elapsed_seconds,
            distance_delta_km=distance_delta_km,
            instant_consumption_wh_per_s=ev[
                "instant_consumption_wh_per_s"
            ],
            total_energy_consumed_kwh=ev["total_energy_consumed_kwh"],
            total_energy_regenerated_kwh=ev[
                "total_energy_regenerated_kwh"
            ],
            battery_low=ev["battery_low"],
        )

    def step(self, seconds: float = 1.0) -> dict[str, Any]:
        """Advance SUMO and update the combined electric-vehicle state."""

        if not self._started or self.sumo is None:
            raise RuntimeError("Call start() before advancing the simulation.")
        if seconds <= 0:
            raise ValueError("seconds must be greater than zero.")

        target_time = self.sumo.get_simulation_time() + float(seconds)
        while self.sumo.get_simulation_time() < target_time:
            previous_time = self.sumo.get_simulation_time()
            self.sumo.step()
            current_time = self.sumo.get_simulation_time()
            vehicle_ids = self.sumo.get_vehicle_ids()
            if self.ego_vehicle_id not in vehicle_ids:
                if self._sumo_vehicle_seen:
                    break
                continue

            state = self.sumo.get_vehicle_state(self.ego_vehicle_id)
            distance_km = state["ev"]["distance_travelled_km"]
            distance_delta = max(distance_km - self.travelled_distance_km, 0.0)
            self.travelled_distance_km = distance_km
            self._update_ev_health(
                state,
                elapsed_seconds=current_time - previous_time,
                distance_delta_km=distance_delta,
            )
            self._sumo_last_state = state
            self._sumo_vehicle_seen = True

        self.elapsed_seconds = self.sumo.get_simulation_time()
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return combined SUMO motion, EV energy, health, and traffic state."""

        if (
            not self._started
            or self.sumo is None
            or self.route_info is None
            or self._sumo_last_state is None
        ):
            raise RuntimeError("Call start() before requesting simulation state.")

        vehicle_active = self.ego_vehicle_id in self.sumo.get_vehicle_ids()
        motion = dict(self._sumo_last_state["motion"])
        ev = dict(self._sumo_last_state["ev"])
        geo = motion.get("position_geo")
        if not isinstance(geo, dict):
            raise RuntimeError("The SUMO Sfax network did not provide GPS coordinates.")

        route_distance = float(self.route_info["distance_km"])
        progress = min(self.travelled_distance_km / route_distance * 100, 100.0)
        speed_kph = motion["speed_kmh"] if vehicle_active else 0.0
        acceleration = motion["acceleration_mps2"] if vehicle_active else 0.0
        position_xy = motion["position_xy"]
        return {
            "status": "running" if vehicle_active else "completed",
            "simulation_backend": "sumo",
            "origin": dict(self.route_info["origin"]),
            "destination": dict(self.route_info["destination"]),
            "route": {
                "distance_km": route_distance,
                "duration_minutes": None,
                "uses_highway": False,
                "has_toll": False,
                "has_ferry": False,
            },
            "movement": {
                "speed_kph": round(float(speed_kph), 2),
                "acceleration_mps2": round(float(acceleration), 3),
                "position": {
                    "latitude": round(float(geo["latitude"]), 7),
                    "longitude": round(float(geo["longitude"]), 7),
                    "x": round(float(position_xy["x"]), 2),
                    "y": round(float(position_xy["y"]), 2),
                },
                "current_road": motion["road_name"],
                "road_name": motion["road_name"],
                "road_id": motion["road_id"],
                "lane_id": motion["lane_id"],
                "lane_position_m": round(float(motion["lane_position_m"]), 2),
                "angle_deg": round(float(motion["angle_deg"]), 2),
                "waiting_time_s": round(float(motion["waiting_time_s"]), 2),
                "elapsed_seconds": round(float(self.elapsed_seconds), 2),
                "simulation_time_s": round(float(self.elapsed_seconds), 2),
                "travelled_distance_km": round(self.travelled_distance_km, 3),
                "trip_progress_percent": round(progress, 2),
            },
            "ev": ev,
            "health": self.ev_health.get_state(),
            "traffic": self.sumo.get_traffic_state(
                self.ego_vehicle_id if vehicle_active else None
            ),
        }

    def stop(self) -> None:
        """Stop the active SUMO process; repeated calls are safe."""

        if self.sumo is not None:
            self.sumo.stop()
        self._started = False
