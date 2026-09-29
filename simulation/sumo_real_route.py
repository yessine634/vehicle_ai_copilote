"""Run one SUMO vehicle on real OSM roads in Sfax."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# Support both ``python simulation\sumo_real_route.py`` and module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.sumo_runner import SumoTrafficSimulator


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parent / "sumo_real_route_demo"
)
REAL_ROUTE_CONFIG = SCENARIO_DIRECTORY / "sfax_route.sumocfg"

ORIGIN_NAME = "Route El Ain km 4, Sfax"
DESTINATION_NAME = "Route Teniour km 8, Sfax"
ORIGIN_COORDINATES = (34.7518952, 10.7296079)
DESTINATION_COORDINATES = (34.8157219, 10.7342885)
ROUTE_DISTANCE_KM = 6.86


def run_real_route_demo(
    *,
    use_gui: bool = False,
    max_steps: int = 1500,
    print_every: int = 10,
) -> None:
    """Drive the real Sfax route and periodically print live TraCI state."""

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")
    if print_every <= 0:
        raise ValueError("print_every must be greater than zero.")

    simulator = SumoTrafficSimulator(REAL_ROUTE_CONFIG)
    vehicle_seen = False
    gui_tracking_enabled = False
    steps = 0

    print(f"Origin: {ORIGIN_NAME} {ORIGIN_COORDINATES}")
    print(f"Destination: {DESTINATION_NAME} {DESTINATION_COORDINATES}")
    print(f"SUMO route distance: approximately {ROUTE_DISTANCE_KM:.2f} km")

    try:
        simulator.start(use_gui=use_gui)
        while simulator.has_pending_vehicles() and steps < max_steps:
            simulation_time = simulator.step()
            steps += 1
            vehicle_ids = simulator.get_vehicle_ids()
            vehicle_seen = vehicle_seen or bool(vehicle_ids)

            if use_gui and vehicle_ids and not gui_tracking_enabled:
                simulator.track_vehicle("ego_vehicle", zoom=3000.0)
                gui_tracking_enabled = True
                print("SUMO GUI is now tracking ego_vehicle.")

            if vehicle_ids and (steps == 1 or steps % print_every == 0):
                state = simulator.get_vehicle_state(vehicle_ids[0])
                motion = state["motion"]
                position = motion["position_xy"]
                print(
                    f"t={simulation_time:6.1f}s | "
                    f"vehicle={state['vehicle_id']} | "
                    f"speed={motion['speed_kmh']:5.1f} km/h | "
                    f"road={motion['road_name']} | "
                    f"lane={motion['lane_id']} | "
                    f"position=({position['x']:.1f}, {position['y']:.1f}) | "
                    f"waiting={motion['waiting_time_s']:.1f}s"
                )

            if use_gui:
                time.sleep(0.05)

        if not vehicle_seen:
            raise RuntimeError("The ego vehicle never entered the SUMO simulation.")
        if simulator.has_pending_vehicles():
            print(f"Stopped at the safety limit of {max_steps} steps.")
        else:
            print(
                f"The vehicle reached {DESTINATION_NAME} at "
                f"t={simulator.get_simulation_time():.1f}s."
            )
    finally:
        simulator.stop()


if __name__ == "__main__":
    run_real_route_demo(use_gui=os.getenv("RUN_SUMO_GUI_DEMO") == "1")
