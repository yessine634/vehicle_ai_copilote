"""Run a watchable congested SUMO trip from Route El Ain to ENET'Com."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.sumo_runner import SumoTrafficSimulator


SCENARIO_DIRECTORY = Path(__file__).resolve().parent / "sumo_enetcom"
ENETCOM_CONFIG = SCENARIO_DIRECTORY / "enetcom.sumocfg"

ORIGIN_NAME = "Route El Ain km 4, Sfax"
DESTINATION_NAME = "ENET'Com, Technopole de Sfax"
ORIGIN_COORDINATES = (34.7518952, 10.7296079)
DESTINATION_COORDINATES = (34.839620, 10.756662)
ROUTE_DISTANCE_KM = 11.25


def _print_console(value: object) -> None:
    """Print OSM road names safely in Windows terminals."""

    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def run_enetcom_demo(
    *,
    use_gui: bool = False,
    max_steps: int = 2400,
    print_every: int = 20,
    gui_delay_seconds: float = 0.15,
) -> None:
    """Drive to ENET'Com with commuter traffic and watchable GUI pacing."""

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")
    if print_every <= 0:
        raise ValueError("print_every must be greater than zero.")
    if gui_delay_seconds < 0:
        raise ValueError("gui_delay_seconds cannot be negative.")

    simulator = SumoTrafficSimulator(
        ENETCOM_CONFIG,
        traffic_level="heavy",
    )
    vehicle_seen = False
    tracking_enabled = False
    last_state = None

    print(f"Origin: {ORIGIN_NAME} {ORIGIN_COORDINATES}")
    print(f"Destination: {DESTINATION_NAME} {DESTINATION_COORDINATES}")
    print(f"Route distance: approximately {ROUTE_DISTANCE_KM:.2f} km")
    print("Traffic: busy commuter flow (approximately 1,800 vehicles/hour)")

    try:
        simulator.start(use_gui=use_gui)

        for step_number in range(1, max_steps + 1):
            simulation_time = simulator.step()
            vehicle_ids = simulator.get_vehicle_ids()
            ego_active = "ego_vehicle" in vehicle_ids

            if ego_active:
                vehicle_seen = True
                last_state = simulator.get_vehicle_state("ego_vehicle")

                if use_gui and not tracking_enabled:
                    simulator.track_vehicle("ego_vehicle", zoom=1800.0)
                    tracking_enabled = True
                    print("SUMO GUI is now tracking the green ego vehicle.")

                if step_number == 1 or step_number % print_every == 0:
                    motion = last_state["motion"]
                    ev = last_state["ev"]
                    traffic = simulator.get_traffic_state("ego_vehicle")
                    _print_console(
                        f"t={simulation_time:6.1f}s | "
                        f"speed={motion['speed_kmh']:5.1f} km/h | "
                        f"road={motion['road_name']} | "
                        f"SOC={ev['battery_soc_percent']:5.1f}% | "
                        f"active={traffic['active_vehicles']} | "
                        f"halted={traffic['halted_vehicles']} | "
                        f"mean speed={traffic['average_speed_kph']:5.1f} km/h"
                    )
            elif vehicle_seen:
                print(
                    f"The vehicle reached {DESTINATION_NAME} at "
                    f"t={simulation_time:.1f}s."
                )
                break

            if use_gui and gui_delay_seconds:
                time.sleep(gui_delay_seconds)
        else:
            if not vehicle_seen:
                raise RuntimeError("ego_vehicle never entered the simulation.")
            print(f"Stopped at the safety limit of {max_steps} steps.")

        if last_state is not None:
            ev = last_state["ev"]
            print(
                "Final EV state: "
                f"SOC={ev['battery_soc_percent']:.2f}%, "
                f"consumed={ev['total_energy_consumed_kwh']:.3f} kWh, "
                f"regenerated={ev['total_energy_regenerated_kwh']:.3f} kWh"
            )
    finally:
        simulator.stop()


if __name__ == "__main__":
    run_enetcom_demo(
        use_gui=os.getenv("RUN_SUMO_GUI_DEMO") == "1",
        gui_delay_seconds=float(os.getenv("SUMO_GUI_DELAY_SECONDS", "0.15")),
    )
