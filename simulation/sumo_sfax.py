"""Configuration and demo helpers for the compact central-Sfax SUMO scenario."""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ev_config import DEFAULT_INITIAL_SOC_PERCENT

SCENARIO_DIRECTORY = Path(__file__).resolve().parent / "sumo" / "sfax"
SFAX_CONFIGS = {
    level: SCENARIO_DIRECTORY / f"sfax_{level}.sumocfg"
    for level in ("low", "normal", "heavy")
}
SFAX_ORIGIN = {
    "name": "Route El Ain, central Sfax",
    "city": "Sfax",
    "country": "Tunisia",
    "latitude": 34.7518952,
    "longitude": 10.7296079,
}
SFAX_DESTINATION = {
    "name": "North-east central Sfax",
    "city": "Sfax",
    "country": "Tunisia",
    "latitude": 34.762,
    "longitude": 10.746,
}
SFAX_ROUTE_DISTANCE_KM = 1.824


def run_sfax_trip_demo(
    *,
    traffic_level: str = "normal",
    use_gui: bool = False,
    max_steps: int = 600,
    print_every: int = 10,
    initial_soc_percent: float = DEFAULT_INITIAL_SOC_PERCENT,
    ambient_temperature_c: float = 25.0,
) -> None:
    """Print the live SUMO IONIQ 5 motion, EV-energy, and traffic state."""

    from simulation.trip_simulator import TripSimulator

    simulator = TripSimulator(
        origin=SFAX_ORIGIN["name"],
        destination=SFAX_DESTINATION["name"],
        origin_coordinates=(SFAX_ORIGIN["latitude"], SFAX_ORIGIN["longitude"]),
        destination_coordinates=(
            SFAX_DESTINATION["latitude"],
            SFAX_DESTINATION["longitude"],
        ),
        traffic_level=traffic_level,
        initial_soc_percent=initial_soc_percent,
        ambient_temperature_c=ambient_temperature_c,
    )
    try:
        state = simulator.start(use_gui=use_gui)
        steps = 0
        while state["status"] == "running" and steps < max_steps:
            state = simulator.step()
            steps += 1
            if steps == 1 or steps % print_every == 0:
                motion = state["movement"]
                ev = state["ev"]
                health = state["health"]
                traffic = state["traffic"]
                print(f"\nSimulation time: {motion['elapsed_seconds']:.0f} s")
                print(ev["model"])
                print("Motion:")
                print(f"  Speed: {motion['speed_kph']:.1f} km/h")
                print(
                    f"  Acceleration: {motion['acceleration_mps2']:.2f} m/s^2"
                )
                print(f"  Road: {motion['current_road']}")
                print(f"  Lane: {motion['lane_id']}")
                print(f"  Waiting time: {motion['waiting_time_s']:.1f} s")
                print("EV state:")
                print(f"  Battery SOC: {ev['battery_soc_percent']:.2f} %")
                print(f"  Battery energy: {ev['battery_energy_kwh']:.3f} kWh")
                print(
                    "  Electricity consumption: "
                    f"{ev['instant_consumption_wh_per_s']:.3f} Wh/s"
                )
                print(
                    f"  Total consumed: {ev['total_energy_consumed_kwh']:.4f} kWh"
                )
                print(
                    "  Regenerated energy: "
                    f"{ev['total_energy_regenerated_kwh']:.4f} kWh"
                )
                print(
                    "  Estimated range: "
                    f"{ev['estimated_remaining_range_km']:.1f} km "
                    f"({ev['range_estimate_source']})"
                )
                print("Estimated health:")
                print(
                    f"  Battery SOH: "
                    f"{health['traction_battery_soh_percent']:.3f} %"
                )
                print(
                    f"  Temperatures: battery={health['battery_temperature_c']:.1f} C, "
                    f"motor={health['motor_temperature_c']:.1f} C, "
                    f"inverter={health['inverter_temperature_c']:.1f} C"
                )
                print(
                    f"  Tire pressure: "
                    f"{health['tire_pressure_psi']['front_left']:.1f} psi"
                )
                print(
                    f"  12 V battery: {health['auxiliary_battery_voltage']:.2f} V | "
                    f"brake pads={health['brake_pad_life_percent']:.3f}%"
                )
                print(
                    f"  Status: {health['overall_status']} | "
                    f"warnings={health['warnings']}"
                )
                print(
                    f"Traffic: {traffic['traffic_level']} | "
                    f"active={traffic['active_vehicles']} | "
                    f"mean speed={traffic['average_speed_kph']:.1f} km/h"
                )

        print(f"Sfax SUMO trip ended with status: {state['status']}")
    finally:
        simulator.stop()


if __name__ == "__main__":
    run_sfax_trip_demo(
        traffic_level=os.getenv("SUMO_TRAFFIC_LEVEL", "normal"),
        use_gui=os.getenv("RUN_SUMO_GUI_DEMO") == "1",
        initial_soc_percent=float(
            os.getenv("SUMO_INITIAL_SOC_PERCENT", DEFAULT_INITIAL_SOC_PERCENT)
        ),
        ambient_temperature_c=float(
            os.getenv("SUMO_AMBIENT_TEMPERATURE_C", "25")
        ),
    )
