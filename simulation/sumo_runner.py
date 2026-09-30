"""Run a small SUMO scenario and expose live vehicle data through TraCI."""



from __future__ import annotations



import os

from pathlib import Path

import shutil

import time

from typing import Any

import xml.etree.ElementTree as ET



from simulation.ev_config import (

    DEFAULT_INITIAL_SOC_PERCENT,

    IONIQ5_LONG_RANGE_RWD,

    MIN_DISTANCE_FOR_OBSERVED_RANGE_KM,

)



try:

    import traci

    import sumolib

except ImportError as exc:

    raise RuntimeError(

        "TraCI is not installed in the active Python environment. "

        "Activate llm-env and run: pip install traci sumolib"

    ) from exc





DEFAULT_CONFIG = Path(__file__).resolve().parent / "sumo_demo" / "demo.sumocfg"





class SumoTrafficSimulator:

    """Control one SUMO process and read its live state through TraCI."""



    def __init__(

        self,

        config_file: str | Path = DEFAULT_CONFIG,

        *,

        traffic_level: str = "normal",

        initial_soc_percent: float = DEFAULT_INITIAL_SOC_PERCENT,

    ) -> None:

        if traffic_level not in {"low", "normal", "heavy"}:

            raise ValueError("traffic_level must be 'low', 'normal', or 'heavy'.")

        if not 0 <= initial_soc_percent <= 100:

            raise ValueError("initial_soc_percent must be between 0 and 100.")

        self.config_file = Path(config_file).resolve()

        self.traffic_level = traffic_level

        self.initial_soc_percent = float(initial_soc_percent)

        self._running = False

        self._configured_ev_ids: set[str] = set()

        self._road_network: Any | None = None



    def _get_road_network(self):

        """Load the configured SUMO network lazily for OSM road-name lookup."""



        if self._road_network is not None:

            return self._road_network

        try:

            config_root = ET.parse(self.config_file).getroot()

            net_file = config_root.find("./input/net-file")

            net_value = net_file.get("value") if net_file is not None else None

            if not net_value:

                return None

            network_path = (self.config_file.parent / net_value).resolve()

            self._road_network = sumolib.net.readNet(

                str(network_path), withInternal=True

            )

        except (OSError, ET.ParseError, KeyError, RuntimeError):

            return None

        return self._road_network



    def _edge_name(self, edge_id: str) -> str | None:

        network = self._get_road_network()

        if network is None or not network.hasEdge(edge_id):

            return None

        name = network.getEdge(edge_id).getName().strip()

        return name or None



    def _get_driver_road_name(self, vehicle_id: str, road_id: str) -> str:

        """Return the current or nearest named OSM road, never an internal ID."""



        direct_name = self._edge_name(road_id)

        if direct_name:

            return direct_name



        route = list(traci.vehicle.getRoute(vehicle_id))

        route_index = int(traci.vehicle.getRouteIndex(vehicle_id))

        for offset in (0, 1, -1, 2, -2):

            candidate_index = route_index + offset

            if 0 <= candidate_index < len(route):

                candidate_name = self._edge_name(route[candidate_index])

                if candidate_name:

                    return candidate_name

        return "Unnamed road"



    @property

    def is_running(self) -> bool:

        """Return whether this instance currently owns a TraCI session."""



        return self._running



    def start(self, use_gui: bool = False) -> float:

        """Launch SUMO or SUMO GUI and return the initial simulation time."""



        if self._running:

            raise RuntimeError("The SUMO simulation is already running.")

        if not self.config_file.is_file():

            raise FileNotFoundError(

                f"SUMO configuration file was not found: {self.config_file}"

            )



        executable_name = "sumo-gui" if use_gui else "sumo"

        executable = shutil.which(executable_name)

        if executable is None:

            raise FileNotFoundError(

                f"Could not find '{executable_name}' on PATH. Confirm that "

                "%SUMO_HOME%\\\bin is on PATH and reopen the terminal."

            )



        command = [

            executable,

            "-c",

            str(self.config_file),

            "--no-step-log",

            "true",

            "--no-warnings",

            "true",

        ]

        if use_gui:

            command.append("--start")



        try:
            traci.start(command)
        except Exception as exc:
            # Clean up a partially-created TraCI connection
            # if SUMO crashes during startup.
            try:
                traci.close(False)
            except Exception:
                pass

            self._running = False

            raise RuntimeError(
                f"SUMO could not start with configuration: {self.config_file}"
            ) from exc

        self._running = True
        self._configured_ev_ids.clear()
        return self.get_simulation_time()



    def _require_running(self) -> None:

        if not self._running:

            raise RuntimeError("Start the SUMO simulation before requesting data.")



    def step(self) -> float:

        """Advance SUMO by one configured step and return simulation time."""



        self._require_running()

        traci.simulationStep()

        for vehicle_id in traci.vehicle.getIDList():

            self._configure_initial_ev_charge(vehicle_id)

        return self.get_simulation_time()



    def _configure_initial_ev_charge(self, vehicle_id: str) -> None:

        """Apply the requested initial SOC once when an EV enters SUMO."""



        if vehicle_id in self._configured_ev_ids:

            return

        if not traci.vehicle.getEmissionClass(vehicle_id).startswith("Energy/"):

            self._configured_ev_ids.add(vehicle_id)

            return



        try:

            capacity_wh = float(

                traci.vehicle.getParameter(

                    vehicle_id, "device.battery.capacity"

                )

            )

        except (traci.TraCIException, TypeError, ValueError) as exc:

            raise RuntimeError(

                f"Electric vehicle '{vehicle_id}' has no readable SUMO battery."

            ) from exc

        if capacity_wh <= 0:

            raise RuntimeError(

                f"Electric vehicle '{vehicle_id}' has invalid battery capacity."

            )



        charge_wh = capacity_wh * self.initial_soc_percent / 100.0

        traci.vehicle.setParameter(

            vehicle_id,

            "device.battery.chargeLevel",

            str(charge_wh),

        )

        self._configured_ev_ids.add(vehicle_id)



    @staticmethod

    def _battery_number(vehicle_id: str, parameter: str) -> float:

        """Read one required numeric SUMO battery parameter."""



        try:

            return float(traci.vehicle.getParameter(vehicle_id, parameter))

        except (traci.TraCIException, TypeError, ValueError) as exc:

            raise RuntimeError(

                f"SUMO battery parameter '{parameter}' is unavailable for "

                f"vehicle '{vehicle_id}'."

            ) from exc



    def get_vehicle_ids(self) -> list[str]:

        """Return IDs for all vehicles currently present in the simulation."""



        self._require_running()

        return list(traci.vehicle.getIDList())



    def get_vehicle_state(self, vehicle_id: str) -> dict[str, Any]:

        """Return directly observed TraCI values for one active vehicle."""



        self._require_running()

        if vehicle_id not in traci.vehicle.getIDList():

            raise ValueError(f"Vehicle '{vehicle_id}' is not currently active in SUMO.")



        self._configure_initial_ev_charge(vehicle_id)

        speed_mps = float(traci.vehicle.getSpeed(vehicle_id))

        x, y = traci.vehicle.getPosition(vehicle_id)

        road_id = traci.vehicle.getRoadID(vehicle_id)

        motion = {

            "simulation_time_s": self.get_simulation_time(),

            "speed_mps": speed_mps,

            "speed_kmh": speed_mps * 3.6,

            "acceleration_mps2": float(

                traci.vehicle.getAcceleration(vehicle_id)

            ),

            "road_id": road_id,

            "road_name": self._get_driver_road_name(vehicle_id, road_id),

            "lane_id": traci.vehicle.getLaneID(vehicle_id),

            "lane_position_m": float(

                traci.vehicle.getLanePosition(vehicle_id)

            ),

            "position_xy": {"x": float(x), "y": float(y)},

            "angle_deg": float(traci.vehicle.getAngle(vehicle_id)),

            "waiting_time_s": float(traci.vehicle.getWaitingTime(vehicle_id)),

        }

        try:

            longitude, latitude = traci.simulation.convertGeo(x, y)

        except traci.TraCIException:

            pass

        else:

            motion["position_geo"] = {

                "latitude": float(latitude),

                "longitude": float(longitude),

            }



        emission_class = traci.vehicle.getEmissionClass(vehicle_id)

        state: dict[str, Any] = {

            "vehicle_id": vehicle_id,

            "motion": motion,

        }

        if emission_class.startswith("Energy/"):

            capacity_wh = self._battery_number(

                vehicle_id, "device.battery.capacity"

            )

            charge_wh = self._battery_number(

                vehicle_id, "device.battery.chargeLevel"

            )

            consumed_wh = self._battery_number(

                vehicle_id, "device.battery.totalEnergyConsumed"

            )

            regenerated_wh = self._battery_number(

                vehicle_id, "device.battery.totalEnergyRegenerated"

            )

            if capacity_wh <= 0:

                raise RuntimeError("SUMO returned a non-positive battery capacity.")



            distance_km = max(float(traci.vehicle.getDistance(vehicle_id)), 0.0) / 1000

            capacity_kwh = capacity_wh / 1000.0

            energy_kwh = max(charge_wh, 0.0) / 1000.0

            consumed_kwh = max(consumed_wh, 0.0) / 1000.0

            regenerated_kwh = max(regenerated_wh, 0.0) / 1000.0

            soc_percent = min(max(charge_wh / capacity_wh * 100.0, 0.0), 100.0)



            actual_consumption = None

            range_source = "reference"

            consumption_for_range = (

                IONIQ5_LONG_RANGE_RWD.reference_consumption_kwh_100km

            )

            if (

                distance_km >= MIN_DISTANCE_FOR_OBSERVED_RANGE_KM

                and consumed_kwh > 0

            ):

                actual_consumption = consumed_kwh / distance_km * 100.0

                consumption_for_range = actual_consumption

                range_source = "observed_consumption"

            estimated_range = (

                energy_kwh / consumption_for_range * 100.0

                if consumption_for_range > 0

                else 0.0

            )



            try:

                charging_station_id = traci.vehicle.getParameter(

                    vehicle_id, "device.battery.chargingStationId"

                )

            except traci.TraCIException:

                charging_station_id = ""

            if charging_station_id.strip().upper() in {"", "NULL", "NONE"}:

                charging_station_id = ""



            state["ev"] = {

                "model": IONIQ5_LONG_RANGE_RWD.model_name,

                "drive": IONIQ5_LONG_RANGE_RWD.drive,

                "battery_capacity_kwh": capacity_kwh,

                "battery_energy_kwh": energy_kwh,

                "battery_soc_percent": soc_percent,

                "battery_low": soc_percent <= 15.0,

                "instant_consumption_wh_per_s": float(

                    traci.vehicle.getElectricityConsumption(vehicle_id)

                ),

                "total_energy_consumed_kwh": consumed_kwh,

                "total_energy_regenerated_kwh": regenerated_kwh,

                "reference_consumption_kwh_100km": (

                    IONIQ5_LONG_RANGE_RWD.reference_consumption_kwh_100km

                ),

                "reference_range_km": IONIQ5_LONG_RANGE_RWD.reference_range_km,

                "distance_travelled_km": distance_km,

                "actual_consumption_kwh_100km": actual_consumption,

                "estimated_remaining_range_km": max(estimated_range, 0.0),

                "range_estimate_source": range_source,

                "charging_station_id": charging_station_id or None,

                "is_charging": bool(charging_station_id),

            }

        return state



    def get_traffic_state(self, vehicle_id: str | None = None) -> dict[str, Any]:

        """Return traffic measurements observed directly from the current step."""



        self._require_running()

        vehicle_ids = list(traci.vehicle.getIDList())

        speeds_mps = [float(traci.vehicle.getSpeed(item)) for item in vehicle_ids]

        waiting_times = [

            float(traci.vehicle.getWaitingTime(item)) for item in vehicle_ids

        ]

        active_count = len(vehicle_ids)

        halted_count = sum(speed < 0.1 for speed in speeds_mps)

        mean_speed_kph = (

            sum(speeds_mps) / active_count * 3.6 if active_count else 0.0

        )

        mean_waiting_time = (

            sum(waiting_times) / active_count if active_count else 0.0

        )

        halted_percent = halted_count / active_count * 100 if active_count else 0.0



        road_vehicle_count = None

        road_mean_speed_kph = None

        if vehicle_id in vehicle_ids:

            road_id = traci.vehicle.getRoadID(vehicle_id)

            if road_id and not road_id.startswith(":"):

                road_vehicle_count = int(

                    traci.edge.getLastStepVehicleNumber(road_id)

                )

                road_speed = float(traci.edge.getLastStepMeanSpeed(road_id))

                road_mean_speed_kph = max(road_speed, 0.0) * 3.6



        return {

            "completed": traci.simulation.getMinExpectedNumber() == 0,

            "scenario": self.traffic_level,

            "traffic_level": self.traffic_level,

            "is_congested": (

                active_count >= 5 and halted_percent >= 20.0

            ) or mean_waiting_time >= 10.0,

            "active_vehicles": active_count,

            "halted_vehicles": halted_count,

            "average_speed_kph": round(mean_speed_kph, 2),

            "average_delay_seconds": round(mean_waiting_time, 2),

            "congestion_index_percent": round(halted_percent, 2),

            "current_road_vehicle_count": road_vehicle_count,

            "current_road_mean_speed_kph": (

                round(road_mean_speed_kph, 2)

                if road_mean_speed_kph is not None

                else None

            ),

        }



    def get_simulation_time(self) -> float:

        """Return the current SUMO simulation time in seconds."""



        self._require_running()

        return float(traci.simulation.getTime())



    def has_pending_vehicles(self) -> bool:

        """Return whether vehicles are active or still waiting to depart."""



        self._require_running()

        return traci.simulation.getMinExpectedNumber() > 0



    def track_vehicle(

        self,

        vehicle_id: str,

        *,

        view_id: str = "View #0",

        zoom: float = 3000.0,

    ) -> None:

        """Center a SUMO GUI view on an active vehicle and follow it."""



        self._require_running()

        if vehicle_id not in traci.vehicle.getIDList():

            raise ValueError(f"Vehicle '{vehicle_id}' is not currently active in SUMO.")

        if view_id not in traci.gui.getIDList():

            raise RuntimeError(

                f"SUMO GUI view '{view_id}' is unavailable. Start with use_gui=True."

            )

        traci.gui.trackVehicle(view_id, vehicle_id)

        traci.gui.setZoom(view_id, zoom)

        traci.gui.toggleSelection(vehicle_id, "vehicle")



    def stop(self) -> None:

        """Close the active TraCI connection; repeated calls are safe."""



        if not self._running:

            return

        try:

            traci.close()

        finally:

            self._running = False





def run_demo(*, use_gui: bool = False, max_steps: int = 200) -> None:

    """Run the bundled scenario and print live state until all trips finish."""



    if max_steps <= 0:

        raise ValueError("max_steps must be greater than zero.")



    simulator = SumoTrafficSimulator()

    try:

        simulator.start(use_gui=use_gui)

        steps = 0

        while simulator.has_pending_vehicles() and steps < max_steps:

            simulation_time = simulator.step()

            vehicle_ids = simulator.get_vehicle_ids()

            print(f"\nSimulation time: {simulation_time:.1f} s")

            print(f"Active vehicles: {vehicle_ids}")



            for vehicle_id in vehicle_ids:

                state = simulator.get_vehicle_state(vehicle_id)

                motion = state["motion"]

                position = motion["position_xy"]

                print(f"\n{vehicle_id}:")

                print(f"  speed: {motion['speed_kmh']:.1f} km/h")

                print(

                    f"  acceleration: {motion['acceleration_mps2']:.2f} m/s^2"

                )

                print(f"  road: {motion['road_name']}")

                print(f"  lane: {motion['lane_id']}")

                print(f"  lane position: {motion['lane_position_m']:.1f} m")

                print(f"  position: ({position['x']:.1f}, {position['y']:.1f})")

                print(f"  angle: {motion['angle_deg']:.1f} degrees")

                print(f"  waiting time: {motion['waiting_time_s']:.1f} s")



            steps += 1

            if use_gui:

                time.sleep(0.1)



        if simulator.has_pending_vehicles():

            print(f"\nStopped at the safety limit of {max_steps} steps.")

        else:

            print(f"\nAll vehicles completed their routes in {steps} steps.")

    finally:

        simulator.stop()





if __name__ == "__main__":

    run_demo(use_gui=os.getenv("RUN_SUMO_GUI_DEMO") == "1")
