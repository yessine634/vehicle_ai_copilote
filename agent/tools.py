"""Expose copilot functions as LangChain tools."""

from collections.abc import Mapping

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime

from simulation.energy_estimation import (
    build_energy_estimation_graph,
    energy_result,
)
from tools.routing import get_route
from tools.vehicle import VehicleStateProvider, get_vehicle_state
from tools.weather import get_weather


def compact_route_result(route: Mapping) -> dict:
    """Return the small navigation summary that is safe to send to the LLM.

    Detailed Valhalla segments remain available to internal consumers such as
    the energy-estimation subgraph, which calls ``get_route`` directly.
    """

    road_types: dict[str, dict[str, float]] = {}
    raw_segments = route.get("segments")
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, Mapping):
                continue
            road_type = str(segment.get("type") or "unknown").casefold()
            try:
                distance_km = max(float(segment.get("distance_km", 0.0)), 0.0)
            except (TypeError, ValueError):
                continue

            summary = road_types.setdefault(
                road_type,
                {
                    "distance_km": 0.0,
                    "speed_distance_product": 0.0,
                    "speed_distance_km": 0.0,
                },
            )
            summary["distance_km"] += distance_km

            try:
                average_speed = float(segment.get("average_speed_kmh"))
            except (TypeError, ValueError):
                average_speed = 0.0
            if average_speed > 0 and distance_km > 0:
                summary["speed_distance_product"] += average_speed * distance_km
                summary["speed_distance_km"] += distance_km

    road_type_summary = []
    for road_type, summary in road_types.items():
        item = {
            "type": road_type,
            "distance_km": round(summary["distance_km"], 3),
        }
        if summary["speed_distance_km"] > 0:
            item["average_speed_kmh"] = round(
                summary["speed_distance_product"]
                / summary["speed_distance_km"],
                1,
            )
        road_type_summary.append(item)

    def location_name(value) -> str:
        if isinstance(value, Mapping):
            return str(value.get("name") or "unknown")
        return str(value or "unknown")

    return {
        "origin": location_name(route.get("origin")),
        "destination": location_name(route.get("destination")),
        "distance_km": route.get("distance_km"),
        "duration_minutes": route.get("duration_minutes"),
        "uses_highway": bool(route.get("uses_highway", False)),
        "has_toll": bool(route.get("has_toll", False)),
        "has_ferry": bool(route.get("has_ferry", False)),
        "road_type_summary": road_type_summary,
    }


@tool
def weather_tool(city: str) -> dict:
    """Get current weather conditions for a city."""

    return get_weather(city)


@tool
def route_tool(origin: str, destination: str) -> dict:
    """Get a compact driving-route summary between two locations."""

    return compact_route_result(get_route(origin, destination))


def create_vehicle_tool(simulator: VehicleStateProvider) -> BaseTool:
    """Create a vehicle tool connected to an active SUMO trip."""

    @tool
    def vehicle_tool() -> dict:
        """Get the current electric-vehicle, health, and traffic state."""

        return get_vehicle_state(simulator)

    return vehicle_tool


def create_energy_consumption_tool(
    simulator: VehicleStateProvider,
    *,
    energy_graph=None,
) -> BaseTool:
    """Create the ML energy tool that explicitly invokes its subgraph."""

    graph = energy_graph or build_energy_estimation_graph(simulator)

    @tool
    def energy_consumption_kwh(
        runtime: ToolRuntime,
        origin: str | None = None,
        destination: str | None = None,
        reserve_soc_percent: float = 15.0,
        driver_max_speed_kmh: float | None = None,
    ) -> dict:
        """Estimate trip energy, arrival SOC, and battery sufficiency.

        Select this tool directly for any question about energy consumption,
        energy required for a journey, battery sufficiency, trip feasibility,
        or projected arrival charge. Do not call route_tool first: this tool
        invokes its own subgraph and fetches a fresh detailed route, weather,
        and vehicle state internally. Origin and destination may be omitted
        when saved in conversation State. Missing route grade and driving
        profiles use configured flat-road and steady-speed defaults.
        """

        saved_state = runtime.state
        if not isinstance(saved_state, Mapping):
            saved_state = {}
        resolved_origin = origin or saved_state.get("origin")
        resolved_destination = destination or saved_state.get("destination")
        calculation = graph.invoke(
            {
                "origin": resolved_origin,
                "destination": resolved_destination,
                "reserve_soc_percent": reserve_soc_percent,
                "driver_max_speed_kmh": driver_max_speed_kmh,
                "warnings": [],
                "errors": [],
            }
        )
        return energy_result(calculation)

    return energy_consumption_kwh


def create_route_tool(simulator: VehicleStateProvider) -> BaseTool:
    """Create the compact navigation tool used by the main copilot LLM."""

    # Keep the existing factory signature because callers construct every
    # simulator-connected tool through one interface. Route details themselves
    # come from Valhalla; the energy subgraph reads the simulator separately.
    del simulator

    @tool
    def route_tool(origin: str, destination: str) -> dict:
        """Get compact navigation distance, duration, and road information.

        Use only for navigation questions. Do not use this tool as a
        prerequisite for trip-energy, arrival-SOC, battery-sufficiency, or
        trip-feasibility questions; energy_consumption_kwh retrieves its own
        detailed route internally.
        """

        return compact_route_result(get_route(origin, destination))

    return route_tool


def build_tools(simulator: VehicleStateProvider) -> list[BaseTool]:
    """Return all tools connected to the active SUMO simulation."""

    return [
        weather_tool,
        create_route_tool(simulator),
        create_vehicle_tool(simulator),
        create_energy_consumption_tool(simulator),
    ]
