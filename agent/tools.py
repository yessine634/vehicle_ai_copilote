"""Expose copilot functions as LangChain tools."""

from langchain_core.tools import BaseTool, tool

from simulation.route_energy import estimate_route_energy
from tools.routing import get_route
from tools.vehicle import VehicleStateProvider, get_vehicle_state
from tools.weather import get_weather


@tool
def weather_tool(city: str) -> dict:
    """Get current weather conditions for a city."""

    return get_weather(city)


@tool
def route_tool(origin: str, destination: str) -> dict:
    """Get driving route information between two locations."""

    return get_route(origin, destination)


def create_vehicle_tool(simulator: VehicleStateProvider) -> BaseTool:
    """Create a vehicle tool connected to an active SUMO trip."""

    @tool
    def vehicle_tool() -> dict:
        """Get the current electric-vehicle, health, and traffic state."""

        return get_vehicle_state(simulator)

    return vehicle_tool


def create_route_tool(simulator: VehicleStateProvider) -> BaseTool:
    """Create a route tool with deterministic EV feasibility estimation."""

    @tool
    def route_tool(origin: str, destination: str) -> dict:
        """Get a route plus deterministic EV energy and arrival feasibility."""

        route = get_route(origin, destination)
        vehicle = get_vehicle_state(simulator)
        segments = route.get("segments") or [
            {
                "type": "unknown",
                "distance_km": route["distance_km"],
                "average_speed_kmh": None,
            }
        ]
        route["ev_estimate"] = estimate_route_energy(
            route_segments=segments,
            battery_capacity_kwh=vehicle["ev"]["battery_capacity_kwh"],
            current_energy_kwh=vehicle["ev"]["battery_energy_kwh"],
        )
        return route

    return route_tool


def build_tools(simulator: VehicleStateProvider) -> list[BaseTool]:
    """Return all tools connected to the active SUMO simulation."""

    return [
        weather_tool,
        create_route_tool(simulator),
        create_vehicle_tool(simulator),
    ]
