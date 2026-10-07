"""Expose copilot functions as LangChain tools."""

from langchain_core.tools import BaseTool, tool

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


def build_tools(simulator: VehicleStateProvider) -> list[BaseTool]:
    """Return all tools connected to the active SUMO simulation."""

    return [
        weather_tool,
        route_tool,
        create_vehicle_tool(simulator),
    ]
