"""Test LangChain selection and execution of all public copilot tools."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from typing import Any
from unittest.mock import Mock, patch

# Allow this file to run directly from the tests directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_groq import ChatGroq

from tools.routing import get_route
from tools.vehicle import VehicleStateProvider, get_vehicle_state
from tools.weather import get_weather

load_dotenv(PROJECT_ROOT / ".env")


@tool
def weather_tool(city: str) -> dict:
    """Get current weather conditions for a city."""

    return get_weather(city)


@tool
def route_tool(origin: str, destination: str) -> dict:
    """Get driving route information between an origin and destination."""

    return get_route(origin, destination)


def create_vehicle_tool(simulator: VehicleStateProvider) -> BaseTool:
    """Create a vehicle tool bound to one explicitly supplied simulator."""

    if simulator is None:
        raise ValueError("A simulator is required to create vehicle_tool.")

    @tool
    def vehicle_tool() -> dict:
        """Get current simulated electric-car motion, health, and traffic."""

        return get_vehicle_state(simulator)

    return vehicle_tool


def build_tools(
    simulator: VehicleStateProvider,
) -> tuple[list[BaseTool], dict[str, BaseTool]]:
    """Build the three tools and their name-to-tool execution map."""

    vehicle_tool = create_vehicle_tool(simulator)
    tools = [weather_tool, route_tool, vehicle_tool]
    tool_map = {selected_tool.name: selected_tool for selected_tool in tools}
    return tools, tool_map


def create_llm_with_tools(tools: list[BaseTool]):
    """Create Groq and bind the supplied LangChain tools."""

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    return llm.bind_tools(tools)


def print_console(value: Any) -> None:
    """Print text safely in Windows terminals with legacy encodings."""

    text = (
        str(value)
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2248", "approximately ")
    )
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding)
    print(safe_text)


def request_tool_selection(llm_with_tools, question: str):
    """Ask the LLM to select tools and print its first tool calls."""

    messages = [HumanMessage(content=question)]
    response = llm_with_tools.invoke(messages)

    print(f"Question: {question}")
    if not response.tool_calls:
        print("Selected tool: None")
        print("Tool arguments: {}")
    else:
        for tool_call in response.tool_calls:
            print(f"Selected tool: {tool_call['name']}")
            print(f"Tool arguments: {tool_call['args']}")

    return messages, response


def execute_tool_calls(
    messages: list,
    response: AIMessage,
    tool_map: dict[str, BaseTool],
) -> list[dict[str, Any]]:
    """Execute selected tools and append correctly linked ToolMessages."""

    messages.append(response)
    executed_calls: list[dict[str, Any]] = []

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = tool_map.get(tool_name)
        if selected_tool is None:
            raise ValueError(f"The LLM requested an unsupported tool: {tool_name}")

        result = selected_tool.invoke(tool_args)
        messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )
        executed_calls.append(
            {"name": tool_name, "arguments": tool_args, "result": result}
        )

        print(f"Executed tool: {tool_name}")
        print(f"Arguments: {tool_args}")
        print(f"Tool result: {result}")

    return executed_calls


def run_complete_tool_turn(
    llm_with_tools,
    tool_map: dict[str, BaseTool],
    question: str,
):
    """Select tools, execute them, and request the final LLM answer."""

    messages, first_response = request_tool_selection(llm_with_tools, question)
    if not first_response.tool_calls:
        return first_response, []

    executed_calls = execute_tool_calls(messages, first_response, tool_map)
    final_response = llm_with_tools.invoke(messages)

    print("Final answer:")
    print_console(final_response.content)
    return final_response, executed_calls


FAKE_SIMULATOR_STATE = {
    "status": "running",
    "movement": {
        "speed_kph": 58.0,
        "position": {"latitude": 34.75, "longitude": 10.76},
        "current_road": "Route El Ain",
        "road_name": "Route El Ain",
        "road_id": "729752355#3",
        "elapsed_seconds": 300.0,
        "travelled_distance_km": 4.8,
        "trip_progress_percent": 31.0,
    },
    "ev": {
        "model": "Hyundai IONIQ 5 Long Range RWD",
        "drive": "RWD",
        "battery_soc_percent": 72.0,
        "battery_capacity_kwh": 84.0,
        "battery_energy_kwh": 60.48,
        "battery_low": False,
        "instant_consumption_wh_per_s": 4.2,
        "total_energy_consumed_kwh": 1.4,
        "total_energy_regenerated_kwh": 0.18,
        "reference_consumption_kwh_100km": 16.0,
        "reference_range_km": 570.0,
        "distance_travelled_km": 4.8,
        "actual_consumption_kwh_100km": 29.17,
        "estimated_remaining_range_km": 207.37,
        "range_estimate_source": "observed_consumption",
        "charging_station_id": None,
        "is_charging": False,
    },
    "health": {
        "source": "supplemental_simulation",
        "is_estimated": True,
        "overall_status": "normal",
        "warnings": [],
        "traction_battery_soh_percent": 99.9,
        "battery_temperature_c": 31.0,
        "motor_temperature_c": 48.0,
        "inverter_temperature_c": 42.0,
        "auxiliary_battery_voltage": 14.1,
        "brake_pad_life_percent": 99.8,
        "odometer_km": 4.8,
        "tire_pressure_psi": {
            "front_left": 32.2,
            "front_right": 32.2,
            "rear_left": 32.2,
            "rear_right": 32.2,
        },
        "tire_temperature_c": {
            "front_left": 27.0,
            "front_right": 27.0,
            "rear_left": 27.0,
            "rear_right": 27.0,
        },
    },
    "traffic": {
        "completed": True,
        "scenario": "rush_hour",
        "traffic_level": "heavy",
        "is_congested": True,
        "average_speed_kph": 21.0,
        "average_delay_seconds": 360.0,
        "average_travel_time_seconds": 720.0,
        "congestion_index_percent": 50.0,
    },
}


class FakeTripSimulator:
    """Deterministic EV state provider used without running SUMO or APIs."""

    def get_state(self) -> dict[str, Any]:
        return deepcopy(FAKE_SIMULATOR_STATE)


class AllToolsUnitTests(unittest.TestCase):
    """Test all wrappers and the tool loop without live services."""

    def setUp(self) -> None:
        self.tools, self.tool_map = build_tools(FakeTripSimulator())

    @patch(f"{__name__}.get_route")
    @patch(f"{__name__}.get_weather")
    def test_simple_prompts_select_and_execute_expected_tools(
        self,
        mock_get_weather: Mock,
        mock_get_route: Mock,
    ) -> None:
        mock_get_weather.return_value = {
            "city": "Sfax",
            "temperature_c": 25.0,
            "weather_description": "Clear sky",
        }
        mock_get_route.return_value = {
            "origin": {"name": "Sfax"},
            "destination": {"name": "Tunis"},
            "distance_km": 269.0,
            "duration_minutes": 180.0,
            "uses_highway": True,
            "has_toll": True,
            "has_ferry": False,
        }
        cases = [
            (
                "What is the weather in Sfax?",
                "weather_tool",
                {"city": "Sfax"},
            ),
            (
                "How far is Tunis from Sfax by car?",
                "route_tool",
                {"origin": "Sfax", "destination": "Tunis"},
            ),
            (
                "How is my car doing right now?",
                "vehicle_tool",
                {},
            ),
        ]

        for index, (question, expected_tool, arguments) in enumerate(cases):
            with self.subTest(question=question):
                tool_call_id = f"call_{index}"
                first_response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": expected_tool,
                            "args": arguments,
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                )
                final_response = AIMessage(
                    content=f"Final answer produced with {expected_tool}."
                )
                mocked_llm = Mock()
                mocked_llm.invoke.side_effect = [first_response, final_response]

                answer, executed_calls = run_complete_tool_turn(
                    mocked_llm,
                    self.tool_map,
                    question,
                )

                self.assertEqual(executed_calls[0]["name"], expected_tool)
                self.assertEqual(executed_calls[0]["arguments"], arguments)
                self.assertTrue(answer.content)
                second_messages = mocked_llm.invoke.call_args_list[1].args[0]
                self.assertIsInstance(second_messages[-1], ToolMessage)
                self.assertEqual(second_messages[-1].tool_call_id, tool_call_id)

    def test_vehicle_tool_uses_injected_simulator(self) -> None:
        result = self.tool_map["vehicle_tool"].invoke({})

        self.assertEqual(result["motion"]["speed_kmh"], 58.0)
        self.assertEqual(result["ev"]["battery_soc_percent"], 72.0)
        self.assertEqual(result["ev"]["battery_capacity_kwh"], 84.0)
        self.assertEqual(
            result["health"]["source"], "supplemental_simulation"
        )
        self.assertEqual(result["traffic"]["level"], "heavy")

    def test_mixed_prompt_only_reports_first_requested_tools(self) -> None:
        first_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "vehicle_tool",
                    "args": {},
                    "id": "mixed_vehicle",
                    "type": "tool_call",
                },
                {
                    "name": "route_tool",
                    "args": {"origin": "current location", "destination": "Tunis"},
                    "id": "mixed_route",
                    "type": "tool_call",
                },
            ],
        )
        mocked_llm = Mock()
        mocked_llm.invoke.return_value = first_response

        _, response = request_tool_selection(
            mocked_llm,
            "Can I safely continue driving to Tunis?",
        )

        self.assertEqual(
            [tool_call["name"] for tool_call in response.tool_calls],
            ["vehicle_tool", "route_tool"],
        )


class SumoBoundVehicleToolTests(unittest.TestCase):
    """Verify the LangChain vehicle tool reads the active SUMO session."""

    def test_vehicle_tool_reads_live_sumo_ioniq5(self) -> None:
        from simulation.trip_simulator import TripSimulator

        simulator = TripSimulator(
            origin="Route El Ain, central Sfax",
            destination="North-east central Sfax",
            traffic_level="normal",
            initial_soc_percent=80.0,
        )
        try:
            simulator.start(use_gui=False)
            simulator.step(10)
            _, tool_map = build_tools(simulator)

            result = tool_map["vehicle_tool"].invoke({})

            self.assertEqual(
                result["ev"]["model"],
                "Hyundai IONIQ 5 Long Range RWD",
            )
            self.assertEqual(result["ev"]["battery_capacity_kwh"], 84.0)
            self.assertGreater(result["motion"]["speed_kmh"], 0)
            self.assertEqual(
                result["health"]["source"],
                "supplemental_simulation",
            )
            self.assertEqual(result["traffic"]["level"], "normal")
        finally:
            simulator.stop()


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TOOL_TESTS") == "1",
    "Set RUN_LIVE_TOOL_TESTS=1 to call Groq and external APIs.",
)
class LiveAllToolsTests(unittest.TestCase):
    """Verify Groq selection using live APIs and the real Sfax SUMO trip."""

    @classmethod
    def setUpClass(cls) -> None:
        from simulation.trip_simulator import TripSimulator

        cls.simulator = TripSimulator(
            origin="Route El Ain, central Sfax",
            destination="North-east central Sfax",
            traffic_level="normal",
            initial_soc_percent=80.0,
        )
        cls.simulator.start(use_gui=False)
        cls.simulator.step(10)
        cls.tools, cls.tool_map = build_tools(cls.simulator)
        cls.llm_with_tools = create_llm_with_tools(cls.tools)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.simulator.stop()

    def _assert_complete_turn(self, question: str, expected_tool: str) -> None:
        final_response, executed_calls = run_complete_tool_turn(
            self.llm_with_tools,
            self.tool_map,
            question,
        )

        self.assertTrue(executed_calls)
        self.assertEqual(executed_calls[0]["name"], expected_tool)
        self.assertIsInstance(final_response.content, str)
        self.assertTrue(final_response.content.strip())

    def test_weather_selection_live(self) -> None:
        self._assert_complete_turn(
            "What is the weather in Sfax?",
            "weather_tool",
        )

    def test_route_selection_live(self) -> None:
        self._assert_complete_turn(
            "How far is Tunis from Sfax by car?",
            "route_tool",
        )

    def test_vehicle_selection_live(self) -> None:
        self._assert_complete_turn(
            "How is my car doing right now?",
            "vehicle_tool",
        )

    def test_mixed_prompt_first_selection_live(self) -> None:
        _, response = request_tool_selection(
            self.llm_with_tools,
            "Can I safely continue driving to Tunis?",
        )

        print(
            "First requested tools:",
            [tool_call["name"] for tool_call in response.tool_calls],
        )


if __name__ == "__main__":
    unittest.main()
