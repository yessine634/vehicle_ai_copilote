"""Tests for the LangGraph ReAct copilot workflow."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages

from agent.graph import build_copilot_graph
from agent.nodes import MEMORY_SUMMARY_PREFIX, create_memory_node
from agent.tools import build_tools, create_route_tool
from tests.test_vehicle import SIMULATOR_STATE


TRIP_STATE = {
    **deepcopy(SIMULATOR_STATE),
    "origin": {
        "name": "Route El Ain, central Sfax",
        "city": "Sfax",
        "country": "Tunisia",
        "latitude": 34.7518952,
        "longitude": 10.7296079,
    },
    "destination": {
        "name": "North-east central Sfax",
        "city": "Sfax",
        "country": "Tunisia",
        "latitude": 34.762,
        "longitude": 10.746,
    },
    "route": {
        "distance_km": 1.824,
        "duration_minutes": None,
        "uses_highway": False,
        "has_toll": False,
        "has_ferry": False,
    },
}


class FakeTripSimulator:
    """Provide deterministic trip state without starting SUMO."""

    def get_state(self) -> dict:
        return deepcopy(TRIP_STATE)


class MemoryNodeTests(unittest.TestCase):
    def test_history_at_limit_is_not_summarized(self) -> None:
        llm = Mock()
        node = create_memory_node(llm)
        messages = [
            HumanMessage(content=f"Question {index}")
            for index in range(5)
        ]

        self.assertEqual(node({"messages": messages}), {})
        llm.invoke.assert_not_called()

    def test_oversized_history_is_summarized_by_llm(self) -> None:
        llm = Mock()
        llm.invoke.return_value = AIMessage(
            content="The driver is planning a trip from Sfax to Tunis."
        )
        node = create_memory_node(llm)
        messages = [
            HumanMessage(content="I want to travel from Sfax to Tunis."),
            tool_call("vehicle_tool", {}, "old-vehicle-call"),
            ToolMessage(
                content='{"speed_kmh": 40, "battery_soc": 79.6}',
                tool_call_id="old-vehicle-call",
                name="vehicle_tool",
            ),
            AIMessage(content="Your speed is 40 and SOC is 79.6 percent."),
            HumanMessage(content="Please remember my destination."),
            AIMessage(content="Your destination is Tunis."),
            HumanMessage(content="What is the weather there now?"),
        ]

        update = node({"messages": messages})
        compacted = add_messages(messages, update["messages"])

        self.assertEqual(len(compacted), 2)
        self.assertIsInstance(compacted[0], SystemMessage)
        self.assertTrue(
            compacted[0].content.startswith(MEMORY_SUMMARY_PREFIX)
        )
        self.assertEqual(
            compacted[1].content,
            "What is the weather there now?",
        )
        summary_input = llm.invoke.call_args.args[0]
        self.assertFalse(
            any(isinstance(message, ToolMessage) for message in summary_input)
        )
        self.assertNotIn(
            "79.6",
            "\n".join(str(message.content) for message in summary_input),
        )

    def test_active_tool_protocol_is_not_compacted(self) -> None:
        llm = Mock()
        node = create_memory_node(llm)
        messages = [
            HumanMessage(content="Old question"),
            AIMessage(content="Old answer"),
            HumanMessage(content="Current question"),
            tool_call("vehicle_tool", {}, "current-vehicle-call"),
            ToolMessage(
                content='{"speed_kmh": 40}',
                tool_call_id="current-vehicle-call",
                name="vehicle_tool",
            ),
        ]

        self.assertEqual(node({"messages": messages}), {})
        llm.invoke.assert_not_called()


def tool_call(name: str, arguments: dict, call_id: str) -> AIMessage:
    """Return an AI message containing one native tool call."""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class ReActGraphTests(unittest.TestCase):
    def build_graph(self, responses: list[AIMessage]):
        fake_llm = Mock()
        bound_llm = Mock()
        fake_llm.bind_tools.return_value = bound_llm
        bound_llm.invoke.side_effect = responses

        patcher = patch("agent.graph.ChatGroq", return_value=fake_llm)
        patcher.start()
        self.addCleanup(patcher.stop)

        graph = build_copilot_graph(FakeTripSimulator())
        return graph, fake_llm, bound_llm

    def invoke(self, graph, question: str, thread: str):
        return graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread}},
        )

    def test_no_tool_needed_ends_after_assistant(self) -> None:
        graph, fake_llm, bound_llm = self.build_graph(
            [AIMessage(content="Regenerative braking recovers kinetic energy.")]
        )

        result = self.invoke(
            graph,
            "Explain regenerative braking briefly.",
            "no-tool-test",
        )

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(
            result["messages"][-1].content,
            "Regenerative braking recovers kinetic energy.",
        )
        self.assertEqual(bound_llm.invoke.call_count, 1)
        fake_llm.bind_tools.assert_called_once()

    def test_one_vehicle_tool_call_loops_back_to_assistant(self) -> None:
        graph, _, bound_llm = self.build_graph(
            [
                tool_call("vehicle_tool", {}, "vehicle-call"),
                AIMessage(content="The simulated vehicle is operating normally."),
            ]
        )

        result = self.invoke(
            graph,
            "How is my car doing?",
            "one-tool-test",
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual([message.name for message in tool_messages], ["vehicle_tool"])
        self.assertEqual(bound_llm.invoke.call_count, 2)
        self.assertEqual(
            result["messages"][-1].content,
            "The simulated vehicle is operating normally.",
        )

    @patch("agent.tools.get_weather")
    @patch("agent.tools.get_route")
    def test_sequential_route_then_weather_tools(
        self,
        mock_route: Mock,
        mock_weather: Mock,
    ) -> None:
        mock_route.return_value = route_result()
        mock_weather.return_value = {
            "city": "Tunis",
            "temperature_c": 24.0,
            "weather_description": "Clear sky",
        }
        graph, _, bound_llm = self.build_graph(
            [
                tool_call(
                    "route_tool",
                    {"origin": "Sfax", "destination": "Tunis"},
                    "route-call",
                ),
                tool_call(
                    "weather_tool",
                    {"city": "Tunis"},
                    "weather-call",
                ),
                AIMessage(content="The route and destination weather are available."),
            ]
        )

        result = self.invoke(
            graph,
            "Check the route to Tunis and then its weather.",
            "sequential-tools-test",
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(
            [message.name for message in tool_messages],
            ["route_tool", "weather_tool"],
        )
        self.assertEqual(bound_llm.invoke.call_count, 3)
        mock_route.assert_called_once_with("Sfax", "Tunis")
        mock_weather.assert_called_once_with("Tunis")

    @patch("agent.tools.get_route")
    def test_vehicle_and_route_tools_share_normal_react_loop(
        self,
        mock_route: Mock,
    ) -> None:
        mock_route.return_value = route_result()
        multi_tool_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "vehicle_tool",
                    "args": {},
                    "id": "vehicle-call",
                    "type": "tool_call",
                },
                {
                    "name": "route_tool",
                    "args": {"origin": "Sfax", "destination": "Tunis"},
                    "id": "route-call",
                    "type": "tool_call",
                },
            ],
        )
        graph, _, bound_llm = self.build_graph(
            [
                multi_tool_message,
                AIMessage(content="The current battery is sufficient for the route."),
            ]
        )

        result = self.invoke(
            graph,
            "Can my current vehicle reach Tunis?",
            "vehicle-route-test",
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertCountEqual(
            [message.name for message in tool_messages],
            ["vehicle_tool", "route_tool"],
        )
        self.assertEqual(bound_llm.invoke.call_count, 2)

    @patch("agent.tools.get_weather")
    @patch("agent.tools.get_route")
    def test_oversized_history_is_summarized_before_next_live_question(
        self,
        mock_route: Mock,
        mock_weather: Mock,
    ) -> None:
        mock_route.return_value = route_result()
        mock_weather.return_value = {
            "city": "Sfax",
            "temperature_c": 24.0,
            "weather_description": "Partly cloudy",
        }
        graph, fake_llm, bound_llm = self.build_graph(
            [
                tool_call(
                    "route_tool",
                    {"origin": "Sfax", "destination": "Tunis"},
                    "first-route-call",
                ),
                tool_call(
                    "weather_tool",
                    {"city": "Tunis"},
                    "first-weather-call",
                ),
                AIMessage(content="First route and weather answer."),
                tool_call(
                    "weather_tool",
                    {"city": "Tunis"},
                    "fresh-weather-call",
                ),
                AIMessage(content="Fresh weather answer."),
            ]
        )
        fake_llm.invoke.return_value = AIMessage(
            content="The driver is planning a trip from Sfax to Tunis."
        )
        thread = "fresh-weather-memory-test"

        self.invoke(
            graph,
            "Check the route from Sfax to Tunis and its weather.",
            thread,
        )
        result = self.invoke(graph, "What is the weather there now?", thread)

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(
            [message.tool_call_id for message in tool_messages],
            ["fresh-weather-call"],
        )
        self.assertEqual(mock_weather.call_count, 2)
        self.assertEqual(fake_llm.invoke.call_count, 1)
        self.assertEqual(bound_llm.invoke.call_count, 5)

    @patch("agent.tools.get_route")
    def test_bound_route_tool_keeps_energy_estimate(self, mock_route: Mock) -> None:
        mock_route.return_value = route_result()

        result = create_route_tool(FakeTripSimulator()).invoke(
            {"origin": "Sfax", "destination": "Tunis"}
        )

        self.assertEqual(result["ev_estimate"]["method"], "route_aware_baseline")
        self.assertIn("can_reach_with_reserve", result["ev_estimate"]["feasibility"])

    def test_existing_tools_are_bound_without_duplicates(self) -> None:
        self.assertEqual(
            [tool.name for tool in build_tools(FakeTripSimulator())],
            ["weather_tool", "route_tool", "vehicle_tool"],
        )


def route_result() -> dict:
    return {
        "origin": {"name": "Sfax"},
        "destination": {"name": "Tunis"},
        "distance_km": 110.0,
        "duration_minutes": 80.0,
        "uses_highway": True,
        "has_toll": True,
        "has_ferry": False,
        "segments": [
            {"type": "urban", "distance_km": 10.0},
            {"type": "highway", "distance_km": 100.0},
        ],
    }


if __name__ == "__main__":
    unittest.main()
