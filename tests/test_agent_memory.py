"""Tests for persistent LangGraph session context and LLM message filtering."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_copilot_graph
from agent.nodes import (
    extract_explicit_route_context,
    filter_messages_for_llm,
)
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


class MessageFilteringTests(unittest.TestCase):
    def test_last_ten_messages_do_not_modify_full_history(self) -> None:
        messages = [
            HumanMessage(content=f"Message {index}")
            for index in range(12)
        ]
        original_ids = [message.id for message in messages]

        filtered = filter_messages_for_llm(messages)

        self.assertEqual(len(filtered), 10)
        self.assertEqual(filtered, messages[-10:])
        self.assertEqual([message.id for message in messages], original_ids)
        self.assertEqual(len(messages), 12)

    def test_filter_preserves_complete_tool_call_result_pair(self) -> None:
        route_request = tool_call(
            "route_tool",
            {"origin": "Sfax", "destination": "Tunis"},
            "route-call",
        )
        route_result_message = ToolMessage(
            content='{"distance_km": 270}',
            tool_call_id="route-call",
            name="route_tool",
        )
        messages = [
            HumanMessage(content=f"Old {index}") for index in range(8)
        ] + [
            route_request,
            route_result_message,
            AIMessage(content="The route is ready."),
        ]

        filtered = filter_messages_for_llm(messages)

        self.assertLessEqual(len(filtered), 10)
        self.assertIn(route_request, filtered)
        self.assertIn(route_result_message, filtered)
        self.assertLess(
            filtered.index(route_request),
            filtered.index(route_result_message),
        )

    def test_orphan_tool_result_is_not_sent_to_llm(self) -> None:
        orphan = ToolMessage(
            content="old result",
            tool_call_id="missing-call",
            name="weather_tool",
        )

        filtered = filter_messages_for_llm(
            [orphan, HumanMessage(content="Hello")]
        )

        self.assertNotIn(orphan, filtered)


class RouteContextExtractionTests(unittest.TestCase):
    def test_explicit_origin_and_destination_are_extracted(self) -> None:
        self.assertEqual(
            extract_explicit_route_context(
                "Set my origin to Sfax and my destination to Tunis."
            ),
            {"origin": "Sfax", "destination": "Tunis"},
        )

    def test_destination_change_does_not_clear_origin(self) -> None:
        self.assertEqual(
            extract_explicit_route_context(
                "Change my destination to Hammamet."
            ),
            {"destination": "Hammamet"},
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

    def test_origin_and_destination_persist_in_same_thread(self) -> None:
        graph, _, bound_llm = self.build_graph(
            [
                AIMessage(content="Route context saved."),
                AIMessage(content="Regenerative braking recovers energy."),
            ]
        )
        thread = "route-context-persistence"

        self.invoke(
            graph,
            "Set my origin to Sfax and my destination to Tunis.",
            thread,
        )
        result = self.invoke(
            graph,
            "Explain regenerative braking.",
            thread,
        )

        self.assertEqual(result["origin"], "Sfax")
        self.assertEqual(result["destination"], "Tunis")
        second_prompt = bound_llm.invoke.call_args_list[1].args[0][0].content
        self.assertIn("Origin: Sfax", second_prompt)
        self.assertIn("Destination: Tunis", second_prompt)

    def test_new_destination_overwrites_only_destination(self) -> None:
        graph, _, _ = self.build_graph(
            [
                AIMessage(content="Route context saved."),
                AIMessage(content="Destination updated."),
            ]
        )
        thread = "destination-update"

        self.invoke(
            graph,
            "My origin is Sfax and my destination is Tunis.",
            thread,
        )
        result = self.invoke(
            graph,
            "Change my destination to Hammamet.",
            thread,
        )

        self.assertEqual(result["origin"], "Sfax")
        self.assertEqual(result["destination"], "Hammamet")

    def test_full_history_is_stored_but_only_ten_messages_reach_llm(self) -> None:
        graph, _, bound_llm = self.build_graph(
            [AIMessage(content=f"Answer {index}") for index in range(6)]
        )
        thread = "filtered-history"

        result = None
        for index in range(6):
            result = self.invoke(graph, f"Question {index}", thread)

        self.assertIsNotNone(result)
        self.assertEqual(len(result["messages"]), 12)
        last_llm_input = bound_llm.invoke.call_args.args[0]
        self.assertEqual(len(last_llm_input), 11)  # system + 10 messages

    @patch("agent.tools.get_weather")
    @patch("agent.tools.get_route")
    def test_sequential_live_tools_keep_normal_react_loop(
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
                AIMessage(content="Fresh route and weather are available."),
            ]
        )

        result = self.invoke(
            graph,
            "Check the route from Sfax to Tunis and then its weather.",
            "sequential-live-tools",
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
