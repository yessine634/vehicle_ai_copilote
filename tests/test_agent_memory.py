"""Regression tests for the copilot's structured per-thread memory."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_copilot_graph
from agent.nodes import (
    create_final_answer_node,
    create_refresh_vehicle_node,
    prepare_selected_tool_calls,
)
from agent.prompts import SYSTEM_PROMPT
from agent.tools import build_tools
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


class MutableFakeTripSimulator:
    """Provide state whose simulation time can be advanced by a test."""

    def __init__(self) -> None:
        self.state = deepcopy(TRIP_STATE)

    def get_state(self) -> dict:
        return deepcopy(self.state)

    def advance(self, seconds: float, speed_kph: float) -> None:
        movement = self.state["movement"]
        movement["elapsed_seconds"] += seconds
        movement["speed_kph"] = speed_kph


class AgentMemoryTests(unittest.TestCase):
    def test_final_answer_node_forbids_tool_calls(self) -> None:
        fake_llm = Mock()
        fake_llm.invoke.return_value = AIMessage(content="Final answer.")

        final_answer = create_final_answer_node(fake_llm)
        result = final_answer(
            {
                "question": "Can I reach Hammamet?",
                "turn_messages": [
                    HumanMessage(content="Can I reach Hammamet?"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "route_tool",
                                "args": {
                                    "origin": "Sfax",
                                    "destination": "Hammamet",
                                },
                                "id": "route-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content='{"distance_km": 200.0}',
                        tool_call_id="route-call",
                        name="route_tool",
                    ),
                ],
                "selected_tools": ["route_tool"],
                "called_tools": ["route_tool"],
                "current_vehicle_state": deepcopy(TRIP_STATE),
            }
        )

        sent_messages = fake_llm.invoke.call_args.args[0]
        self.assertIn("Do not call tools", sent_messages[0].content)
        self.assertEqual(result["answer"], "Final answer.")

    def test_final_answer_node_rejects_late_tool_calls(self) -> None:
        fake_llm = Mock()
        fake_llm.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "route_tool",
                    "args": {
                        "origin": "Sfax",
                        "destination": "Hammamet",
                    },
                    "id": "late-route-call",
                    "type": "tool_call",
                }
            ],
        )

        final_answer = create_final_answer_node(fake_llm)

        with self.assertRaisesRegex(
            RuntimeError,
            "final-answer LLM tried to call a tool",
        ):
            final_answer(
                {
                    "question": "Can I reach Hammamet?",
                    "turn_messages": [
                        HumanMessage(content="Can I reach Hammamet?")
                    ],
                    "current_vehicle_state": deepcopy(TRIP_STATE),
                }
            )

    def test_tool_selection_comes_only_from_llm_planner(self) -> None:
        result = prepare_selected_tool_calls(
            {
                "question": "What is the weather in Sfax?",
                "turn_messages": [
                    HumanMessage(content="What is the weather in Sfax?"),
                    AIMessage(content="No tool selected."),
                ],
            }
        )

        self.assertEqual(result["selected_tools"], [])

    def test_refresh_replaces_old_messages_and_exposes_location(self) -> None:
        refresh = create_refresh_vehicle_node(FakeTripSimulator())
        old_tool_message = ToolMessage(
            content="stale weather",
            tool_call_id="old-call",
            name="weather_tool",
        )

        result = refresh(
            {
                "question": "Can I reach Hammamet?",
                "turn_messages": [old_tool_message],
            }
        )

        self.assertEqual(len(result["turn_messages"]), 1)
        self.assertIsInstance(result["turn_messages"][0], HumanMessage)
        self.assertEqual(
            result["turn_messages"][0].content,
            "Can I reach Hammamet?",
        )
        self.assertEqual(
            result["current_location"]["name"],
            "Route El Ain, central Sfax",
        )
        self.assertEqual(
            result["current_location"]["routing_name"],
            "Sfax",
        )
        self.assertEqual(result["planned_origin"], "Sfax")

    def test_same_time_questions_preserve_last_distinct_snapshot(self) -> None:
        fake_llm = Mock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = AIMessage(content="Vehicle state received.")

        simulator = MutableFakeTripSimulator()

        with patch("agent.graph.ChatGroq", return_value=fake_llm):
            graph = build_copilot_graph(simulator)

        config = {"configurable": {"thread_id": "driver-memory-test"}}
        first = graph.invoke({"question": "How is the car?"}, config=config)
        simulator.advance(seconds=30.0, speed_kph=42.0)
        second = graph.invoke({"question": "What changed?"}, config=config)
        third = graph.invoke(
            {"question": "Compare the states again."},
            config=config,
        )

        self.assertIsNone(first["previous_vehicle_state"])
        self.assertEqual(
            second["previous_vehicle_state"],
            first["current_vehicle_state"],
        )
        self.assertEqual(
            third["previous_vehicle_state"],
            first["current_vehicle_state"],
        )
        self.assertEqual(
            third["current_vehicle_state"],
            second["current_vehicle_state"],
        )
        self.assertEqual(len(third["turn_messages"]), 3)
        self.assertEqual(
            third["turn_messages"][0].content,
            "Compare the states again.",
        )

    @patch("agent.tools.get_weather")
    def test_weather_is_called_again_for_each_weather_question(
        self,
        mock_weather: Mock,
    ) -> None:
        mock_weather.return_value = {
            "city": "Sfax",
            "temperature_c": 25.0,
            "weather_description": "Clear sky",
        }
        fake_llm = Mock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "weather_tool",
                        "args": {"city": "Sfax"},
                        "id": "weather-one",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Fresh weather one."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "weather_tool",
                        "args": {"city": "Sfax"},
                        "id": "weather-two",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Fresh weather two."),
        ]

        with patch("agent.graph.ChatGroq", return_value=fake_llm):
            graph = build_copilot_graph(FakeTripSimulator())

        config = {"configurable": {"thread_id": "weather-freshness-test"}}
        graph.invoke({"question": "What is the weather in Sfax?"}, config=config)
        graph.invoke(
            {"question": "Has the weather in Sfax changed?"},
            config=config,
        )

        self.assertEqual(mock_weather.call_count, 2)

    @patch("agent.tools.get_route")
    def test_route_is_called_again_for_each_route_question(
        self,
        mock_route: Mock,
    ) -> None:
        mock_route.return_value = {
            "origin": {"name": "Sfax"},
            "destination": {"name": "Hammamet"},
            "distance_km": 200.0,
            "duration_minutes": 150.0,
            "uses_highway": True,
            "has_toll": True,
            "has_ferry": False,
        }

        def route_call(call_id: str) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "route_tool",
                        "args": {
                            "origin": "Sfax",
                            "destination": "Hammamet",
                        },
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )

        fake_llm = Mock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.side_effect = [
            route_call("route-one"),
            AIMessage(content="Fresh route one."),
            route_call("route-two"),
            AIMessage(content="Fresh route two."),
        ]

        with patch("agent.graph.ChatGroq", return_value=fake_llm):
            graph = build_copilot_graph(FakeTripSimulator())

        config = {"configurable": {"thread_id": "route-freshness-test"}}
        graph.invoke({"question": "Can I reach Hammamet?"}, config=config)
        graph.invoke(
            {"question": "How long is the trip to Hammamet?"},
            config=config,
        )

        self.assertEqual(mock_route.call_count, 2)

    @patch("agent.tools.get_route")
    def test_route_call_stores_compact_plan_facts(self, mock_route: Mock) -> None:
        mock_route.return_value = {
            "origin": {"name": "Sfax"},
            "destination": {"name": "Hammamet"},
            "distance_km": 200.0,
            "duration_minutes": 150.0,
            "uses_highway": True,
            "has_toll": True,
            "has_ferry": False,
        }

        fake_llm = Mock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "route_tool",
                        "args": {
                            "origin": "Sfax",
                            "destination": "Hammamet",
                        },
                        "id": "route-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The route is available."),
        ]

        with patch("agent.graph.ChatGroq", return_value=fake_llm):
            graph = build_copilot_graph(FakeTripSimulator())

        result = graph.invoke(
            {"question": "Can I drive from Sfax to Hammamet?"},
            config={"configurable": {"thread_id": "route-memory-test"}},
        )

        self.assertEqual(result["planned_origin"], "Sfax")
        self.assertEqual(result["planned_destination"], "Hammamet")
        self.assertEqual(result["answer"], "The route is available.")

    @patch("agent.tools.get_route")
    def test_destination_only_route_replaces_invented_origin(
        self,
        mock_route: Mock,
    ) -> None:
        mock_route.return_value = {
            "origin": {"name": "Sfax"},
            "destination": {"name": "Hammamet"},
            "distance_km": 200.0,
            "duration_minutes": 150.0,
            "uses_highway": True,
            "has_toll": True,
            "has_ferry": False,
        }

        fake_llm = Mock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "route_tool",
                        "args": {
                            "origin": "Sousse",
                            "destination": "Hammamet",
                        },
                        "id": "destination-only-route",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="You can reach Hammamet from Sfax."),
        ]

        with patch("agent.graph.ChatGroq", return_value=fake_llm):
            graph = build_copilot_graph(FakeTripSimulator())

        result = graph.invoke(
            {"question": "Can I reach Hammamet?"},
            config={"configurable": {"thread_id": "origin-guard-test"}},
        )

        mock_route.assert_called_once_with("Sfax", "Hammamet")
        self.assertEqual(result["planned_origin"], "Sfax")
        self.assertEqual(result["planned_destination"], "Hammamet")

    def test_graph_binds_only_fresh_external_tools(self) -> None:
        self.assertEqual(
            [
                selected_tool.name
                for selected_tool in build_tools(FakeTripSimulator())
            ],
            ["weather_tool", "route_tool", "vehicle_tool"],
        )
        self.assertIn("never invent an origin", SYSTEM_PROMPT.lower())
        self.assertIn("can reach", SYSTEM_PROMPT.lower())
        self.assertIn("route_tool and weather_tool", SYSTEM_PROMPT.lower())
        self.assertIn("do not call a tool", SYSTEM_PROMPT.lower())
        self.assertIn("all required tools in the same response", SYSTEM_PROMPT.lower())
        self.assertIn("current battery and current weather", SYSTEM_PROMPT.lower())

    def test_graph_enables_parallel_tool_calls(self) -> None:
        fake_llm = Mock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = AIMessage(content="No tools needed.")

        with patch("agent.graph.ChatGroq", return_value=fake_llm):
            build_copilot_graph(FakeTripSimulator())

        fake_llm.bind_tools.assert_called_once()
        _, bind_kwargs = fake_llm.bind_tools.call_args
        self.assertEqual(bind_kwargs["tool_choice"], "auto")
        self.assertIs(bind_kwargs["parallel_tool_calls"], True)


if __name__ == "__main__":
    unittest.main()
