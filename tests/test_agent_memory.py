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
from agent.tools import build_tools, compact_route_result, create_route_tool
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
    def test_last_ten_conversational_messages_do_not_modify_full_history(self) -> None:
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

        self.assertEqual(len(filtered), 11)
        self.assertIn(route_request, filtered)
        self.assertIn(route_result_message, filtered)
        self.assertLess(
            filtered.index(route_request),
            filtered.index(route_result_message),
        )

    def test_all_current_turn_tool_pairs_are_kept_beyond_limit(self) -> None:
        question = HumanMessage(content="Run a long multi-tool workflow")
        interactions = []
        for number in range(12):
            request = tool_call("route_tool", {}, f"call-{number}")
            result = ToolMessage(
                content=f"result-{number}",
                tool_call_id=f"call-{number}",
                name="route_tool",
            )
            interactions.extend([request, result])

        filtered = filter_messages_for_llm([question, *interactions])

        self.assertEqual(filtered, [question, *interactions])
        self.assertEqual(len(filtered), 25)

    def test_multiple_calls_in_one_ai_message_require_all_matching_results(self) -> None:
        request = AIMessage(
            content="",
            tool_calls=[
                {"name": "route_tool", "args": {}, "id": "r1", "type": "tool_call"},
                {"name": "weather_tool", "args": {}, "id": "w1", "type": "tool_call"},
            ],
        )
        route_result = ToolMessage(content="route", tool_call_id="r1", name="route_tool")
        weather_result = ToolMessage(content="weather", tool_call_id="w1", name="weather_tool")
        question = HumanMessage(content="Route and weather")

        filtered = filter_messages_for_llm([question, request, route_result, weather_result])

        self.assertEqual(filtered, [question, request, route_result, weather_result])

        incomplete = filter_messages_for_llm([question, request, route_result])
        self.assertEqual(incomplete, [question])

    def test_conversation_and_current_tools_merge_chronologically(self) -> None:
        question = HumanMessage(content="Current question")
        request = tool_call("route_tool", {}, "current")
        result = ToolMessage(content="route", tool_call_id="current", name="route_tool")
        answer = AIMessage(content="Current answer")

        self.assertEqual(
            filter_messages_for_llm([question, request, result, answer]),
            [question, request, result, answer],
        )

    def test_historical_tools_excluded_but_recent_plain_conversation_kept(self) -> None:
        old_request = tool_call("route_tool", {}, "old")
        old_result = ToolMessage(content="stale", tool_call_id="old", name="route_tool")
        history = [HumanMessage(content=f"h{i}") for i in range(11)]
        current = HumanMessage(content="current")

        filtered = filter_messages_for_llm(
            [history[0], old_request, old_result, *history[1:], current]
        )

        self.assertNotIn(old_request, filtered)
        self.assertNotIn(old_result, filtered)
        self.assertEqual(len(filtered), 10)
        self.assertEqual(filtered[-1], current)

    def test_latest_user_message_kept_with_long_tool_loop(self) -> None:
        question = HumanMessage(content="Latest user question")
        interactions = []
        for number in range(15):
            interactions.extend([
                tool_call("route_tool", {}, f"loop-{number}"),
                ToolMessage(content="ok", tool_call_id=f"loop-{number}", name="route_tool"),
            ])
        filtered = filter_messages_for_llm([question, *interactions])
        self.assertEqual(filtered[0], question)
        self.assertEqual(len(filtered), 31)

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

    def test_old_tool_payload_is_not_resent_on_a_new_turn(self) -> None:
        old_request = tool_call(
            "energy_consumption_kwh",
            {"origin": "Sfax", "destination": "Tunis"},
            "old-energy-call",
        )
        old_result = ToolMessage(
            content='{"estimated_energy_consumption_kwh": 40}',
            tool_call_id="old-energy-call",
            name="energy_consumption_kwh",
        )
        latest_question = HumanMessage(content="Explain regenerative braking.")

        filtered = filter_messages_for_llm(
            [
                HumanMessage(content="Can I reach Tunis?"),
                old_request,
                old_result,
                AIMessage(content="The old estimate was available."),
                latest_question,
            ]
        )

        self.assertNotIn(old_request, filtered)
        self.assertNotIn(old_result, filtered)
        self.assertIn(latest_question, filtered)

    def test_current_turn_tool_pair_is_preserved(self) -> None:
        latest_question = HumanMessage(content="Recalculate my trip energy.")
        current_request = tool_call(
            "energy_consumption_kwh",
            {},
            "current-energy-call",
        )
        current_result = ToolMessage(
            content='{"estimated_energy_consumption_kwh": 35}',
            tool_call_id="current-energy-call",
            name="energy_consumption_kwh",
        )

        filtered = filter_messages_for_llm(
            [latest_question, current_request, current_result]
        )

        self.assertEqual(
            filtered,
            [latest_question, current_request, current_result],
        )

    def test_long_history_keeps_only_current_complete_tool_groups(self) -> None:
        history = []
        for index in range(12):
            request = tool_call("route_tool", {}, f"historical-{index}")
            result = ToolMessage(
                content=f"stale-route-{index}",
                tool_call_id=f"historical-{index}",
                name="route_tool",
            )
            history.extend([
                HumanMessage(content=f"Earlier question {index}"),
                request,
                result,
                AIMessage(content=f"Earlier answer {index}"),
            ])

        question = HumanMessage(content="Check the route and weather again")
        parallel_request = AIMessage(
            content="",
            tool_calls=[
                {"name": "route_tool", "args": {}, "id": "new-route", "type": "tool_call"},
                {"name": "weather_tool", "args": {}, "id": "new-weather", "type": "tool_call"},
            ],
        )
        # Tool results may arrive in a different order from the requests.
        weather_result = ToolMessage(
            content="fresh-weather", tool_call_id="new-weather", name="weather_tool"
        )
        route_result = ToolMessage(
            content="fresh-route", tool_call_id="new-route", name="route_tool"
        )
        messages = [
            *history, question, parallel_request, weather_result, route_result
        ]

        filtered = filter_messages_for_llm(messages)

        recent_conversation = [
            message for message in messages
            if isinstance(message, HumanMessage)
            or isinstance(message, AIMessage) and not message.tool_calls
        ][-10:]
        self.assertEqual(
            filtered,
            [*recent_conversation, parallel_request, weather_result, route_result],
        )
        self.assertEqual(len(messages), 52)
        self.assertFalse(any(
            isinstance(message, ToolMessage) and "stale" in message.content
            for message in filtered
        ))


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
    def test_bound_route_tool_returns_compact_summary(self, mock_route: Mock) -> None:
        mock_route.return_value = route_result()

        result = create_route_tool(FakeTripSimulator()).invoke(
            {"origin": "Sfax", "destination": "Tunis"}
        )

        self.assertEqual(result["origin"], "Sfax")
        self.assertEqual(result["destination"], "Tunis")
        self.assertEqual(result["distance_km"], 110.0)
        self.assertNotIn("segments", result)
        self.assertNotIn("ev_estimate", result)
        self.assertEqual(
            result["road_type_summary"],
            [
                {"type": "urban", "distance_km": 10.0},
                {"type": "highway", "distance_km": 100.0},
            ],
        )

    def test_compact_route_aggregates_repeated_road_types(self) -> None:
        detailed = route_result()
        detailed["segments"] = [
            {
                "type": "urban",
                "distance_km": 4.0,
                "average_speed_kmh": 30.0,
                "road_names": ["Local road"],
            },
            {
                "type": "highway",
                "distance_km": 100.0,
                "average_speed_kmh": 100.0,
                "road_names": ["A1"],
            },
            {
                "type": "urban",
                "distance_km": 6.0,
                "average_speed_kmh": 50.0,
                "road_names": ["Another local road"],
            },
        ]

        result = compact_route_result(detailed)

        self.assertEqual(
            result["road_type_summary"],
            [
                {
                    "type": "urban",
                    "distance_km": 10.0,
                    "average_speed_kmh": 42.0,
                },
                {
                    "type": "highway",
                    "distance_km": 100.0,
                    "average_speed_kmh": 100.0,
                },
            ],
        )
        self.assertNotIn("road_names", str(result))

    def test_existing_tools_are_bound_without_duplicates(self) -> None:
        tools = build_tools(FakeTripSimulator())
        self.assertEqual(
            [tool.name for tool in tools],
            [
                "weather_tool",
                "route_tool",
                "vehicle_tool",
                "energy_consumption_kwh",
            ],
        )
        descriptions = {tool.name: tool.description for tool in tools}
        self.assertIn(
            "Do not call route_tool first",
            descriptions["energy_consumption_kwh"],
        )
        self.assertIn("Use only for navigation", descriptions["route_tool"])
        self.assertIn("prerequisite", descriptions["route_tool"])

    @patch("agent.tools.build_energy_estimation_graph")
    def test_energy_tool_uses_saved_context_in_existing_react_loop(
        self,
        mock_build_energy_graph: Mock,
    ) -> None:
        fake_energy_graph = Mock()
        fake_energy_graph.invoke.return_value = {
            "origin": "Sfax",
            "destination": "Tunis",
            "reserve_soc_percent": 15.0,
            "route_segments": [],
            "total_distance_km": 270.0,
            "total_predicted_energy_kwh": 30.0,
            "available_energy_kwh": 33.0,
            "reserve_energy_kwh": 6.33,
            "energy_available_after_reserve_kwh": 26.67,
            "energy_after_trip_kwh": 3.0,
            "projected_arrival_soc_percent": 7.1,
            "feasible": False,
            "warnings": ["mock estimate"],
            "errors": [],
        }
        mock_build_energy_graph.return_value = fake_energy_graph
        graph, _, bound_llm = self.build_graph(
            [
                tool_call(
                    "energy_consumption_kwh",
                    {},
                    "energy-call",
                ),
                AIMessage(content="The fresh estimate is not sufficient."),
            ]
        )

        result = self.invoke(
            graph,
            "Can my EV go from Sfax to Tunis?",
            "energy-react-loop",
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual([item.name for item in tool_messages], [
            "energy_consumption_kwh"
        ])
        self.assertEqual(bound_llm.invoke.call_count, 2)
        calculation_input = fake_energy_graph.invoke.call_args.args[0]
        self.assertEqual(calculation_input["origin"], "Sfax")
        self.assertEqual(calculation_input["destination"], "Tunis")

    @patch("agent.tools.build_energy_estimation_graph")
    def test_energy_request_runs_subgraph_again_in_checkpointed_thread(
        self,
        mock_build_energy_graph: Mock,
    ) -> None:
        def calculation(available: float) -> dict:
            return {
                "origin": "Sfax",
                "destination": "Tunis",
                "reserve_soc_percent": 15.0,
                "route_segments": [],
                "total_distance_km": 270.0,
                "total_predicted_energy_kwh": 30.0,
                "available_energy_kwh": available,
                "reserve_energy_kwh": 6.33,
                "energy_available_after_reserve_kwh": available - 6.33,
                "energy_after_trip_kwh": available - 30.0,
                "projected_arrival_soc_percent": 0.0,
                "feasible": available - 6.33 >= 30.0,
                "warnings": [],
                "errors": [],
            }

        fake_energy_graph = Mock()
        fake_energy_graph.invoke.side_effect = [
            calculation(35.0),
            calculation(25.0),
        ]
        mock_build_energy_graph.return_value = fake_energy_graph
        graph, _, bound_llm = self.build_graph(
            [
                tool_call("energy_consumption_kwh", {}, "energy-1"),
                AIMessage(content="First fresh estimate."),
                tool_call("energy_consumption_kwh", {}, "energy-2"),
                AIMessage(content="Second fresh estimate."),
            ]
        )
        thread = "repeat-energy-request"

        self.invoke(
            graph,
            "Estimate energy from Sfax to Tunis.",
            thread,
        )
        result = self.invoke(
            graph,
            "Recalculate whether my battery is still sufficient.",
            thread,
        )

        self.assertEqual(fake_energy_graph.invoke.call_count, 2)
        energy_results = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
            and message.name == "energy_consumption_kwh"
        ]
        self.assertEqual(len(energy_results), 2)
        self.assertEqual(
            [message.tool_call_id for message in energy_results],
            ["energy-1", "energy-2"],
        )
        second_turn_llm_input = bound_llm.invoke.call_args.args[0]
        self.assertIn(energy_results[1], second_turn_llm_input)
        self.assertNotIn(energy_results[0], second_turn_llm_input)
        self.assertEqual(len(result["messages"]), 8)


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
