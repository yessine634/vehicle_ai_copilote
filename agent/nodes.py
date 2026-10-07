"""Node functions for the copilot graph."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYSTEM_PROMPT
from agent.state import CopilotState
from tools.vehicle import VehicleStateProvider, get_vehicle_state


def create_refresh_vehicle_node(simulator: VehicleStateProvider):
    """Refresh vehicle telemetry before processing each question."""

    def refresh_vehicle(state: CopilotState) -> dict:
        stored_current = state.get("current_vehicle_state")
        current_state = get_vehicle_state(simulator)
        old_simulation_time = state.get("vehicle_simulation_time")
        new_simulation_time = current_state["motion"][
            "elapsed_time_seconds"
        ]

        # Preserve the last distinct simulation snapshot. Asking another
        # question without advancing SUMO must not erase the comparison
        # baseline.
        if stored_current is None:
            previous_state = None
        elif new_simulation_time != old_simulation_time:
            previous_state = stored_current
        else:
            previous_state = state.get("previous_vehicle_state")

        trip = current_state.get("trip", {})
        origin = trip.get("origin") if isinstance(trip, dict) else None
        motion = current_state["motion"]

        current_location = {
            "name": origin.get("name") if isinstance(origin, dict) else None,
            "routing_name": (
                origin.get("city") or origin.get("name")
                if isinstance(origin, dict)
                else None
            ),
            "latitude": motion.get("latitude"),
            "longitude": motion.get("longitude"),
            "current_road": motion.get("current_road"),
        }

        planned_origin = state.get("planned_origin")
        if planned_origin is None and current_location["routing_name"]:
            planned_origin = current_location["routing_name"]

        return {
            "previous_vehicle_state": previous_state,
            "current_vehicle_state": current_state,
            "current_location": current_location,
            "vehicle_simulation_time": new_simulation_time,
            "planned_origin": planned_origin,
            "selected_tools": [],
            "called_tools": [],
            # Replace messages from the previous turn.
            "turn_messages": [
                HumanMessage(content=state["question"])
            ],
        }

    return refresh_vehicle


def _comparison_requested(question: str) -> bool:
    """Return whether the driver explicitly requests a comparison."""

    comparison_words = (
        "changed",
        "change",
        "compare",
        "before",
        "previous",
        "earlier",
        "since",
    )
    question = question.lower()
    return any(word in question for word in comparison_words)


def _route_requested(question: str) -> bool:
    """Return whether the current question requires fresh route data."""

    route_words = (
        "route",
        "distance",
        "how far",
        "drive to",
        "driving to",
        "reach",
        "travel time",
        "how long",
        "destination",
        "highway",
        "toll",
        "ferry",
        "trip to",
    )
    question = question.casefold()
    return any(word in question for word in route_words)


def prepare_selected_tool_calls(state: CopilotState) -> dict:
    """Record exactly the tool calls selected by the LLM planner."""

    turn_messages = state.get("turn_messages", [])
    if not turn_messages or not isinstance(turn_messages[-1], AIMessage):
        raise RuntimeError("The tool planner did not return an AI message.")

    selected_tools: list[str] = []
    for call in turn_messages[-1].tool_calls:
        tool_name = call.get("name")
        if isinstance(tool_name, str) and tool_name not in selected_tools:
            selected_tools.append(tool_name)

    return {
        "selected_tools": selected_tools,
        "called_tools": [],
    }


def validate_selected_tool_calls(state: CopilotState) -> dict:
    """Prevent an answer when an LLM-selected tool did not run."""

    called_tools = [
        message.name
        for message in state.get("turn_messages", [])
        if getattr(message, "type", None) == "tool" and message.name
    ]
    selected_tools = state.get("selected_tools", [])
    missing = set(selected_tools) - set(called_tools)
    if missing:
        raise RuntimeError(
            "LLM-selected tools were not executed during the current turn: "
            f"{sorted(missing)}"
        )
    return {"called_tools": called_tools}


def create_assistant_node(llm_with_tools):
    """Create the node that interprets fresh structured state."""

    def assistant_node(state: CopilotState) -> dict:
        question = state.get("question", "")

        vehicle_context = {
            "current_vehicle_state": state.get(
                "current_vehicle_state"
            ),
            "current_location": state.get("current_location"),
            "vehicle_simulation_time": state.get(
                "vehicle_simulation_time"
            ),
        }

        # Remembered route facts are arguments only. Hide them from unrelated
        # questions so the model cannot turn them into invented measurements.
        if _route_requested(question):
            vehicle_context["planned_origin"] = state.get("planned_origin")
            vehicle_context["planned_destination"] = state.get(
                "planned_destination"
            )

        # Only expose the older snapshot for comparison questions.
        if _comparison_requested(question):
            vehicle_context["previous_vehicle_state"] = state.get(
                "previous_vehicle_state"
            )

        context_text = json.dumps(
            vehicle_context,
            ensure_ascii=False,
            default=str,
        )

        system_message = SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                "The following vehicle information was refreshed for "
                "this turn. Treat current_vehicle_state as the only "
                "source for current vehicle, battery, health, and "
                "traffic data.\n\n"
                f"{context_text}"
            )
        )

        response = llm_with_tools.invoke(
            [
                system_message,
                *state.get("turn_messages", []),
            ]
        )

        updated_messages = [
            *state.get("turn_messages", []),
            response,
        ]

        result = {
            "turn_messages": updated_messages,
        }

        if not response.tool_calls:
            result["answer"] = str(response.content)

        return result

    return assistant_node


def create_final_answer_node(llm):
    """Create the node that writes the final answer without tool calls."""

    def final_answer_node(state: CopilotState) -> dict:
        question = state.get("question", "")

        vehicle_context = {
            "current_vehicle_state": state.get(
                "current_vehicle_state"
            ),
            "current_location": state.get("current_location"),
            "vehicle_simulation_time": state.get(
                "vehicle_simulation_time"
            ),
            "selected_tools": state.get("selected_tools", []),
            "called_tools": state.get("called_tools", []),
        }

        if _route_requested(question):
            vehicle_context["planned_origin"] = state.get("planned_origin")
            vehicle_context["planned_destination"] = state.get(
                "planned_destination"
            )

        if _comparison_requested(question):
            vehicle_context["previous_vehicle_state"] = state.get(
                "previous_vehicle_state"
            )

        context_text = json.dumps(
            vehicle_context,
            ensure_ascii=False,
            default=str,
        )

        system_message = SystemMessage(
            content=(
                "You are writing the final answer for an intelligent "
                "electric-vehicle copilot.\n\n"
                "Do not call tools in this step. Do not request weather_tool, "
                "route_tool, or any other tool. Use only the current vehicle "
                "state and the tool results already present in this turn's "
                "messages. If a needed weather or route result is missing, "
                "say that fresh information is unavailable instead of "
                "inventing it.\n\n"
                "Answer the driver clearly and concisely. Clearly identify "
                "estimated or simulated information. Never state "
                "charging-station availability because no charging-station "
                "tool is available.\n\n"
                "Current refreshed context:\n"
                f"{context_text}"
            )
        )

        response = llm.invoke(
            [
                system_message,
                *state.get("turn_messages", []),
            ]
        )

        if response.tool_calls:
            raise RuntimeError(
                "The final-answer LLM tried to call a tool. Tool calls are "
                "only allowed in the planner step."
            )

        return {
            "turn_messages": [
                *state.get("turn_messages", []),
                response,
            ],
            "answer": str(response.content),
        }

    return final_answer_node
