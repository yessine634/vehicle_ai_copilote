"""Construct the LangGraph vehicle-copilot workflow."""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import (
    create_assistant_node,
    create_final_answer_node,
    create_refresh_vehicle_node,
    prepare_selected_tool_calls,
    validate_selected_tool_calls,
)
from agent.state import CopilotState
from agent.tools import build_tools
from tools.vehicle import VehicleStateProvider


def build_copilot_graph(simulator: VehicleStateProvider):
    """Build a stateful copilot connected to an active simulator."""

    load_dotenv()

    tools = build_tools(simulator)

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    # Mixed questions can require route and weather in the same planner turn.
    # Groq accepts the OpenAI-compatible parallel_tool_calls option.
    llm_with_tools = llm.bind_tools(
        tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    builder = StateGraph(CopilotState)

    builder.add_node(
        "refresh_vehicle",
        create_refresh_vehicle_node(simulator),
    )

    builder.add_node(
        "planner",
        create_assistant_node(llm_with_tools),
    )
    builder.add_node("record_selected_tools", prepare_selected_tool_calls)
    builder.add_node("validate_selected_tools", validate_selected_tool_calls)
    builder.add_node(
        "final_answer",
        create_final_answer_node(llm),
    )

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,
        messages_key="turn_messages",
    )

    def execute_tools(state: CopilotState) -> dict:
        """Execute requested tools and retain this turn's messages."""

        planned_origin = state.get("planned_origin")
        planned_destination = state.get("planned_destination")
        turn_messages = list(state.get("turn_messages", []))

        # A destination-only question must use the known simulated location,
        # never an origin guessed by the model. Explicit origins written by
        # the driver are preserved.
        if turn_messages:
            last_message = turn_messages[-1]
            normalized_calls = []
            question = state.get("question", "").casefold()
            current_location = state.get("current_location") or {}
            current_origin = current_location.get("routing_name")
            for tool_call in getattr(last_message, "tool_calls", []):
                normalized_call = dict(tool_call)
                arguments = dict(normalized_call.get("args", {}))
                if normalized_call.get("name") == "route_tool":
                    requested_origin = arguments.get("origin")
                    requested_origin_name = (
                        requested_origin.split(",", maxsplit=1)[0].strip()
                        if isinstance(requested_origin, str)
                        else ""
                    )
                    origin_was_explicit = (
                        bool(requested_origin_name)
                        and requested_origin_name.casefold() in question
                    )
                    if current_origin and not origin_was_explicit:
                        arguments["origin"] = current_origin
                normalized_call["args"] = arguments
                normalized_calls.append(normalized_call)

            if normalized_calls:
                turn_messages[-1] = last_message.model_copy(
                    update={"tool_calls": normalized_calls}
                )

        tool_state = {**state, "turn_messages": turn_messages}
        result = tool_node.invoke(tool_state)

        if turn_messages:
            last_message = turn_messages[-1]
            for tool_call in getattr(last_message, "tool_calls", []):
                if tool_call.get("name") != "route_tool":
                    continue
                arguments = tool_call.get("args", {})
                origin = arguments.get("origin")
                destination = arguments.get("destination")
                if isinstance(origin, str) and origin.strip():
                    planned_origin = origin.strip()
                if isinstance(destination, str) and destination.strip():
                    planned_destination = destination.strip()

        return {
            "turn_messages": [
                *turn_messages,
                *result.get("turn_messages", []),
            ],
            "planned_origin": planned_origin,
            "planned_destination": planned_destination,
        }

    builder.add_node("tools", execute_tools)

    builder.add_edge(START, "refresh_vehicle")
    builder.add_edge("refresh_vehicle", "planner")
    builder.add_edge("planner", "record_selected_tools")

    builder.add_conditional_edges(
        "record_selected_tools",
        lambda state: tools_condition(
            state,
            messages_key="turn_messages",
        ),
        {
            "tools": "tools",
            "__end__": "final_answer",
        },
    )

    builder.add_edge("tools", "validate_selected_tools")
    builder.add_edge("validate_selected_tools", "final_answer")
    builder.add_edge("final_answer", END)

    return builder.compile(checkpointer=InMemorySaver())
