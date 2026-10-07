"""Define state shared by the LangGraph nodes."""

from typing import Any, TypedDict

from langchain_core.messages import AnyMessage


class CopilotState(TypedDict, total=False):
    # Input and output for the current turn
    question: str
    turn_messages: list[AnyMessage]
    answer: str
    selected_tools: list[str]
    called_tools: list[str]

    # Persistent structured vehicle memory
    previous_vehicle_state: dict[str, Any] | None
    current_vehicle_state: dict[str, Any] | None
    current_location: dict[str, Any] | None
    vehicle_simulation_time: float | None

    # Optional stable conversational facts
    planned_origin: str | None
    planned_destination: str | None
