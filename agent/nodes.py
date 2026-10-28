"""Node functions for the LangGraph ReAct copilot."""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.prompts import SYSTEM_PROMPT
from agent.state import CopilotState


LLM_MESSAGE_LIMIT = 10


def _tool_call_ids(message: AIMessage) -> set[str]:
    """Return the non-empty IDs requested by one AI tool-call message."""

    return {
        call_id
        for call in message.tool_calls
        if isinstance((call_id := call.get("id")), str) and call_id
    }


def filter_messages_for_llm(
    messages: Sequence[AnyMessage],
    limit: int = LLM_MESSAGE_LIMIT,
) -> list[AnyMessage]:
    """Return recent messages without separating tool calls from results.

    State is never modified. AI tool-call messages and their matching tool
    results are treated as an atomic unit. Invalid orphan tool messages are
    omitted so the model always receives a valid tool protocol.
    """

    if limit < 1:
        raise ValueError("The LLM message limit must be at least 1.")

    message_list = list(messages)
    units: list[list[AnyMessage]] = []
    index = 0

    while index < len(message_list):
        message = message_list[index]

        if isinstance(message, AIMessage) and message.tool_calls:
            expected_ids = _tool_call_ids(message)
            unit: list[AnyMessage] = [message]
            result_ids: set[str] = set()
            cursor = index + 1

            while cursor < len(message_list) and isinstance(
                message_list[cursor], ToolMessage
            ):
                tool_message = message_list[cursor]
                if tool_message.tool_call_id in expected_ids:
                    unit.append(tool_message)
                    result_ids.add(tool_message.tool_call_id)
                cursor += 1

            if expected_ids and expected_ids.issubset(result_ids):
                units.append(unit)
            index = cursor
            continue

        if not isinstance(message, ToolMessage):
            units.append([message])
        index += 1

    selected_units: list[list[AnyMessage]] = []
    selected_count = 0
    for unit in reversed(units):
        if selected_units and selected_count + len(unit) > limit:
            break
        if not selected_units or selected_count + len(unit) <= limit:
            selected_units.append(unit)
            selected_count += len(unit)

    return [
        message
        for unit in reversed(selected_units)
        for message in unit
    ]


def _clean_location(value: str) -> str | None:
    """Normalize a location captured from an explicit driver statement."""

    cleaned = value.strip(" \t\n\r,.;:?!\"'")
    cleaned = re.sub(
        r"\s+(?:by car|right now|today|tomorrow|please)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or None


def extract_explicit_route_context(text: str) -> dict[str, str]:
    """Extract only clearly stated origin/destination values from text."""

    update: dict[str, str] = {}

    from_to = re.search(
        r"\bfrom\s+(.+?)\s+to\s+"
        r"(.+?)(?=\s+(?:and|but)\b|[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if from_to:
        origin = _clean_location(from_to.group(1))
        destination = _clean_location(from_to.group(2))
        if origin:
            update["origin"] = origin
        if destination:
            update["destination"] = destination

    origin_match = re.search(
        r"\b(?:origin|starting point)\s*(?:is|=|:|to)?\s+"
        r"(.+?)(?=\s+(?:and|but)\b|[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if origin_match:
        origin = _clean_location(origin_match.group(1))
        if origin:
            update["origin"] = origin

    current_location = re.search(
        r"\b(?:i am|i'm|currently)\s+(?:located\s+)?in\s+"
        r"(.+?)(?=\s+(?:and|but)\b|[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if current_location:
        origin = _clean_location(current_location.group(1))
        if origin:
            update["origin"] = origin

    destination_match = re.search(
        r"\b(?:destination|destination city)\s*"
        r"(?:is|=|:|to)?\s+(.+?)(?=\s+(?:and|but)\b|[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if destination_match:
        destination = _clean_location(destination_match.group(1))
        if destination:
            update["destination"] = destination

    return update


def _latest_human_text(messages: Sequence[AnyMessage]) -> str:
    """Return the latest driver text, or an empty string when unavailable."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _explicit_tool_route_context(
    response: AIMessage,
    driver_text: str,
) -> dict[str, str]:
    """Use route arguments only when their place names occur in driver text."""

    update: dict[str, str] = {}
    folded_driver_text = driver_text.casefold()

    for tool_call in response.tool_calls:
        if tool_call.get("name") != "route_tool":
            continue
        arguments = tool_call.get("args", {})
        if not isinstance(arguments, dict):
            continue

        for field in ("origin", "destination"):
            value = arguments.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = value.strip()
            short_name = normalized.split(",", maxsplit=1)[0].strip()
            if short_name and short_name.casefold() in folded_driver_text:
                update[field] = normalized

    return update


def _system_prompt(state: CopilotState) -> str:
    """Include stable route context without treating it as live tool data."""

    origin = state.get("origin") or "not provided"
    destination = state.get("destination") or "not provided"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Persistent route context for this conversation:\n"
        f"Origin: {origin}\n"
        f"Destination: {destination}\n\n"
        "These values are conversational context only. Retrieve fresh route, "
        "weather, vehicle, battery, health, and traffic information through "
        "the appropriate tools whenever the current question requires it."
    )


def create_assistant_node(llm_with_tools):
    """Create the ReAct assistant with a filtered, read-only LLM context."""

    def assistant(state: CopilotState) -> dict:
        stored_messages = list(state.get("messages", []))
        driver_text = _latest_human_text(stored_messages)
        context_update = extract_explicit_route_context(driver_text)
        prompt_state = {**state, **context_update}

        response = llm_with_tools.invoke(
            [
                SystemMessage(content=_system_prompt(prompt_state)),
                *filter_messages_for_llm(stored_messages),
            ]
        )

        context_update.update(
            _explicit_tool_route_context(response, driver_text)
        )
        return {"messages": [response], **context_update}

    return assistant
