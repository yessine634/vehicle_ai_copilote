"""Try Groq tool calling with the project's weather and routing functions."""

import os
from pathlib import Path
import sys
import unittest

# Make project packages importable when this file runs from tests/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from tools.routing import get_route
from tools.weather import get_weather

load_dotenv(PROJECT_ROOT / ".env")


@tool
def weather_tool(city: str) -> dict:
    """Get the current weather for a city."""
    return get_weather(city)


@tool
def route_tool(origin: str, destination: str) -> dict:
    """Get driving route information between an origin and destination."""
    return get_route(origin, destination)


AVAILABLE_TOOLS = {
    weather_tool.name: weather_tool,
    route_tool.name: route_tool,
}

LIVE_ROUTE_QUESTION = (
    "I'm currently in Tunis and I want to drive to Houmt Souk in Djerba. "
    "How far is the trip, approximately how long will it take, and does the "
    "route include highways, toll roads, or a ferry?"
)


def ask_with_tools(
    llm_with_tools,
    question: str,
    *,
    executed_tools: list[str] | None = None,
):
    """Execute requested local tools and return the LLM's final response."""

    messages = [HumanMessage(content=question)]
    first_response = llm_with_tools.invoke(messages)
    messages.append(first_response)

    if not first_response.tool_calls:
        return first_response

    for tool_call in first_response.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = AVAILABLE_TOOLS.get(tool_name)
        if selected_tool is None:
            raise ValueError(f"The LLM requested an unsupported tool: {tool_name}")

        tool_message = selected_tool.invoke(tool_call)
        messages.append(tool_message)
        if executed_tools is not None:
            executed_tools.append(tool_name)

        print(f"Executed tool: {tool_name}")
        print(f"Tool arguments: {tool_call['args']}")
        print(f"Tool result: {tool_message.content}\n")

    return llm_with_tools.invoke(messages)


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TOOL_CALLING_TEST") == "1",
    "Set RUN_LIVE_TOOL_CALLING_TEST=1 to call Groq and Valhalla.",
)
class LiveToolCallingTests(unittest.TestCase):
    """Exercise the real LLM-to-routing-tool conversation."""

    def test_tunis_to_houmt_souk_route_question(self) -> None:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
        )
        llm_with_tools = llm.bind_tools(list(AVAILABLE_TOOLS.values()))
        executed_tools: list[str] = []

        final_response = ask_with_tools(
            llm_with_tools,
            LIVE_ROUTE_QUESTION,
            executed_tools=executed_tools,
        )

        self.assertIn("route_tool", executed_tools)
        self.assertIsInstance(final_response.content, str)
        self.assertTrue(final_response.content.strip())
        print("Final answer:")
        print(final_response.content)


def main() -> None:
    """Run the complete Groq routing-tool conversation."""

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(list(AVAILABLE_TOOLS.values()))
    final_response = ask_with_tools(
        llm_with_tools,
        LIVE_ROUTE_QUESTION,
    )

    print("Final answer:")
    print(final_response.content)


if __name__ == "__main__":
    main()
