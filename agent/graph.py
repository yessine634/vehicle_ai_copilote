"""Construct the LangGraph vehicle-copilot workflow."""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import create_assistant_node
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
    llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(CopilotState)

    builder.add_node(
        "assistant",
        create_assistant_node(llm_with_tools),
    )
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=True),
    )

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=InMemorySaver())