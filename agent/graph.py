"""Construct the LangGraph ReAct vehicle-copilot workflow."""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import create_assistant_node, create_memory_node
from agent.state import MessagesState
from agent.tools import build_tools
from tools.vehicle import VehicleStateProvider


def build_copilot_graph(simulator: VehicleStateProvider):
    """Build a stateful ReAct copilot connected to an active simulator."""

    load_dotenv()

    tools = build_tools(simulator)
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(MessagesState)
    builder.add_node("manage_memory", create_memory_node(llm))
    builder.add_node("assistant", create_assistant_node(llm_with_tools))
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=True),
    )

    builder.add_edge(START, "manage_memory")
    builder.add_edge("manage_memory", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=InMemorySaver())
