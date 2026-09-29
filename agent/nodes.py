"""Node functions for the copilot graph."""

from langchain_core.messages import SystemMessage

from agent.prompts import SYSTEM_PROMPT
from agent.state import CopilotState


def create_assistant_node(llm_with_tools):
    """Create the node that lets Groq answer or request tools."""

    def assistant_node(state: CopilotState) -> dict:
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    return assistant_node