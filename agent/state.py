"""Define state shared by the LangGraph nodes."""

from langgraph.graph import MessagesState


class CopilotState(MessagesState):
    """Conversation messages accumulated during one copilot workflow."""