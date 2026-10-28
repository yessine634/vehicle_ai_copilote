"""Define the persistent state shared by the LangGraph copilot nodes."""

from langgraph.graph import MessagesState


class CopilotState(MessagesState, total=False):
    """Conversation history plus stable route context for one thread."""

    origin: str | None
    destination: str | None


__all__ = ["CopilotState"]
