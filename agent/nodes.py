"""Node functions for the LangGraph ReAct copilot."""

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent.prompts import MEMORY_SUMMARY_PROMPT, SYSTEM_PROMPT
from agent.state import MessagesState


MEMORY_MESSAGE_LIMIT = 5
MEMORY_SUMMARY_PREFIX = "Conversation memory summary:"


def _message_text(content) -> str:
    """Return readable text from common LangChain content formats."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    return str(content)


def create_memory_node(llm):
    """Create a node that summarizes completed history before a new turn."""

    def manage_memory(state: MessagesState) -> dict:
        messages = list(state.get("messages", []))
        if len(messages) <= MEMORY_MESSAGE_LIMIT:
            return {}

        human_indices = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        if not human_indices:
            return {}

        latest_human_index = human_indices[-1]
        previous_messages = messages[:latest_human_index]
        current_messages = messages[latest_human_index:]

        # This node runs only from START. Do not summarize an active tool-call
        # protocol if the node is invoked from another context later.
        if len(current_messages) != 1:
            return {}

        stable_context = []
        for message in previous_messages:
            if isinstance(message, HumanMessage):
                stable_context.append(
                    f"Driver: {_message_text(message.content)}"
                )
            elif (
                isinstance(message, SystemMessage)
                and _message_text(message.content).startswith(
                    MEMORY_SUMMARY_PREFIX
                )
            ):
                stable_context.append(_message_text(message.content))

        if stable_context:
            summary_response = llm.invoke(
                [
                    SystemMessage(content=MEMORY_SUMMARY_PROMPT),
                    HumanMessage(content="\n\n".join(stable_context)),
                ]
            )
            summary_text = _message_text(summary_response.content).strip()
        else:
            summary_text = "No stable earlier driver context was retained."

        summary_message = SystemMessage(
            content=f"{MEMORY_SUMMARY_PREFIX}\n{summary_text}"
        )

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_message,
                current_messages[0],
            ]
        }

    return manage_memory


def create_assistant_node(llm_with_tools):
    """Create the single assistant used for tool calls and final answers."""

    def assistant(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    return assistant
