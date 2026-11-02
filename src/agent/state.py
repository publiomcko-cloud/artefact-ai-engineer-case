"""Minimal conversation state for the single customer-service agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Messages are appended with LangGraph's standard message reducer."""

    messages: Annotated[list[AnyMessage], add_messages]
