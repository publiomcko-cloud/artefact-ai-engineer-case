"""Minimal LangGraph single-agent customer-service orchestration."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.prompt import SYSTEM_PROMPT
from src.agent.state import AgentState
from src.agent.tools import PolicySearcher, build_agent_tools
from src.data_loader import StoreData


class ModelConfigurationError(ValueError):
    """Raised when chat model configuration is missing or invalid."""


GENERIC_MODEL_FAILURE = "I could not retrieve that information right now. Please try again."


def _create_chat_model() -> Any:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not model_name:
        raise ModelConfigurationError(
            "Set OPENAI_API_KEY and OPENAI_MODEL before starting the chat agent."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            retries=1,
            request_timeout=45,
        )
    except ImportError:
        raise ModelConfigurationError(
            "The langchain-google-genai package is required for the chat agent."
        ) from None
    except Exception:
        raise ModelConfigurationError(
            "The configured Gemini chat model could not be initialized. Check OPENAI_MODEL and OPENAI_API_KEY."
        ) from None


def _safe_tool_error(_: Exception) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {
                "code": "tool_unavailable",
                "message": "The requested information is unavailable right now.",
            },
        },
        separators=(",", ":"),
    )


def build_agent(
    model: Any | None = None,
    *,
    data: StoreData | None = None,
    policy_searcher: PolicySearcher | None = None,
):
    """Build the message/tool loop; inject a model to run without credentials."""
    chat_model = model if model is not None else _create_chat_model()
    tools = build_agent_tools(data=data, policy_searcher=policy_searcher)
    try:
        model_with_tools = chat_model.bind_tools(tools)
    except Exception:
        raise ModelConfigurationError(
            "The configured chat model must support tool calling."
        ) from None

    def call_agent(state: AgentState) -> dict[str, list[AIMessage]]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        try:
            response = model_with_tools.invoke(messages)
        except Exception:
            return {"messages": [AIMessage(content=GENERIC_MODEL_FAILURE)]}
        if not isinstance(response, AIMessage):
            return {"messages": [AIMessage(content=GENERIC_MODEL_FAILURE)]}
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", ToolNode(tools, handle_tool_errors=_safe_tool_error))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
