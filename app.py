"""Streamlit entry point for the Empório da Música customer-service assistant."""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage

from src.agent.graph import build_agent
from src.data_loader import StoreData
from src.ui_helpers import (
    AGENT_MESSAGES_KEY,
    CHAT_INPUT_GENERATION_KEY,
    SAFE_ASSISTANT_FALLBACK,
    append_user_message,
    configuration_error_message,
    latest_final_assistant_text,
    missing_configuration,
    replace_agent_history,
    reset_conversation,
    visible_transcript,
)


LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Empório da Música — AI Customer Service",
    page_icon="🎵",
    layout="centered",
)
load_dotenv()


@st.cache_resource(show_spinner=False)
def load_store_data() -> StoreData:
    """Share the immutable source tables across sessions and Streamlit reruns."""
    return StoreData.load()


@st.cache_resource(show_spinner=False, scope="session")
def load_agent_graph():
    """Reuse one stateless compiled graph and model client in this user session."""
    return build_agent(data=load_store_data())


def _initialize_session() -> None:
    if AGENT_MESSAGES_KEY not in st.session_state:
        st.session_state[AGENT_MESSAGES_KEY] = []
    if CHAT_INPUT_GENERATION_KEY not in st.session_state:
        st.session_state[CHAT_INPUT_GENERATION_KEY] = 0


def _reset_session_conversation() -> None:
    reset_conversation(st.session_state)


def _render_sidebar(model_name: str) -> None:
    with st.sidebar:
        st.subheader("Conversation")
        st.button(
            "New conversation",
            on_click=_reset_session_conversation,
            use_container_width=True,
        )
        st.caption(f"Chat model: {model_name}")
        st.caption("Operational answers are grounded in store data and policies.")


def _invoke_agent(agent: Any, graph_input: list[BaseMessage]) -> list[BaseMessage]:
    result = agent.invoke({"messages": graph_input})
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, (list, tuple)):
        raise ValueError("The agent returned an invalid message history.")
    if not all(isinstance(message, BaseMessage) for message in messages):
        raise ValueError("The agent returned an invalid message history.")
    return list(messages)


def _process_user_message(agent: Any, user_text: str) -> str:
    history = st.session_state[AGENT_MESSAGES_KEY]
    graph_input = append_user_message(history, user_text)
    try:
        with st.spinner("Thinking..."):
            returned_messages = _invoke_agent(agent, graph_input)
        answer = latest_final_assistant_text(returned_messages)
        if answer is None:
            answer = SAFE_ASSISTANT_FALLBACK
            returned_messages.append(AIMessage(content=answer))
        replace_agent_history(st.session_state, returned_messages)
        return answer
    except Exception:
        LOGGER.exception("The assistant invocation failed.")
        fallback_history = [*graph_input, AIMessage(content=SAFE_ASSISTANT_FALLBACK)]
        replace_agent_history(st.session_state, fallback_history)
        return SAFE_ASSISTANT_FALLBACK


def main() -> None:
    st.title("Empório da Música — AI Customer Service")
    st.caption(
        "Ask about products, prices, availability, promotions, orders, and store policies."
    )

    missing = missing_configuration()
    if missing:
        st.error(configuration_error_message(missing))
        st.stop()

    _initialize_session()
    model_name = os.getenv("OPENAI_MODEL", "").strip()
    _render_sidebar(model_name)

    try:
        agent = load_agent_graph()
    except Exception:
        LOGGER.exception("The assistant could not be initialized.")
        st.error("The assistant could not be initialized. Check its configuration and data files.")
        st.stop()

    for message in visible_transcript(st.session_state[AGENT_MESSAGES_KEY]):
        with st.chat_message(message.role):
            st.markdown(message.content)

    widget_generation = st.session_state[CHAT_INPUT_GENERATION_KEY]
    user_text = st.chat_input(
        "Type your message",
        key=f"chat_input_{widget_generation}",
    )
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            answer = _process_user_message(agent, user_text)
            st.markdown(answer)


if __name__ == "__main__":
    main()
