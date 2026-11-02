"""Pure helpers for safe Streamlit transcript and session-state handling."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


REQUIRED_ENVIRONMENT_VARIABLES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
)
AGENT_MESSAGES_KEY = "agent_messages"
CHAT_INPUT_GENERATION_KEY = "chat_input_generation"
SAFE_ASSISTANT_FALLBACK = (
    "I could not retrieve that information right now. Please try again."
)


@dataclass(frozen=True, slots=True)
class VisibleChatMessage:
    """A customer-visible message, separate from internal graph messages."""

    role: str
    content: str


def missing_configuration(
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return names of required variables that are absent or blank, never their values."""
    values = os.environ if environment is None else environment
    return tuple(
        name
        for name in REQUIRED_ENVIRONMENT_VARIABLES
        if not values.get(name, "").strip()
    )


def configuration_error_message(missing: Sequence[str]) -> str:
    """Build a configuration hint from known variable names only."""
    names = [name for name in REQUIRED_ENVIRONMENT_VARIABLES if name in missing]
    if not names:
        names = list(REQUIRED_ENVIRONMENT_VARIABLES)
    return (
        "Configuration is incomplete. Set "
        f"{', '.join(names)} in your .env file, then restart the app."
    )


def append_user_message(
    messages: Sequence[BaseMessage],
    content: str,
) -> list[BaseMessage]:
    """Create the next graph input while retaining the complete prior history."""
    return [*messages, HumanMessage(content=content)]


def replace_agent_history(
    session_state: MutableMapping[str, Any],
    messages: Sequence[BaseMessage],
) -> None:
    """Save the graph's complete returned message list in this user's session."""
    session_state[AGENT_MESSAGES_KEY] = list(messages)


def reset_conversation(session_state: MutableMapping[str, Any]) -> None:
    """Clear conversational state and rotate the chat-input widget key."""
    session_state[AGENT_MESSAGES_KEY] = []
    generation = session_state.get(CHAT_INPUT_GENERATION_KEY, 0)
    try:
        next_generation = int(generation) + 1
    except (TypeError, ValueError):
        next_generation = 1
    session_state[CHAT_INPUT_GENERATION_KEY] = next_generation


def _text_content(content: object) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if not isinstance(content, list):
        return None
    parts = [
        block["text"].strip()
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"text", "output_text"}
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    text = "\n".join(parts).strip()
    return text or None


def _has_tool_call(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None):
        return True
    metadata = getattr(message, "additional_kwargs", {})
    return isinstance(metadata, dict) and bool(
        metadata.get("tool_calls") or metadata.get("function_call")
    )


def latest_final_assistant_text(messages: Sequence[object]) -> str | None:
    """Find the newest user-facing AI text, ignoring tool-call and empty messages."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or _has_tool_call(message):
            continue
        text = _text_content(message.content)
        if text is not None:
            return text
    return None


def extract_final_assistant_text(messages: Sequence[object]) -> str:
    """Return the latest final AI text or a safe customer-facing fallback."""
    return latest_final_assistant_text(messages) or SAFE_ASSISTANT_FALLBACK


def visible_transcript(messages: Sequence[object]) -> list[VisibleChatMessage]:
    """Project graph history to user and final assistant text only."""
    transcript: list[VisibleChatMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = _text_content(message.content)
            if text is not None:
                transcript.append(VisibleChatMessage(role="user", content=text))
        elif isinstance(message, AIMessage) and not _has_tool_call(message):
            text = _text_content(message.content)
            if text is not None:
                transcript.append(VisibleChatMessage(role="assistant", content=text))
    return transcript
