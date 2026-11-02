from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.ui_helpers import (
    AGENT_MESSAGES_KEY,
    CHAT_INPUT_GENERATION_KEY,
    SAFE_ASSISTANT_FALLBACK,
    append_user_message,
    configuration_error_message,
    extract_final_assistant_text,
    latest_final_assistant_text,
    missing_configuration,
    replace_agent_history,
    reset_conversation,
    visible_transcript,
)


def _tool_call_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup_order",
                "args": {"order_id": 15, "email": "private@example.com"},
                "id": "private-tool-call-id",
                "type": "tool_call",
            }
        ],
    )


def test_final_assistant_extraction_returns_latest_valid_ai_text() -> None:
    messages = [
        AIMessage(content="Earlier response."),
        _tool_call_message(),
        ToolMessage(content='{"internal":"tool output"}', tool_call_id="tool-1"),
        AIMessage(content="The final response."),
        AIMessage(content="  "),
    ]

    assert latest_final_assistant_text(messages) == "The final response."
    assert extract_final_assistant_text(messages) == "The final response."


def test_final_assistant_extraction_ignores_empty_and_tool_call_only_messages() -> None:
    messages = [AIMessage(content=""), _tool_call_message()]

    assert latest_final_assistant_text(messages) is None
    assert extract_final_assistant_text(messages) == SAFE_ASSISTANT_FALLBACK


def test_final_assistant_extraction_supports_text_content_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "The product is available."},
            {"type": "reasoning", "reasoning": "must not be rendered"},
        ]
    )

    assert extract_final_assistant_text([message]) == "The product is available."


def test_visible_transcript_includes_only_user_and_final_assistant_text() -> None:
    messages = [
        SystemMessage(content="hidden system prompt"),
        HumanMessage(content="What is my order status?"),
        _tool_call_message(),
        ToolMessage(
            content='{"status":"pending","customer_id":18}',
            tool_call_id="private-tool-call-id",
        ),
        AIMessage(content="Please verify your email first."),
    ]

    assert [(item.role, item.content) for item in visible_transcript(messages)] == [
        ("user", "What is my order status?"),
        ("assistant", "Please verify your email first."),
    ]


def test_conversation_helpers_retain_graph_history_and_append_user_turn() -> None:
    prior = [
        HumanMessage(content="How much is the Yamaha C40?"),
        AIMessage(content="It costs R$ 599.90."),
    ]
    turn_messages = append_user_message(prior, "And with PIX?")
    returned = [
        *turn_messages,
        _tool_call_message(),
        ToolMessage(content='{"effective_price_brl":"569.91"}', tool_call_id="offer"),
        AIMessage(content="With PIX, it is R$ 569.91."),
    ]
    session_state: dict[str, object] = {"application_resource": object()}

    replace_agent_history(session_state, returned)
    next_turn = append_user_message(session_state[AGENT_MESSAGES_KEY], "Is it in stock?")

    assert turn_messages[:-1] == prior
    assert isinstance(turn_messages[-1], HumanMessage)
    assert turn_messages[-1].content == "And with PIX?"
    assert session_state[AGENT_MESSAGES_KEY] == returned
    assert next_turn[:-1] == returned
    assert next_turn[-1].content == "Is it in stock?"


def test_reset_clears_conversation_without_clearing_application_resources() -> None:
    cached_graph = object()
    session_state: dict[str, object] = {
        AGENT_MESSAGES_KEY: [HumanMessage(content="previous turn")],
        CHAT_INPUT_GENERATION_KEY: 2,
        "application_resource": cached_graph,
    }

    reset_conversation(session_state)

    assert session_state[AGENT_MESSAGES_KEY] == []
    assert session_state[CHAT_INPUT_GENERATION_KEY] == 3
    assert session_state["application_resource"] is cached_graph


def test_configuration_validation_lists_missing_names_without_values() -> None:
    environment = {
        "OPENAI_API_KEY": "secret-value-must-not-appear",
        "OPENAI_MODEL": "",
        "OPENAI_EMBEDDING_MODEL": "embedding-model-name",
        "OPENAI_BASE_URL": "https://user:password@example.invalid/v1",
    }

    missing = missing_configuration(environment)
    message = configuration_error_message(missing)

    assert missing == ("OPENAI_MODEL",)
    assert "OPENAI_MODEL" in message
    assert "secret-value-must-not-appear" not in message
    assert "password" not in message


def test_configuration_validation_reports_all_missing_required_variables() -> None:
    missing = missing_configuration({"OPENAI_BASE_URL": "https://example.invalid/v1"})

    assert missing == (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    )
