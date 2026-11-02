from __future__ import annotations

import json
from collections import deque

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import (
    GENERIC_MODEL_FAILURE,
    ModelConfigurationError,
    _create_chat_model,
    build_agent,
)
from src.agent.prompt import SYSTEM_PROMPT
from src.data_loader import StoreData
from src.rag.retriever import PolicySearchResult


class ScriptedChatModel:
    """Offline model double that returns pre-scripted tool calls and answers."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = deque(responses)
        self.inputs: list[list[object]] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [item.name for item in tools]
        return self

    def invoke(self, messages):
        self.inputs.append(list(messages))
        if not self.responses:
            raise AssertionError("The scripted model has no response left.")
        return self.responses.popleft()


@pytest.fixture(scope="module")
def store_data() -> StoreData:
    return StoreData.load()


def sample_policy_search(query: str, k: int) -> list[PolicySearchResult]:
    assert query.strip()
    assert 1 <= k <= 5
    return [
        PolicySearchResult(
            chunk_id="local-policy-chunk",
            text="The store accepts PIX, debit, and credit card payments.",
            section="3. Formas de Pagamento",
            page_start=3,
            page_end=3,
            pages=(3,),
            source="data/store_policies.pdf",
            score=0.93,
        )
    ]


def _tool_call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


def _build(model: ScriptedChatModel, data: StoreData):
    return build_agent(
        model=model,
        data=data,
        policy_searcher=sample_policy_search,
    )


def _tool_messages(result) -> list[ToolMessage]:
    return [message for message in result["messages"] if isinstance(message, ToolMessage)]


def test_direct_out_of_scope_answer_ends_without_tool_calls(
    store_data: StoreData,
) -> None:
    model = ScriptedChatModel([AIMessage(content="I can help with store products and policies.")])

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="Who won the World Cup in 1998?")]}
    )

    assert result["messages"][-1].content == "I can help with store products and policies."
    assert not _tool_messages(result)
    assert model.bound_tool_names == [
        "search_products",
        "get_product",
        "get_product_offer",
        "search_policy",
        "lookup_order",
    ]
    assert isinstance(model.inputs[0][0], SystemMessage)
    assert model.inputs[0][0].content == SYSTEM_PROMPT


def test_product_price_call_executes_tool_and_returns_to_agent(
    store_data: StoreData,
) -> None:
    model = ScriptedChatModel(
        [
            _tool_call("get_product", {"product_query": "Yamaha C40"}, "product-call"),
            AIMessage(content="The Yamaha C40 is R$ 599.90 and has 12 in stock."),
        ]
    )

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="How much is the Yamaha C40?")]}
    )

    tool_messages = _tool_messages(result)
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["product"]["price_brl"] == "599.90"
    assert result["messages"][-1].content.startswith("The Yamaha C40")
    assert len(model.inputs) == 2
    assert isinstance(model.inputs[1][-1], ToolMessage)


def test_policy_tool_result_returns_to_agent_as_untrusted_evidence(
    store_data: StoreData,
) -> None:
    model = ScriptedChatModel(
        [
            _tool_call("search_policy", {"query": "payment methods", "k": 2}, "policy-call"),
            AIMessage(content="The policy lists PIX, debit, and credit card payments (section 3)."),
        ]
    )

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="What payment methods do you accept?")]}
    )

    evidence = json.loads(_tool_messages(result)[0].content)["evidence"][0]
    assert evidence["section"] == "3. Formas de Pagamento"
    assert evidence["pages"] == [3]
    assert evidence["source"] == "data/store_policies.pdf"
    model_policy_result = next(
        message for message in model.inputs[1] if isinstance(message, ToolMessage)
    )
    assert "The store accepts PIX" in model_policy_result.content
    assert "payments" in result["messages"][-1].content.casefold()


def test_order_without_identity_returns_verification_request_not_order_details(
    store_data: StoreData,
) -> None:
    model = ScriptedChatModel(
        [
            _tool_call("lookup_order", {"order_id": 15}, "order-call"),
            AIMessage(content="Please provide the email or phone associated with order 15."),
        ]
    )

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="What is the status of order 15?")]}
    )

    tool_output = _tool_messages(result)[0].content
    assert json.loads(tool_output)["error"]["code"] == "verification_required"
    assert "8348" not in tool_output
    assert "pending" not in tool_output
    assert result["messages"][-1].content.startswith("Please provide")


def test_verified_order_flow_exposes_only_allowed_tool_result(
    store_data: StoreData,
) -> None:
    customer = store_data.customers.loc[
        store_data.customers["customer_id"] == 18
    ].iloc[0]
    model = ScriptedChatModel(
        [
            _tool_call(
                "lookup_order",
                {"order_id": 15, "email": customer["email"]},
                "verified-order-call",
            ),
            AIMessage(content="Order 15 is pending. Its total is R$ 8,348.00."),
        ]
    )

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="Check order 15 for me.")]}
    )

    tool_output = _tool_messages(result)[0].content
    payload = json.loads(tool_output)
    assert payload["ok"] is True
    assert payload["order"]["status"] == "pending"
    assert payload["order"]["total_brl"] == "8348.00"
    assert payload["order"]["tracking_code"] is None
    assert "customer_id" not in tool_output
    assert "notes" not in tool_output
    assert "verification" not in tool_output.casefold()
    assert "phone" not in tool_output.casefold()
    assert "email" not in tool_output.casefold()
    assert "_verification_seal" not in tool_output


def test_follow_up_turn_receives_prior_messages_for_product_context(
    store_data: StoreData,
) -> None:
    prior_user = HumanMessage(content="How much is the Yamaha C40?")
    current_user = HumanMessage(content="And with PIX?")
    model = ScriptedChatModel(
        [
            AIMessage(content="The Yamaha C40 costs R$ 599.90."),
            _tool_call(
                "get_product_offer",
                {"product_id": 81, "payment_method": "pix"},
                "pix-offer-call",
            ),
            AIMessage(content="With PIX, the current offer is R$ 569.91."),
        ]
    )
    agent = _build(model, store_data)
    first_turn = agent.invoke({"messages": [prior_user]})

    result = agent.invoke({"messages": [*first_turn["messages"], current_user]})

    model_input = model.inputs[1]
    assert prior_user in model_input
    assert first_turn["messages"][-1] in model_input
    assert current_user in model_input
    offer_result = json.loads(_tool_messages(result)[0].content)
    assert offer_result["offer"]["effective_price_brl"] == "569.91"


def test_order_verification_can_complete_on_a_follow_up_turn(
    store_data: StoreData,
) -> None:
    customer = store_data.customers.loc[
        store_data.customers["customer_id"] == 18
    ].iloc[0]
    initial_user = HumanMessage(content="What is the status of order 15?")
    follow_up = HumanMessage(content=f"My email is {customer['email']}.")
    model = ScriptedChatModel(
        [
            _tool_call("lookup_order", {"order_id": 15}, "order-needs-verification"),
            AIMessage(content="Please provide the email or phone associated with the order."),
            _tool_call(
                "lookup_order",
                {"order_id": 15, "email": customer["email"]},
                "order-verified-on-follow-up",
            ),
            AIMessage(content="Order 15 is pending."),
        ]
    )
    agent = _build(model, store_data)

    first_turn = agent.invoke({"messages": [initial_user]})
    assert json.loads(_tool_messages(first_turn)[0].content)["error"]["code"] == (
        "verification_required"
    )
    second_turn = agent.invoke({"messages": [*first_turn["messages"], follow_up]})

    second_tool_output = _tool_messages(second_turn)[-1].content
    assert json.loads(second_tool_output)["order"]["status"] == "pending"
    assert "customer_id" not in second_tool_output
    assert "_verification_seal" not in second_tool_output
    assert initial_user in model.inputs[2]
    assert follow_up in model.inputs[2]


def test_multiple_tool_calls_loop_back_and_then_end(store_data: StoreData) -> None:
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_products",
                        "args": {"query": "Yamaha C40", "limit": 1},
                        "id": "multi-product-call",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_policy",
                        "args": {"query": "payment methods", "k": 1},
                        "id": "multi-policy-call",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="I found the product and the relevant payment policy."),
        ]
    )

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="Find Yamaha C40 and tell me payment options.")]}
    )

    assert len(_tool_messages(result)) == 2
    assert len(model.inputs) == 2
    assert result["messages"][-1].content == "I found the product and the relevant payment policy."


def test_model_errors_are_replaced_with_generic_customer_message(
    store_data: StoreData,
) -> None:
    class BrokenModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("provider stack and secret details")

    result = build_agent(
        model=BrokenModel(), data=store_data, policy_searcher=sample_policy_search
    ).invoke({"messages": [HumanMessage(content="Hello")]})

    assert result["messages"][-1].content == GENERIC_MODEL_FAILURE
    assert "secret details" not in result["messages"][-1].content


def test_model_configuration_requires_key_and_model_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.agent.graph.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ModelConfigurationError, match="OPENAI_API_KEY and OPENAI_MODEL"):
        build_agent()


def test_default_chat_factory_uses_native_gemini_adapter_and_existing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_google_genai

    captured: dict[str, object] = {}
    expected_model = object()

    class GeminiModelDouble:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("src.agent.graph.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding-endpoint.example/v1")
    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        GeminiModelDouble,
    )

    model = _create_chat_model()

    assert isinstance(model, GeminiModelDouble)
    assert captured == {
        "model": "gemini-3.5-flash-lite",
        "api_key": "test-key",
        "retries": 1,
        "request_timeout": 45,
    }
    assert "base_url" not in captured


def test_injected_fake_model_does_not_construct_live_gemini_adapter(
    monkeypatch: pytest.MonkeyPatch,
    store_data: StoreData,
) -> None:
    model = ScriptedChatModel([AIMessage(content="Hello from the offline fake.")])
    monkeypatch.setattr(
        "src.agent.graph._create_chat_model",
        lambda: pytest.fail("live Gemini model factory should not be used"),
    )

    result = _build(model, store_data).invoke(
        {"messages": [HumanMessage(content="Hello")]}
    )

    assert result["messages"][-1].content == "Hello from the offline fake."
