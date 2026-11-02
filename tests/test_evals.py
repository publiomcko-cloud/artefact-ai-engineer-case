from __future__ import annotations

from decimal import Decimal

from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import GoogleAPIError

import evals.run_evals as eval_runner
from evals.run_evals import (
    ProviderAudit,
    ThrottledModel,
    ToolEvent,
    TurnRecord,
    _build_context,
    _is_retryable_failure,
    _invoke,
    _mentions_value,
    _currency_values,
    _redact,
    evaluate_step,
    load_cases,
    safe_tool_summary,
)
from src.data_loader import StoreData


def test_case_file_covers_required_live_scenarios_and_robustness() -> None:
    cases = load_cases()
    steps = [
        step
        for group in (cases["conversations"], cases["robustness"])
        for conversation in group
        for step in conversation["steps"]
    ]
    ids = {step["id"] for step in steps}

    assert {f"E2E-{number:02d}" for number in range(1, 13)} <= ids
    assert {"R-01", "R-02", "R-03", "R-04"} <= ids


def test_evaluator_checks_structured_budget_and_currency_grounding_offline() -> None:
    data = StoreData.load()
    step = {
        "expected_tools": ["search_products"],
        "checks": {
            "tool_results": [
                {
                    "tool": "search_products",
                    "path": "products",
                    "min_items": 1,
                    "every": {
                        "sellable": {"equals": True},
                        "price_brl": {"lte": "1000.00"},
                    },
                }
            ],
            "assistant": {"currency_grounded": True},
        },
    }
    payload = {
        "ok": True,
        "products": [
            {"name": "Yamaha C40", "price_brl": "599.90", "sellable": True},
            {"name": "Yamaha FG800", "price_brl": "1299.00", "sellable": True},
        ],
    }
    events = [ToolEvent("search_products", payload, "safe summary")]

    failures = evaluate_step(step, events, "The guitar costs R$ 599,90.", data)

    assert any("not every" in failure for failure in failures)
    assert not any("currency amount" in failure for failure in failures)


def test_order_tool_summary_excludes_identity_and_verification_fields() -> None:
    summary = safe_tool_summary(
        "lookup_order",
        {
            "ok": True,
            "order": {
                "order_id": 15,
                "status": "pending",
                "total_brl": "8348.00",
                "items": [{"product_name": "Synthetic instrument"}],
                "tracking_code": None,
                "customer_id": 18,
                "email": "synthetic@example.invalid",
                "_verification_seal": "must-not-appear",
            },
        },
    )

    assert "pending" in summary
    assert "8348.00" in summary
    assert "synthetic@example.invalid" not in summary
    assert "customer_id" not in summary
    assert "must-not-appear" not in summary


def test_dynamic_context_and_record_redaction_use_synthetic_data_safely() -> None:
    data = StoreData.load()
    context = _build_context(data)

    assert int(context["promotion_product_id"]) in set(data.promotions.loc[data.promotions["is_active"], "product_id"])
    assert "Giannini" in context["unavailable_product"]
    assert "@" in context["order_identity"]
    assert "@" not in _redact(
        f"My email is {context['order_identity']}.", context
    )


def test_tool_currency_values_only_come_from_currency_fields() -> None:
    events = [
        ToolEvent(
            "get_product_offer",
            {"offer": {"effective_price_brl": "569.91", "product_id": 81}},
            "safe summary",
        )
    ]

    assert _currency_values(events) == {Decimal("569.91")}


def test_report_assertion_matches_brazilian_currency_formatting() -> None:
    assert _mentions_value("The total is R$ 8.348,00.", "8348.00")


def test_currency_grounding_accepts_period_punctuation_after_tool_values() -> None:
    data = StoreData.load()
    step = {"checks": {"assistant": {"currency_grounded": True}}}
    events = [
        ToolEvent(
            "search_products",
            {"products": [{"price_brl": "799.00"}]},
            "safe summary",
        ),
        ToolEvent(
            "get_product_offer",
            {"offer": {"effective_price_brl": "759.05"}},
            "safe summary",
        ),
    ]

    failures = evaluate_step(
        step,
        events,
        "The catalog price is R$ 799.00. The PIX price is R$ 759.05.",
        data,
    )

    assert failures == []


def test_currency_grounding_allows_repeating_the_users_budget() -> None:
    data = StoreData.load()
    step = {
        "user": "Which guitars do you have for up to R$ 1,000?",
        "checks": {"assistant": {"currency_grounded": True}},
    }
    events = [
        ToolEvent(
            "search_products",
            {"products": [{"price_brl": "599.90"}]},
            "safe summary",
        )
    ]

    failures = evaluate_step(
        step,
        events,
        "This guitar is under R$ 1,000 and costs R$ 599.90.",
        data,
    )

    assert failures == []


def test_ambiguous_product_check_accepts_multiple_safe_candidates() -> None:
    data = StoreData.load()
    step = {
        "expected_tools": ["get_product"],
        "checks": {
            "assistant": {
                "candidate_names_or_contains_any": {
                    "tool": "get_product",
                    "minimum": 2,
                    "contains_any": ["clarify"],
                }
            }
        },
    }
    events = [
        ToolEvent(
            "get_product",
            {
                "ok": False,
                "error": {"code": "ambiguous_product"},
                "candidates": [
                    {"name": "Yamaha C40 Nylon Natural", "price_brl": "599.90"},
                    {"name": "Yamaha C70 Nylon Natural", "price_brl": "849.00"},
                ],
            },
            "safe summary",
        )
    ]

    failures = evaluate_step(
        step,
        events,
        "I found Yamaha C40 Nylon Natural and Yamaha C70 Nylon Natural.",
        data,
    )

    assert failures == []


def test_brand_search_check_accepts_a_safe_candidate_list() -> None:
    data = StoreData.load()
    step = {
        "expected_tools": ["search_products"],
        "checks": {
            "tool_results": [
                {"tool": "search_products", "path": "products", "min_items": 2}
            ],
            "assistant": {
                "candidate_names_or_contains_any": {
                    "tool": "search_products",
                    "minimum": 2,
                    "contains_any": ["clarify"],
                }
            },
        },
    }
    events = [
        ToolEvent(
            "search_products",
            {
                "products": [
                    {"name": "Yamaha C40 Nylon Natural"},
                    {"name": "Yamaha C70 Nylon Natural"},
                ]
            },
            "safe summary",
        )
    ]

    assert evaluate_step(
        step,
        events,
        "Here are Yamaha C40 Nylon Natural and Yamaha C70 Nylon Natural.",
        data,
    ) == []


def test_transient_provider_failures_use_bounded_exponential_backoff(monkeypatch) -> None:
    audit = ProviderAudit()
    sleeps = []

    class FlakyAgent:
        attempts = 0

        def invoke(self, _payload):
            self.attempts += 1
            if self.attempts <= 2:
                audit.failures.append({"type": "GoogleRateLimitError", "status": None})
                raise RuntimeError("simulated transient provider error")
            return {"messages": [AIMessage(content="recovered")]}

    monkeypatch.setattr(eval_runner.time, "sleep", sleeps.append)
    agent = FlakyAgent()

    messages, delta, failure = _invoke(
        agent, audit, [], "probe", retry_base_delay=2, max_retries=2, retry_max_delay=8
    )

    assert agent.attempts == 3
    assert sleeps == [2, 4]
    assert failure is None
    assert messages == delta
    assert messages[-1].content == "recovered"


def test_persistent_provider_failures_stop_after_retry_limit(monkeypatch) -> None:
    audit = ProviderAudit()
    sleeps = []

    class PersistentlyUnavailableAgent:
        attempts = 0

        def invoke(self, _payload):
            self.attempts += 1
            audit.failures.append({"type": "APIStatusError", "status": 503})
            raise RuntimeError("simulated persistent provider error")

    monkeypatch.setattr(eval_runner.time, "sleep", sleeps.append)
    agent = PersistentlyUnavailableAgent()

    messages, delta, failure = _invoke(
        agent, audit, [], "probe", retry_base_delay=2, max_retries=2, retry_max_delay=8
    )

    assert agent.attempts == 3
    assert sleeps == [2, 4]
    assert failure == {"type": "APIStatusError", "status": 503, "attempts": 3}
    assert messages == delta == []


def test_google_transient_error_classes_use_backoff_without_http_status() -> None:
    assert _is_retryable_failure({"type": "GoogleRateLimitError", "status": None})
    assert _is_retryable_failure({"type": "GoogleAPIError", "status": None})
    assert _is_retryable_failure({"type": "GoogleAPIError", "status": 503})
    assert not _is_retryable_failure({"type": "GoogleAPIError", "status": 400})
    assert not _is_retryable_failure({"type": "GoogleInvalidRequestError", "status": 400})


def test_throttled_model_records_google_http_status_without_response_details() -> None:
    class BoundModel:
        def invoke(self, _messages):
            raise GoogleAPIError(code=503, response_json={"error": {"message": "temporary"}})

    class Model:
        def bind_tools(self, _tools):
            return BoundModel()

    audit = ProviderAudit(min_interval=0)
    bound = ThrottledModel(Model(), audit).bind_tools([])

    try:
        bound.invoke([])
    except GoogleAPIError:
        pass
    else:
        raise AssertionError("The provider exception should be preserved.")

    assert audit.failures == [{"type": "GoogleAPIError", "status": 503}]


def test_provider_rejection_blocks_behavior_verdict_without_false_failure() -> None:
    data = StoreData.load()
    step = {
        "expected_tools": ["get_product"],
        "checks": {
            "tool_results": [
                {"tool": "get_product", "path": "product.price_brl", "equals": "599.90"}
            ],
            "assistant": {"contains_any": ["Yamaha", "guitar"]},
        },
    }
    events = [
        ToolEvent(
            "get_product",
            {"ok": True, "product": {"price_brl": "599.90"}},
            "safe summary",
        )
    ]
    failures = evaluate_step(
        step, events, "I could not retrieve that information right now.", data, check_assistant=False
    )
    record = TurnRecord(
        scenario_id="E2E-02",
        conversation_id="current_price",
        user_turn="How much is the Yamaha C40?",
        assistant_answer="I could not retrieve that information right now.",
        tools=["get_product"],
        summaries=["safe summary"],
        failures=failures,
        provider_failure={"type": "OpenAIInvalidRequestError", "status": 400, "attempts": 1},
    )

    assert failures == []
    assert record.status == "BLOCKED"
