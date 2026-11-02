from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.agent.tools import build_agent_tools
from src.data_loader import StoreData
from src.rag.retriever import PolicySearchResult


@pytest.fixture(scope="module")
def store_data() -> StoreData:
    return StoreData.load()


def sample_policy_search(query: str, k: int) -> list[PolicySearchResult]:
    assert query.strip()
    assert 1 <= k <= 5
    return [
        PolicySearchResult(
            chunk_id="local-policy-chunk",
            text="Trocas e devoluções seguem as condições desta política.",
            section="4. Política de Trocas e Devoluções",
            page_start=4,
            page_end=4,
            pages=(4,),
            source="data/store_policies.pdf",
            score=0.91,
        )
    ]


def _tool_map(data: StoreData, policy_searcher=sample_policy_search):
    return {
        item.name: item
        for item in build_agent_tools(data=data, policy_searcher=policy_searcher)
    }


def _call(tool, arguments: dict[str, object]) -> dict[str, object]:
    result = tool.invoke(arguments)
    assert isinstance(result, str)
    return json.loads(result)


def test_product_tool_returns_structured_json_and_exact_money(
    store_data: StoreData,
) -> None:
    tools = _tool_map(store_data)

    result = _call(tools["get_product"], {"product_query": "Yamaha C40"})

    assert result["ok"] is True
    assert result["product"] == {
        "product_id": 81,
        "name": "Yamaha C40 Nylon Natural",
        "category": "Violões",
        "price_brl": "599.90",
        "stock_quantity": 12,
        "status": "active",
        "sellable": True,
    }


def test_search_and_offer_tools_return_json_safe_decimal_strings(
    store_data: StoreData,
) -> None:
    tools = _tool_map(store_data)
    products = _call(
        tools["search_products"],
        {"query": "Yamaha C40", "max_price": "1000", "limit": 2},
    )
    offer = _call(
        tools["get_product_offer"],
        {"product_id": 81, "payment_method": "pix"},
    )
    promotion = _call(
        tools["get_product_offer"],
        {"product_id": 127, "payment_method": "pix"},
    )

    assert products["products"][0]["price_brl"] == "599.90"
    assert offer["offer"]["base_price_brl"] == "599.90"
    assert offer["offer"]["effective_price_brl"] == "569.91"
    assert isinstance(offer["offer"]["effective_price_brl"], str)
    assert promotion["offer"]["promotion_active"] is True
    assert promotion["offer"]["pix_discount_applied"] is False
    assert promotion["offer"]["effective_price_brl"] == "323.10"


def test_ambiguous_product_returns_safe_candidates_without_exception_text(
    store_data: StoreData,
) -> None:
    result = _call(_tool_map(store_data)["get_product"], {"product_query": "Yamaha"})

    assert result["ok"] is False
    assert result["error"]["code"] == "ambiguous_product"
    assert "Matching products" not in result["error"]["message"]
    assert len(result["candidates"]) > 1
    assert all("product_id" in item and "price_brl" in item for item in result["candidates"])


def test_policy_tool_returns_evidence_and_source_metadata(
    store_data: StoreData,
) -> None:
    result = _call(
        _tool_map(store_data)["search_policy"],
        {"query": "How do exchanges work?", "k": 2},
    )

    assert result["ok"] is True
    assert result["evidence"] == [
        {
            "section": "4. Política de Trocas e Devoluções",
            "pages": [4],
            "source": "data/store_policies.pdf",
            "text": "Trocas e devoluções seguem as condições desta política.",
            "score": 0.91,
        }
    ]


def test_order_tool_requires_verification_information(store_data: StoreData) -> None:
    result = _call(_tool_map(store_data)["lookup_order"], {"order_id": 1})

    assert result == {
        "ok": False,
        "error": {
            "code": "verification_required",
            "message": "Please provide the email or phone associated with this order.",
        },
    }


def test_failed_order_verification_returns_no_protected_details(
    store_data: StoreData,
) -> None:
    result = _call(
        _tool_map(store_data)["lookup_order"],
        {"order_id": 1, "email": "wrong@example.com"},
    )
    serialized = json.dumps(result).casefold()

    assert result["error"]["code"] == "verification_failed"
    assert "delivered" not in serialized
    assert "11499" not in serialized
    assert "brab1234567br" not in serialized
    assert "customer_id" not in serialized


def test_successful_order_tool_returns_only_allowed_fields(
    store_data: StoreData,
) -> None:
    customer = store_data.customers.loc[
        store_data.customers["customer_id"] == 3
    ].iloc[0]
    result = _call(
        _tool_map(store_data)["lookup_order"],
        {"order_id": 1, "email": customer["email"]},
    )
    order = result["order"]

    assert result["ok"] is True
    assert order["total_brl"] == "11499.00"
    assert order["order_date"] == "2025-10-15"
    assert order["tracking_code"] == "BRAB1234567BR"
    assert order["items"] == [
        {
            "product_id": 93,
            "product_name": "Martin D-28 Dreadnought Natural",
            "quantity": 1,
        }
    ]
    assert set(order) == {
        "order_id",
        "status",
        "total_brl",
        "order_date",
        "payment_method",
        "estimated_delivery",
        "tracking_code",
        "items",
    }
    serialized = json.dumps(result).casefold()
    assert "customer_id" not in serialized
    assert "email" not in serialized
    assert "phone" not in serialized
    assert "notes" not in serialized
    assert "verification" not in serialized


def test_order_tool_keeps_missing_tracking_absent(store_data: StoreData) -> None:
    customer = store_data.customers.loc[
        store_data.customers["customer_id"] == 6
    ].iloc[0]
    result = _call(
        _tool_map(store_data)["lookup_order"],
        {"order_id": 9, "phone": customer["phone"]},
    )

    assert result["ok"] is True
    assert result["order"]["tracking_code"] is None
    assert result["order"]["estimated_delivery"] is None


def test_policy_failure_is_generic_and_does_not_expose_exception(
    store_data: StoreData,
) -> None:
    def broken_search(_: str, __: int) -> list[PolicySearchResult]:
        raise RuntimeError("provider-secret-stack-detail")

    result = _call(
        _tool_map(store_data, policy_searcher=broken_search)["search_policy"],
        {"query": "payment methods"},
    )

    assert result["error"]["code"] == "policy_unavailable"
    assert "provider-secret-stack-detail" not in json.dumps(result)
