from __future__ import annotations

import re
import shutil
from decimal import Decimal

import pandas as pd
import pytest

from src.data_loader import StoreData
from src.models import OrderVerificationResult
from src.tools.orders import (
    InconsistentVerificationError,
    OrderDataIntegrityError,
    OrderNotFoundError,
    OrderVerificationError,
    VerificationInformationRequiredError,
    VerificationRequiredError,
    get_order_status,
    verify_order_identity,
)


@pytest.fixture(scope="module")
def store_data() -> StoreData:
    return StoreData.load()


def _customer_for_order(data: StoreData, order_id: int) -> pd.Series:
    order = data.orders.loc[data.orders["order_id"] == order_id].iloc[0]
    return data.customers.loc[
        data.customers["customer_id"] == order["customer_id"]
    ].iloc[0]


def _verified_order(data: StoreData, order_id: int) -> OrderVerificationResult:
    customer = _customer_for_order(data, order_id)
    return verify_order_identity(
        order_id,
        email=str(customer["email"]),
        data=data,
    )


def test_order_id_alone_does_not_authorize_disclosure(store_data: StoreData) -> None:
    with pytest.raises(VerificationInformationRequiredError, match="email or phone"):
        verify_order_identity(1, data=store_data)
    with pytest.raises(VerificationRequiredError):
        get_order_status(1, data=store_data)


def test_correct_email_verifies_only_the_associated_order(store_data: StoreData) -> None:
    customer = _customer_for_order(store_data, 1)

    result = verify_order_identity(1, email=str(customer["email"]), data=store_data)

    assert result.order_id == 1
    assert result.verified is True
    assert "customer_id" not in result.model_dump()


def test_both_correct_credentials_verify_the_order(store_data: StoreData) -> None:
    customer = _customer_for_order(store_data, 1)

    result = verify_order_identity(
        1,
        email=str(customer["email"]),
        phone=str(customer["phone"]),
        data=store_data,
    )

    assert result.verified is True


def test_email_comparison_ignores_case_and_surrounding_whitespace(
    store_data: StoreData,
) -> None:
    customer = _customer_for_order(store_data, 1)

    result = verify_order_identity(
        1,
        email=f"  {str(customer['email']).upper()}\n",
        data=store_data,
    )

    assert result.verified is True


def test_wrong_email_fails_without_disclosing_order_details(
    store_data: StoreData,
) -> None:
    with pytest.raises(OrderVerificationError) as error:
        verify_order_identity(1, email="someone.else@example.com", data=store_data)

    message = str(error.value).casefold()
    assert "delivered" not in message
    assert "11499" not in message
    assert "brab1234567br" not in message
    assert "customer" in message


def test_correct_phone_verifies_and_harmless_formatting_is_normalized(
    store_data: StoreData,
) -> None:
    customer = _customer_for_order(store_data, 8)
    digits = re.sub(r"\D", "", str(customer["phone"]))
    formatted = f"{digits[:2]}-{digits[2:7]}.{digits[7:]}"

    result = verify_order_identity(8, phone=formatted, data=store_data)

    assert result.verified is True


def test_credentials_for_another_customer_do_not_verify_the_order(
    store_data: StoreData,
) -> None:
    other_customer = _customer_for_order(store_data, 2)

    with pytest.raises(OrderVerificationError):
        verify_order_identity(
            1,
            email=str(other_customer["email"]),
            phone=str(other_customer["phone"]),
            data=store_data,
        )


def test_all_supplied_credentials_must_match(store_data: StoreData) -> None:
    customer = _customer_for_order(store_data, 1)

    with pytest.raises(OrderVerificationError):
        verify_order_identity(
            1,
            email="wrong@example.com",
            phone=str(customer["phone"]),
            data=store_data,
        )


def test_duplicate_email_is_not_accepted_as_unique_identity(
    store_data: StoreData,
) -> None:
    customer = _customer_for_order(store_data, 10)
    assert (
        store_data.customers["email"].str.casefold()
        == str(customer["email"]).casefold()
    ).sum() == 2

    with pytest.raises(OrderVerificationError):
        verify_order_identity(10, email=str(customer["email"]), data=store_data)

    # The associated customer's distinct phone remains usable.
    assert verify_order_identity(
        10, phone=str(customer["phone"]), data=store_data
    ).verified


def test_unknown_order_fails_clearly(store_data: StoreData) -> None:
    with pytest.raises(OrderNotFoundError, match="No order was found"):
        verify_order_identity(999999, email="a@example.com", data=store_data)


def test_caller_constructed_true_boolean_result_cannot_bypass_verification(
    store_data: StoreData,
) -> None:
    forged = OrderVerificationResult(order_id=1, verified=True)

    with pytest.raises(VerificationRequiredError):
        get_order_status(1, forged, data=store_data)


def test_verified_identity_cannot_be_reused_for_another_order(
    store_data: StoreData,
) -> None:
    proof = _verified_order(store_data, 1)

    with pytest.raises(InconsistentVerificationError, match="does not match"):
        get_order_status(2, proof, data=store_data)


def test_copying_verification_with_a_different_order_id_does_not_rebind_proof(
    store_data: StoreData,
) -> None:
    proof = _verified_order(store_data, 1)
    rebound_proof = proof.model_copy(update={"order_id": 19})

    with pytest.raises(InconsistentVerificationError):
        get_order_status(19, rebound_proof, data=store_data)


def test_verified_request_returns_only_the_requested_order(
    store_data: StoreData,
) -> None:
    result = get_order_status(1, _verified_order(store_data, 1), data=store_data)

    assert result.order_id == 1
    assert result.status == "delivered"
    assert result.payment_method == "pix"
    assert result.tracking_code == "BRAB1234567BR"
    assert "customer_id" not in result.model_dump()
    assert "email" not in result.model_dump()
    assert "phone" not in result.model_dump()
    assert "notes" not in result.model_dump()
    assert set(result.model_dump()) == {
        "order_id",
        "status",
        "total_brl",
        "order_date",
        "payment_method",
        "estimated_delivery",
        "tracking_code",
        "items",
    }


def test_historical_total_is_a_decimal_copied_from_order_history(
    store_data: StoreData,
) -> None:
    result = get_order_status(1, _verified_order(store_data, 1), data=store_data)
    source_total = store_data.orders.loc[
        store_data.orders["order_id"] == 1, "total_brl"
    ].iloc[0]

    assert isinstance(source_total, Decimal)
    assert isinstance(result.total_brl, Decimal)
    assert result.total_brl == source_total == Decimal("11499")


def test_current_product_price_changes_do_not_change_historical_total(
    store_data: StoreData,
) -> None:
    proof = _verified_order(store_data, 1)
    original = get_order_status(1, proof, data=store_data)
    changed_products = store_data.products.copy(deep=True)
    changed_products.loc[changed_products["product_id"] == 93, "price_brl"] = "0.01"
    changed_catalog = StoreData(
        products=changed_products,
        categories=store_data.categories,
        promotions=store_data.promotions,
        customers=store_data.customers,
        orders=store_data.orders,
        order_items=store_data.order_items,
    )

    after_catalog_change = get_order_status(1, proof, data=changed_catalog)

    assert after_catalog_change.total_brl == original.total_brl == Decimal("11499")


def test_order_items_and_product_names_come_from_their_source_tables(
    store_data: StoreData,
) -> None:
    result = get_order_status(1, _verified_order(store_data, 1), data=store_data)

    assert len(result.items) == 1
    assert result.items[0].product_id == 93
    assert result.items[0].product_name == "Martin D-28 Dreadnought Natural"
    assert result.items[0].quantity == 1


def test_multi_product_order_returns_each_dataset_item(store_data: StoreData) -> None:
    result = get_order_status(15, _verified_order(store_data, 15), data=store_data)

    assert {(item.product_id, item.quantity) for item in result.items} == {
        (82, 1),
        (106, 1),
    }


def test_historical_item_quantity_is_returned_without_recalculation(
    store_data: StoreData,
) -> None:
    result = get_order_status(3, _verified_order(store_data, 3), data=store_data)

    assert len(result.items) == 1
    assert result.items[0].product_id == 130
    assert result.items[0].quantity == 2


def test_tracking_is_returned_only_when_present(store_data: StoreData) -> None:
    with_tracking = get_order_status(8, _verified_order(store_data, 8), data=store_data)
    without_tracking = get_order_status(9, _verified_order(store_data, 9), data=store_data)

    assert with_tracking.tracking_code == "BRJL5544332BR"
    assert without_tracking.tracking_code is None
    assert without_tracking.estimated_delivery is None


def test_missing_referenced_product_fails_without_inventing_item_data(
    store_data: StoreData,
) -> None:
    proof = _verified_order(store_data, 1)
    changed_items = store_data.order_items.copy(deep=True)
    changed_items.loc[changed_items["order_id"] == 1, "product_id"] = 999999
    malformed_data = StoreData(
        products=store_data.products,
        categories=store_data.categories,
        promotions=store_data.promotions,
        customers=store_data.customers,
        orders=store_data.orders,
        order_items=changed_items,
    )

    with pytest.raises(OrderDataIntegrityError, match="unavailable product"):
        get_order_status(1, proof, data=malformed_data)


def test_order_totals_are_parsed_directly_from_text_as_exact_decimal(
    tmp_path,
) -> None:
    for filename in (
        "products.csv",
        "categories.csv",
        "promotions.csv",
        "customers.csv",
        "orders.csv",
        "order_items.csv",
    ):
        shutil.copyfile(f"data/{filename}", tmp_path / filename)
    orders_path = tmp_path / "orders.csv"
    orders = pd.read_csv(orders_path, dtype=str, keep_default_na=False)
    orders.loc[orders["order_id"] == "1", "total_brl"] = "0.123456789012345678901"
    orders.to_csv(orders_path, index=False)

    loaded = StoreData.load(tmp_path)

    assert loaded.orders.loc[loaded.orders["order_id"] == 1, "total_brl"].iloc[0] == Decimal(
        "0.123456789012345678901"
    )


def test_malformed_customer_relationship_fails_closed(store_data: StoreData) -> None:
    order = store_data.orders.loc[store_data.orders["order_id"] == 1].iloc[0]
    remaining_customers = store_data.customers.loc[
        store_data.customers["customer_id"] != order["customer_id"]
    ].copy()
    malformed_data = StoreData(
        products=store_data.products,
        categories=store_data.categories,
        promotions=store_data.promotions,
        customers=remaining_customers,
        orders=store_data.orders,
        order_items=store_data.order_items,
    )

    with pytest.raises(OrderDataIntegrityError):
        verify_order_identity(1, email="pedro.oliveira@jmail.com", data=malformed_data)
