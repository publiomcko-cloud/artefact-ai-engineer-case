"""Deterministic order identity verification and protected order lookup."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from src.data_loader import StoreData
from src.models import OrderItemResult, OrderStatusResult, OrderVerificationResult


class OrderLookupError(ValueError):
    """Base class for safe order lookup failures."""


class InvalidOrderIdentifierError(OrderLookupError):
    """Raised when an order identifier is not a positive integer."""


class OrderNotFoundError(OrderLookupError):
    """Raised when the requested order does not exist."""


class VerificationRequiredError(OrderLookupError):
    """Raised when order disclosure lacks a valid verification proof."""


class VerificationInformationRequiredError(OrderLookupError):
    """Raised when no customer verification value was provided."""


class OrderVerificationError(OrderLookupError):
    """Raised when supplied customer information cannot verify the order."""


class InconsistentVerificationError(OrderLookupError):
    """Raised when verification proof does not match the requested order data."""


class OrderDataIntegrityError(OrderLookupError):
    """Raised when source rows cannot safely support an order response."""


_VERIFICATION_SEAL = object()
_EMAIL_WHITESPACE = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"\D+")


def _validated_order_id(value: Any) -> int:
    if isinstance(value, bool):
        raise InvalidOrderIdentifierError("Order ID must be a positive integer.")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError, AttributeError):
        raise InvalidOrderIdentifierError("Order ID must be a positive integer.") from None
    if not parsed.is_finite() or parsed != parsed.to_integral_value() or parsed <= 0:
        raise InvalidOrderIdentifierError("Order ID must be a positive integer.")
    return int(parsed)


def _read_integer(value: Any) -> int:
    if pd.isna(value):
        raise OrderDataIntegrityError("Order data is incomplete and cannot be retrieved.")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        raise OrderDataIntegrityError("Order data is malformed and cannot be retrieved.") from None
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise OrderDataIntegrityError("Order data is malformed and cannot be retrieved.")
    return int(parsed)


def _row_by_id(frame: pd.DataFrame, column: str, identifier: int) -> pd.Series | None:
    numeric_ids = pd.to_numeric(frame[column], errors="coerce")
    matches = frame.loc[numeric_ids == identifier]
    if len(matches) > 1:
        raise OrderDataIntegrityError("Order data contains an ambiguous relationship.")
    return None if matches.empty else matches.iloc[0]


def _required_row(frame: pd.DataFrame, column: str, identifier: int) -> pd.Series:
    row = _row_by_id(frame, column, identifier)
    if row is None:
        raise OrderDataIntegrityError("Order data is incomplete and cannot be retrieved.")
    return row


def _optional_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: Any) -> str:
    text = _optional_text(value)
    if text is None:
        raise OrderDataIntegrityError("Order data is incomplete and cannot be retrieved.")
    return text


def _parse_date(value: Any, *, optional: bool = False) -> date | None:
    text = _optional_text(value)
    if text is None and optional:
        return None
    if text is None:
        raise OrderDataIntegrityError("Order data is incomplete and cannot be retrieved.")
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise OrderDataIntegrityError("Order data contains an invalid date.") from None


def _normalize_email(value: Any) -> str:
    text = _optional_text(value)
    if text is None:
        return ""
    return _EMAIL_WHITESPACE.sub("", text).casefold()


def _normalize_phone(value: Any) -> str:
    text = _optional_text(value)
    if text is None:
        return ""
    return _NON_DIGIT.sub("", text)


def _customer_credential_matches(
    customers: pd.DataFrame,
    *,
    field: str,
    supplied: str,
    customer_id: int,
) -> bool:
    normalizer = _normalize_email if field == "email" else _normalize_phone
    normalized = normalizer(supplied)
    if not normalized:
        return False
    matches = customers.loc[customers[field].map(normalizer) == normalized]
    # A duplicated value is not sufficient evidence for identifying one customer.
    if len(matches) != 1:
        return False
    return _read_integer(matches.iloc[0]["customer_id"]) == customer_id


def verify_order_identity(
    order_id: int,
    email: str | None = None,
    phone: str | None = None,
    *,
    data: StoreData | None = None,
) -> OrderVerificationResult:
    """Verify one or more exact customer values against the order's customer."""
    requested_order_id = _validated_order_id(order_id)
    credentials: list[tuple[str, str]] = []
    for field, value in (("email", email), ("phone", phone)):
        if value is None:
            continue
        if not isinstance(value, str):
            raise OrderVerificationError("Customer verification failed.")
        if value.strip():
            credentials.append((field, value))
    if not credentials:
        raise VerificationInformationRequiredError(
            "Provide the customer email or phone to verify this order."
        )

    store = data or StoreData.load()
    order = _row_by_id(store.orders, "order_id", requested_order_id)
    if order is None:
        raise OrderNotFoundError("No order was found for the supplied order ID.")

    customer_id = _read_integer(order["customer_id"])
    _required_row(store.customers, "customer_id", customer_id)
    if not all(
        _customer_credential_matches(
            store.customers,
            field=field,
            supplied=value,
            customer_id=customer_id,
        )
        for field, value in credentials
    ):
        raise OrderVerificationError("Customer verification failed.")

    result = OrderVerificationResult(order_id=requested_order_id)
    # Store capability details privately: public model dumps contain no customer ID.
    result.__pydantic_private__.update(
        {
            "_customer_id": customer_id,
            "_verified_order_id": requested_order_id,
            "_verification_seal": _VERIFICATION_SEAL,
        }
    )
    return result


def _require_verification(
    order_id: int,
    verification: OrderVerificationResult | None,
) -> OrderVerificationResult:
    if type(verification) is not OrderVerificationResult:
        raise VerificationRequiredError("Successful customer verification is required.")
    if verification._verification_seal is not _VERIFICATION_SEAL:
        raise VerificationRequiredError("Successful customer verification is required.")
    if (
        verification.verified is not True
        or verification.order_id != order_id
        or verification._verified_order_id != order_id
    ):
        raise InconsistentVerificationError(
            "The verified identity does not match the requested order."
        )
    return verification


def get_order_status(
    order_id: int,
    verification: OrderVerificationResult | None = None,
    *,
    data: StoreData | None = None,
) -> OrderStatusResult:
    """Return minimal order details only with proof from the verification function."""
    requested_order_id = _validated_order_id(order_id)
    proof = _require_verification(requested_order_id, verification)
    store = data or StoreData.load()
    order = _row_by_id(store.orders, "order_id", requested_order_id)
    if order is None:
        raise OrderNotFoundError("No order was found for the supplied order ID.")

    customer_id = _read_integer(order["customer_id"])
    _required_row(store.customers, "customer_id", customer_id)
    if proof._customer_id != customer_id:
        raise InconsistentVerificationError(
            "The verified identity does not match the requested order."
        )

    items_frame = store.order_items
    item_order_ids = pd.to_numeric(items_frame["order_id"], errors="coerce")
    order_items = items_frame.loc[item_order_ids == requested_order_id]
    if order_items.empty:
        raise OrderDataIntegrityError("Order items are unavailable for this order.")

    items: list[OrderItemResult] = []
    for _, item in order_items.iterrows():
        product_id = _read_integer(item["product_id"])
        quantity = _read_integer(item["quantity"])
        if quantity <= 0:
            raise OrderDataIntegrityError("Order data contains an invalid item quantity.")
        product = _row_by_id(store.products, "product_id", product_id)
        if product is None:
            raise OrderDataIntegrityError("An order item references an unavailable product.")
        items.append(
            OrderItemResult(
                product_id=product_id,
                product_name=_required_text(product["name"]),
                quantity=quantity,
            )
        )

    total = order["total_brl"]
    if not isinstance(total, Decimal) or not total.is_finite():
        raise OrderDataIntegrityError(
            "Historical order total is unavailable as an exact decimal value."
        )

    return OrderStatusResult(
        order_id=requested_order_id,
        status=_required_text(order["status"]),
        total_brl=total,
        order_date=_parse_date(order["order_date"]),
        payment_method=_required_text(order["payment_method"]),
        estimated_delivery=_parse_date(order["estimated_delivery"], optional=True),
        tracking_code=_optional_text(order["tracking_code"]),
        items=tuple(items),
    )
