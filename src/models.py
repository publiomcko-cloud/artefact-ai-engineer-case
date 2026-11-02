"""Typed result models for deterministic store tools."""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr


class ProductRecord(BaseModel):
    """Authoritative catalog information for one product."""

    model_config = ConfigDict(frozen=True)

    product_id: int
    name: str
    category: str
    price_brl: Decimal
    stock_quantity: int
    status: str
    sellable: bool


class PromotionRecord(BaseModel):
    """An explicit promotion row whose active state comes from the CSV."""

    model_config = ConfigDict(frozen=True)

    promotion_id: int
    product_id: int
    discount_percent: Decimal
    description: str
    is_active: bool


class ProductOffer(BaseModel):
    """Current deterministic offer for a catalog product."""

    model_config = ConfigDict(frozen=True)

    product_id: int
    product_name: str
    sellable: bool
    base_price: Decimal
    promotion_active: bool
    promotion_description: str | None = None
    promotion_discount_percent: Decimal | None = None
    promotion_price: Decimal | None = None
    pix_discount_applied: bool = False
    effective_price: Decimal | None = None


class OrderVerificationResult(BaseModel):
    """Opaque, order-bound proof created only by successful identity verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: int
    verified: Literal[True] = True

    # These private attributes are omitted from model dumps and tool-facing output.
    _customer_id: int = PrivateAttr(default=0)
    _verified_order_id: int = PrivateAttr(default=0)
    _verification_seal: object | None = PrivateAttr(default=None)


class OrderItemResult(BaseModel):
    """A product and quantity attached to a historical order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: int
    product_name: str
    quantity: int


class OrderStatusResult(BaseModel):
    """Minimal protected details for one identity-verified order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: int
    status: str
    total_brl: Decimal
    order_date: date
    payment_method: str
    estimated_delivery: date | None = None
    tracking_code: str | None = None
    items: tuple[OrderItemResult, ...]
