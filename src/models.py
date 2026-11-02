"""Typed result models for deterministic product and promotion tools."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict


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
