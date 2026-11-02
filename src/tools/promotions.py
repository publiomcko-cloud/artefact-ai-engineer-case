"""Deterministic promotion lookup and current offer calculation."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import pandas as pd

from src.data_loader import StoreData
from src.models import ProductOffer, PromotionRecord
from src.tools.products import get_product


CENT = Decimal("0.01")
PIX_DISCOUNT_PERCENT = Decimal("5")


class MultipleActivePromotionsError(ValueError):
    """Raised when the data contains multiple active promotions for one product."""


def _as_decimal(value: Any, *, field_name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        raise ValueError(f"{field_name} must be a valid decimal number.") from None
    if not result.is_finite():
        raise ValueError(f"{field_name} must be a finite decimal number.")
    return result


def _is_active(value: Any) -> bool:
    """Interpret only the explicit active flag; descriptions and dates are ignored."""
    if pd.isna(value):
        raise ValueError("Promotion is_active cannot be empty.")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError("Promotion is_active must be an explicit 0/1 or true/false value.")


def _promotion_from_row(row: pd.Series) -> PromotionRecord:
    return PromotionRecord(
        promotion_id=int(row["promotion_id"]),
        product_id=int(row["product_id"]),
        discount_percent=_as_decimal(row["discount_percent"], field_name="discount_percent"),
        description=str(row["description"]),
        is_active=_is_active(row["is_active"]),
    )


def get_active_promotion(
    product_id: int,
    *,
    data: StoreData | None = None,
) -> PromotionRecord | None:
    """Return the single active promotion recorded for a product, if any."""
    store = data if data is not None else StoreData.load()
    matches = [
        _promotion_from_row(row)
        for _, row in store.promotions.iterrows()
        if int(row["product_id"]) == int(product_id) and _is_active(row["is_active"])
    ]
    if len(matches) > 1:
        ids = ", ".join(str(promotion.promotion_id) for promotion in matches)
        raise MultipleActivePromotionsError(
            f"Product {product_id} has multiple active promotions ({ids}); "
            "the offer cannot be calculated safely."
        )
    return matches[0] if matches else None


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def get_product_offer(
    product_id: str | int,
    payment_method: str | None = None,
    *,
    data: StoreData | None = None,
) -> ProductOffer:
    """Calculate the effective price without stacking PIX and product promotion."""
    store = data if data is not None else StoreData.load()
    product = get_product(product_id, data=store)
    promotion = get_active_promotion(product.product_id, data=store)

    if not product.sellable:
        return ProductOffer(
            product_id=product.product_id,
            product_name=product.name,
            sellable=False,
            base_price=product.price_brl,
            promotion_active=promotion is not None,
            promotion_description=promotion.description if promotion else None,
            promotion_discount_percent=promotion.discount_percent if promotion else None,
        )

    if promotion is not None:
        promotion_price = _money(
            product.price_brl
            * (Decimal("100") - promotion.discount_percent)
            / Decimal("100")
        )
        return ProductOffer(
            product_id=product.product_id,
            product_name=product.name,
            sellable=True,
            base_price=product.price_brl,
            promotion_active=True,
            promotion_description=promotion.description,
            promotion_discount_percent=promotion.discount_percent,
            promotion_price=promotion_price,
            pix_discount_applied=False,
            effective_price=promotion_price,
        )

    is_pix = payment_method is not None and payment_method.strip().casefold() == "pix"
    effective_price = product.price_brl
    if is_pix:
        effective_price = _money(
            product.price_brl
            * (Decimal("100") - PIX_DISCOUNT_PERCENT)
            / Decimal("100")
        )
    return ProductOffer(
        product_id=product.product_id,
        product_name=product.name,
        sellable=True,
        base_price=product.price_brl,
        promotion_active=False,
        pix_discount_applied=is_pix,
        effective_price=effective_price,
    )
