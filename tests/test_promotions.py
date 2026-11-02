from decimal import Decimal

import pytest

from src.data_loader import StoreData
from src.tools.promotions import get_active_promotion, get_product_offer


@pytest.fixture(scope="module")
def store_data() -> StoreData:
    return StoreData.load()


def test_active_promotion_is_recognized_from_dataset_flag(store_data: StoreData) -> None:
    promotion = get_active_promotion(127, data=store_data)

    assert promotion is not None
    assert promotion.product_id == 127
    assert promotion.is_active is True
    assert promotion.discount_percent == Decimal("10")


def test_all_inactive_promotion_rows_are_ignored(
    store_data: StoreData,
) -> None:
    # Product 96 has two rows with different discounts; both have is_active=0.
    assert get_active_promotion(96, data=store_data) is None


def test_pix_discount_is_five_percent_when_no_promotion_applies(
    store_data: StoreData,
) -> None:
    offer = get_product_offer(81, "pix", data=store_data)

    assert offer.promotion_active is False
    assert offer.pix_discount_applied is True
    assert offer.effective_price == Decimal("569.91")


def test_pix_discount_does_not_stack_with_active_promotion(
    store_data: StoreData,
) -> None:
    offer = get_product_offer(127, "pix", data=store_data)

    assert offer.promotion_active is True
    assert offer.promotion_price == Decimal("323.10")
    assert offer.pix_discount_applied is False
    assert offer.effective_price == Decimal("323.10")


def test_currency_math_uses_decimal_and_rounds_half_up(store_data: StoreData) -> None:
    offer = get_product_offer(83, "PIX", data=store_data)

    assert offer.base_price == Decimal("429.9")
    assert isinstance(offer.effective_price, Decimal)
    assert offer.effective_price == Decimal("408.41")


def test_unavailable_product_has_no_effective_offer_price(store_data: StoreData) -> None:
    offer = get_product_offer(96, "pix", data=store_data)

    assert offer.sellable is False
    assert offer.effective_price is None
    assert offer.pix_discount_applied is False
