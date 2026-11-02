from decimal import Decimal

import pandas as pd
import pytest

from src.data_loader import StoreData
from src.tools.products import (
    AmbiguousProductError,
    ProductNotFoundError,
    get_product,
    is_sellable,
    search_products,
)


@pytest.fixture(scope="module")
def store_data() -> StoreData:
    return StoreData.load()


def test_active_product_with_positive_stock_is_sellable(store_data: StoreData) -> None:
    product = get_product(81, data=store_data)

    assert product.status == "active"
    assert product.stock_quantity > 0
    assert product.sellable is True
    assert product.name == "Yamaha C40 Nylon Natural"
    assert product.category == "Violões"
    assert product.price_brl == Decimal("599.9")
    assert is_sellable("active", 1) is True


def test_active_product_with_zero_stock_is_unavailable(store_data: StoreData) -> None:
    product = get_product(96, data=store_data)

    assert product.status == "active"
    assert product.stock_quantity == 0
    assert product.sellable is False
    assert is_sellable("active", 0) is False


def test_discontinued_product_with_positive_stock_is_unavailable(
    store_data: StoreData,
) -> None:
    product = get_product(136, data=store_data)

    assert product.status == "discontinued"
    assert product.stock_quantity > 0
    assert product.sellable is False
    assert is_sellable("discontinued", 5) is False


def test_coming_soon_product_with_positive_stock_is_unavailable(
    store_data: StoreData,
) -> None:
    product = get_product(130, data=store_data)

    assert product.status == "coming_soon"
    assert product.stock_quantity > 0
    assert product.sellable is False
    assert is_sellable("coming_soon", 5) is False


def test_maximum_price_filter_never_returns_more_expensive_products(
    store_data: StoreData,
) -> None:
    maximum = Decimal("1000")

    products = search_products(
        category="Violões",
        max_price=maximum,
        limit=100,
        data=store_data,
    )

    assert products
    assert all(product.price_brl <= maximum for product in products)


def test_available_only_excludes_non_sellable_products(store_data: StoreData) -> None:
    products = search_products(available_only=True, limit=100, data=store_data)

    assert products
    assert all(product.sellable for product in products)
    assert {96, 113, 130, 136}.isdisjoint(
        product.product_id for product in products
    )


def test_get_product_reports_ambiguous_matches_with_candidates(
    store_data: StoreData,
) -> None:
    with pytest.raises(AmbiguousProductError) as error:
        get_product("Yamaha", data=store_data)

    assert len(error.value.candidates) == 10
    assert {candidate.product_id for candidate in error.value.candidates} >= {81, 82}
    assert "ambiguous" in str(error.value).lower()


def test_duplicate_product_id_is_not_resolved_arbitrarily(store_data: StoreData) -> None:
    products = pd.concat(
        [store_data.products, store_data.products.iloc[[0]]],
        ignore_index=True,
    )
    duplicate_data = StoreData(
        products=products,
        categories=store_data.categories,
        promotions=store_data.promotions,
        customers=store_data.customers,
        orders=store_data.orders,
        order_items=store_data.order_items,
    )

    with pytest.raises(AmbiguousProductError) as error:
        get_product(81, data=duplicate_data)

    assert len(error.value.candidates) == 2


def test_unknown_numeric_product_id_fails_explicitly(store_data: StoreData) -> None:
    with pytest.raises(ProductNotFoundError, match="No product found with ID 999999"):
        get_product(999999, data=store_data)


def test_product_search_matches_accented_text_deterministically(
    store_data: StoreData,
) -> None:
    products = search_products(
        query="violao nylon",
        category="Violões",
        limit=20,
        data=store_data,
    )

    assert products
    assert all(product.category == "Violões" for product in products)


def test_product_search_observes_result_limit(store_data: StoreData) -> None:
    products = search_products(available_only=False, limit=3, data=store_data)

    assert len(products) == 3


def test_product_name_matches_take_precedence_over_conflicting_description(
    store_data: StoreData,
) -> None:
    products = search_products(
        query="Yamaha",
        available_only=False,
        limit=100,
        data=store_data,
    )

    product_ids = {product.product_id for product in products}
    assert 81 in product_ids
    assert 144 not in product_ids
