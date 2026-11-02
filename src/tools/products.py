"""Deterministic search and lookup functions for the product catalog."""

from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from src.data_loader import StoreData
from src.models import ProductRecord


class ProductNotFoundError(LookupError):
    """Raised when a query does not identify any catalog product."""


class AmbiguousProductError(LookupError):
    """Raised when a partial product query matches multiple catalog products."""

    def __init__(self, query: str, candidates: list[ProductRecord]) -> None:
        self.query = query
        self.candidates = candidates
        choices = "; ".join(f"{item.product_id}: {item.name}" for item in candidates)
        super().__init__(f"Product query '{query}' is ambiguous. Matching products: {choices}.")


def is_sellable(status: str, stock_quantity: int) -> bool:
    """Apply the catalog sellability invariant without LLM or caller inference."""
    return status == "active" and stock_quantity > 0


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _as_decimal(value: Any, *, field_name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        raise ValueError(f"{field_name} must be a valid decimal number.") from None
    if not result.is_finite():
        raise ValueError(f"{field_name} must be a finite decimal number.")
    return result


def _category_names(data: StoreData) -> dict[int, str]:
    return {
        int(row["category_id"]): str(row["name"])
        for _, row in data.categories.iterrows()
    }


def _product_from_row(row: pd.Series, categories: dict[int, str]) -> ProductRecord:
    product_id = int(row["product_id"])
    category_id = int(row["category_id"])
    try:
        category = categories[category_id]
    except KeyError:
        raise ValueError(
            f"Product {product_id} refers to unknown category ID {category_id}."
        ) from None
    stock_quantity = int(row["stock_quantity"])
    status = str(row["status"])
    return ProductRecord(
        product_id=product_id,
        name=str(row["name"]),
        category=category,
        price_brl=_as_decimal(row["price_brl"], field_name="price_brl"),
        stock_quantity=stock_quantity,
        status=status,
        sellable=is_sellable(status, stock_quantity),
    )


def _all_products(data: StoreData) -> list[ProductRecord]:
    categories = _category_names(data)
    return [_product_from_row(row, categories) for _, row in data.products.iterrows()]


def _searchable_text(row: pd.Series, category: str) -> str:
    values = [row.get("name"), row.get("description"), row.get("specs"), category]
    return " ".join(_normalize_text(value) for value in values)


def search_products(
    query: str | None = None,
    *,
    category: str | int | None = None,
    max_price: Decimal | str | int | float | None = None,
    available_only: bool = True,
    limit: int = 5,
    data: StoreData | None = None,
) -> list[ProductRecord]:
    """Filter the provided catalog deterministically and return structured results."""
    store = data if data is not None else StoreData.load()
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    price_ceiling = None if max_price is None else _as_decimal(max_price, field_name="max_price")
    if price_ceiling is not None and price_ceiling < 0:
        raise ValueError("max_price must be zero or greater.")

    categories = _category_names(store)
    category_filter = None
    category_id_filter = None
    if category is not None:
        if isinstance(category, int):
            category_id_filter = category
        else:
            category_text = str(category).strip()
            if category_text.isdigit():
                category_id_filter = int(category_text)
            else:
                category_filter = _normalize_text(category_text)

    query_tokens = _normalize_text(query).split() if query and query.strip() else []
    name_matches: list[ProductRecord] = []
    detail_matches: list[ProductRecord] = []
    unfiltered_results: list[ProductRecord] = []
    for _, row in store.products.iterrows():
        record = _product_from_row(row, categories)
        if available_only and not record.sellable:
            continue
        if price_ceiling is not None and record.price_brl > price_ceiling:
            continue
        if category_id_filter is not None and int(row["category_id"]) != category_id_filter:
            continue
        if category_filter is not None:
            normalized_category = _normalize_text(record.category)
            if category_filter not in normalized_category:
                continue
        if not query_tokens:
            unfiltered_results.append(record)
        elif all(token in _normalize_text(row["name"]) for token in query_tokens):
            name_matches.append(record)
        elif all(
            token in _searchable_text(row, record.category)
            for token in query_tokens
        ):
            detail_matches.append(record)

    # Prefer product-name evidence when available; use descriptions/specs as fallback.
    if query_tokens:
        results = name_matches or detail_matches
    else:
        results = unfiltered_results
    return results[:limit]


def get_product(
    product_query: str | int,
    *,
    data: StoreData | None = None,
) -> ProductRecord:
    """Return one product, raising explicitly for missing or ambiguous queries."""
    store = data if data is not None else StoreData.load()
    query = str(product_query).strip()
    if not query:
        raise ProductNotFoundError("Product query cannot be empty.")

    records = _all_products(store)
    if query.isdigit():
        product_id = int(query)
        exact_id = [record for record in records if record.product_id == product_id]
        if exact_id:
            if len(exact_id) > 1:
                raise AmbiguousProductError(query, exact_id)
            return exact_id[0]
        raise ProductNotFoundError(f"No product found with ID {product_id}.")

    normalized_query = _normalize_text(query)
    exact_matches = [record for record in records if _normalize_text(record.name) == normalized_query]
    if exact_matches:
        matches = exact_matches
    else:
        categories = _category_names(store)
        matches = [
            _product_from_row(row, categories)
            for _, row in store.products.iterrows()
            if normalized_query in _normalize_text(row["name"])
        ]

    if not matches:
        raise ProductNotFoundError(f"No product matched query '{query}'.")
    if len(matches) > 1:
        raise AmbiguousProductError(query, matches)
    return matches[0]
