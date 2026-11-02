"""Load and validate the store's CSV datasets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR


REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "products.csv": (
        "product_id",
        "price_brl",
        "name",
        "category_id",
        "description",
        "stock_quantity",
        "status",
        "specs",
        "created_at",
    ),
    "categories.csv": ("category_id", "name", "description"),
    "promotions.csv": (
        "promotion_id",
        "product_id",
        "discount_percent",
        "description",
        "is_active",
    ),
    "customers.csv": ("customer_id", "name", "phone", "email", "city"),
    "orders.csv": (
        "order_id",
        "customer_id",
        "order_date",
        "status",
        "total_brl",
        "payment_method",
        "tracking_code",
        "estimated_delivery",
        "notes",
    ),
    "order_items.csv": ("order_id", "quantity", "product_id"),
}

INTEGER_COLUMNS: dict[str, tuple[str, ...]] = {
    "products.csv": ("product_id", "category_id", "stock_quantity"),
    "categories.csv": ("category_id",),
    "promotions.csv": ("promotion_id", "product_id"),
    "customers.csv": ("customer_id",),
    "orders.csv": ("order_id", "customer_id"),
    "order_items.csv": ("order_id", "quantity", "product_id"),
}

DECIMAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "products.csv": ("price_brl",),
    "promotions.csv": ("discount_percent",),
    "orders.csv": ("total_brl",),
}


class DataLoadError(ValueError):
    """Raised when a required dataset is missing or invalid."""


def _decimal_value(value: Any, *, filename: str, column: str, row_number: int) -> Decimal:
    if pd.isna(value):
        raise DataLoadError(
            f"Invalid dataset '{filename}': {column} is empty at data row {row_number}."
        )
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise DataLoadError(
            f"Invalid dataset '{filename}': {column} must contain numeric values "
            f"(invalid value at data row {row_number})."
        ) from None
    if not result.is_finite():
        raise DataLoadError(
            f"Invalid dataset '{filename}': {column} must contain finite numeric values "
            f"(invalid value at data row {row_number})."
        )
    return result


def _parse_active_indicator(value: Any, *, row_number: int) -> bool:
    """Parse the dataset's explicit 0/1 active indicator without truthiness guesses."""
    if pd.isna(value):
        raise DataLoadError(
            "Invalid dataset 'promotions.csv': is_active is empty "
            f"at data row {row_number}."
        )
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise DataLoadError(
        "Invalid dataset 'promotions.csv': is_active must be 0 or 1 "
        f"at data row {row_number}."
    )


@dataclass(slots=True)
class StoreData:
    """In-memory Pandas tables loaded from the six provided CSV files."""

    products: pd.DataFrame
    categories: pd.DataFrame
    promotions: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame

    @classmethod
    def load(cls, data_dir: str | Path = DATA_DIR) -> StoreData:
        """Load every required CSV and fail with a clear English validation error."""
        root = Path(data_dir)
        tables: dict[str, pd.DataFrame] = {}

        for filename, required_columns in REQUIRED_COLUMNS.items():
            path = root / filename
            if not path.is_file():
                raise DataLoadError(f"Required dataset file not found: {path}")

            # Keep currency and percentage source values textual until Decimal parsing.
            dtype: dict[str, str] = {}
            if filename == "products.csv":
                dtype["price_brl"] = "string"
            elif filename == "promotions.csv":
                dtype["discount_percent"] = "string"

            try:
                frame = pd.read_csv(path, dtype=dtype or None)
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
                OSError,
                ValueError,
            ) as exc:
                raise DataLoadError(
                    f"Could not read required dataset '{path}': {exc}"
                ) from exc

            missing_columns = [column for column in required_columns if column not in frame.columns]
            if missing_columns:
                missing = ", ".join(missing_columns)
                raise DataLoadError(
                    f"Invalid dataset '{filename}': missing required columns: {missing}."
                )
            if frame.empty:
                raise DataLoadError(f"Invalid dataset '{filename}': it contains no data rows.")

            for column in INTEGER_COLUMNS[filename]:
                for row_number, value in enumerate(frame[column], start=2):
                    number = _decimal_value(
                        value, filename=filename, column=column, row_number=row_number - 1
                    )
                    if number != number.to_integral_value():
                        raise DataLoadError(
                            f"Invalid dataset '{filename}': {column} must contain whole numbers "
                            f"(invalid value at CSV line {row_number})."
                        )

            for column in DECIMAL_COLUMNS.get(filename, ()):
                for row_number, value in enumerate(frame[column], start=1):
                    amount = _decimal_value(
                        value, filename=filename, column=column, row_number=row_number
                    )
                    if filename == "promotions.csv" and not Decimal("0") <= amount <= Decimal("100"):
                        raise DataLoadError(
                            "Invalid dataset 'promotions.csv': discount_percent must be "
                            f"between 0 and 100 (invalid value at data row {row_number})."
                        )

            if filename == "promotions.csv":
                frame["is_active"] = [
                    _parse_active_indicator(value, row_number=row_number)
                    for row_number, value in enumerate(frame["is_active"], start=1)
                ]

            tables[filename] = frame

        return cls(
            products=tables["products.csv"],
            categories=tables["categories.csv"],
            promotions=tables["promotions.csv"],
            customers=tables["customers.csv"],
            orders=tables["orders.csv"],
            order_items=tables["order_items.csv"],
        )
