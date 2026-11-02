from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.config import DATA_DIR
from src.data_loader import DataLoadError, REQUIRED_COLUMNS, StoreData


def _copy_datasets(destination: Path) -> None:
    for filename in REQUIRED_COLUMNS:
        shutil.copyfile(DATA_DIR / filename, destination / filename)


def test_all_expected_datasets_load_successfully() -> None:
    data = StoreData.load()

    for table_name in (
        "products",
        "categories",
        "promotions",
        "customers",
        "orders",
        "order_items",
    ):
        table = getattr(data, table_name)
        assert not table.empty
    assert data.promotions["is_active"].dtype == bool


def test_missing_dataset_fails_with_clear_error(tmp_path: Path) -> None:
    _copy_datasets(tmp_path)
    (tmp_path / "customers.csv").unlink()

    with pytest.raises(DataLoadError, match="Required dataset file not found:.*customers.csv"):
        StoreData.load(tmp_path)


def test_missing_required_column_fails_with_clear_error(tmp_path: Path) -> None:
    _copy_datasets(tmp_path)
    products_path = tmp_path / "products.csv"
    products = pd.read_csv(products_path).drop(columns=["price_brl"])
    products.to_csv(products_path, index=False)

    with pytest.raises(DataLoadError, match="missing required columns: price_brl"):
        StoreData.load(tmp_path)


def test_malformed_active_indicator_fails_with_clear_error(tmp_path: Path) -> None:
    _copy_datasets(tmp_path)
    promotions_path = tmp_path / "promotions.csv"
    promotions = pd.read_csv(promotions_path)
    promotions["is_active"] = promotions["is_active"].astype(str)
    promotions.loc[0, "is_active"] = "sometimes"
    promotions.to_csv(promotions_path, index=False)

    with pytest.raises(DataLoadError, match="is_active must be 0 or 1"):
        StoreData.load(tmp_path)
