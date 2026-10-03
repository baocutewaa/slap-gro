"""Unified data loader module for SLAP-GRO datasets."""

from pathlib import Path
from typing import Any, List, Literal, Optional, Union
import pandas as pd


from slap_gro.config.loader import get_project_root
from slap_gro.data.preprocess import (
    clean_string_column,
    normalize_customer_orders,
    normalize_navigation_points,
    normalize_picking_waves,
    normalize_products,
    normalize_storage_locations,
    normalize_storage_matrix,
)
from slap_gro.data.schemas import (
    CustomerOrderRecord,
    NavigationPointRecord,
    PickingWaveRecord,
    ProductRecord,
    StorageLocationRecord,
    StorageUnitRecord,
)


def _resolve_path(file_path: Path | str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _read_csv_robust(file_path: Path | str, sep: Optional[str] = None) -> pd.DataFrame:
    """Read CSV handling utf-8-sig encoding and auto-detecting delimiter if needed."""
    path = _resolve_path(file_path)
    if sep is None:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_products(
    file_path: str = "data/raw/Product.csv",
    as_records: bool = False,
    as_catalog: bool = False,
) -> Any:
    """
    Load and normalize Product.csv.

    Columns:
        Reference: str (SKU ID)
        ABCCOD: str ('A', 'B', 'C')
        Sector: str ('PF')
    """
    df = _read_csv_robust(file_path, sep=";")
    df = normalize_products(df)
    if as_catalog:
        from slap_gro.data.product import ProductCatalog
        return ProductCatalog.from_dataframe(df)
    if as_records:
        return ProductRecord.from_dataframe(df)
    return df



def load_storage_locations(
    file_path: str = "data/raw/Storage_Location.csv",
    as_records: bool = False,
) -> Union[pd.DataFrame, List[StorageLocationRecord]]:
    """
    Load and normalize Storage_Location.csv.

    Columns:
        originalLocation: str (e.g., 'A-14-11')
        position: str ('x, y, z')
        x: float
        y: float
        z: int (1, 2, 3, 4)
        block: str (optional parsed)
        aisle: int (optional parsed)
        bay: int (optional parsed)
    """
    df = _read_csv_robust(file_path, sep=",")
    df = normalize_storage_locations(df)
    if as_records:
        return StorageLocationRecord.from_dataframe(df)
    return df


def load_navigation_points(
    file_path: str = "data/raw/Support_Points_Navigation.csv",
    parse_coordinates: bool = False,
    as_records: bool = False,
) -> Union[pd.DataFrame, List[NavigationPointRecord]]:
    """
    Load and normalize Support_Points_Navigation.csv.

    Columns:
        points_specified: str '(x, y, z)'
        labels: str (e.g. 'LC-01')
        x, y, z: float parsed coordinates (if parse_coordinates=True)
    """
    df = _read_csv_robust(file_path, sep=";")
    df = normalize_navigation_points(df, parse_coordinates=parse_coordinates)
    if as_records:
        return NavigationPointRecord.from_dataframe(df)
    return df



def load_orders(
    file_path: str = "data/raw/Customer_Order.csv",
    impute_missing_size: bool = True,
    parse_dates: bool = True,
    as_records: bool = False,
) -> Union[pd.DataFrame, List[CustomerOrderRecord]]:
    """
    Load and normalize Customer_Order.csv.

    Columns:
        codCustomer, orderNumber, orderToCollect, Reference,
        Size (US), quantity (units), creationDate, waveNumber, operator
    """
    df = _read_csv_robust(file_path, sep=";")
    df = normalize_customer_orders(
        df,
        impute_missing_size=impute_missing_size,
        parse_dates=parse_dates,
    )
    if as_records:
        return CustomerOrderRecord.from_dataframe(df)
    return df


load_customer_orders = load_orders


def load_picking_waves(
    file_path: str = "data/raw/Picking_Wave.csv",
    as_records: bool = False,
) -> Union[pd.DataFrame, List[PickingWaveRecord]]:
    """
    Load and normalize Picking_Wave.csv.

    Columns:
        waveNumber, reference, Size (US), quantityToPick (units), locations, operator
    """
    df = _read_csv_robust(file_path, sep=";")
    df = normalize_picking_waves(df)
    if as_records:
        return PickingWaveRecord.from_dataframe(df)
    return df


def load_storage_matrix(
    policy: Literal["random", "class_based", "dedicated", "hybrid"] = "random",
    file_path: Optional[str] = None,
    as_records: bool = False,
) -> Union[pd.DataFrame, List[StorageUnitRecord]]:
    """
    Load and normalize storage matrix for the specified policy.

    Available policies:
        - 'random' -> Random_Storage.csv (delimiter: ',')
        - 'class_based' -> Class_Based_Storage.csv (delimiter: ';')
        - 'dedicated' -> Dedicated_Storage.csv (delimiter: ';')
        - 'hybrid' -> Hybrid_Storage.csv (delimiter: ';')
    """
    policy_files = {
        "random": "data/raw/Random_Storage.csv",
        "class_based": "data/raw/Class_Based_Storage.csv",
        "dedicated": "data/raw/Dedicated_Storage.csv",
        "hybrid": "data/raw/Hybrid_Storage.csv",
    }

    key = policy.lower().replace("-", "_")
    if key not in policy_files:
        raise ValueError(f"Unknown storage policy: {policy}. Must be one of {list(policy_files.keys())}")

    target_path = file_path if file_path else policy_files[key]
    df = _read_csv_robust(target_path)
    df = normalize_storage_matrix(df, policy=key)

    if as_records:
        return StorageUnitRecord.from_dataframe(df, policy=key)
    return df
