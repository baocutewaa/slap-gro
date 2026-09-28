"""Unified data loader module for SLAP-GRO datasets."""

from pathlib import Path
from typing import Literal, Optional
import pandas as pd

from slap_gro.config.loader import get_project_root
from slap_gro.data.preprocess import clean_string_column


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


def load_products(file_path: str = "data/raw/Product.csv") -> pd.DataFrame:
    """
    Load Product.csv.

    Columns:
        Reference: str (SKU ID)
        ABCCOD: str ('A', 'B', 'C')
        Sector: str ('PF')
    """
    df = _read_csv_robust(file_path, sep=";")
    for col in ["Reference", "ABCCOD", "Sector"]:
        if col in df.columns:
            df[col] = clean_string_column(df[col])
    return df


def load_storage_locations(file_path: str = "data/raw/Storage_Location.csv") -> pd.DataFrame:
    """
    Load Storage_Location.csv.

    Columns:
        originalLocation: str (e.g., 'A-14-11')
        position: str ('x, y, z')
        x: float
        y: float
        z: int (1, 2, 3, 4)
    """
    df = _read_csv_robust(file_path, sep=",")
    df["originalLocation"] = clean_string_column(df["originalLocation"])
    df["position"] = clean_string_column(df["position"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce").astype(int)
    return df


def load_navigation_points(file_path: str = "data/raw/Support_Points_Navigation.csv") -> pd.DataFrame:
    """
    Load Support_Points_Navigation.csv.

    Columns:
        points_specified: str '(x, y, z)'
        labels: str (e.g. 'LC-01')
    """
    df = _read_csv_robust(file_path, sep=";")
    df["points_specified"] = clean_string_column(df["points_specified"])
    df["labels"] = clean_string_column(df["labels"])
    return df


def load_customer_orders(
    file_path: str = "data/raw/Customer_Order.csv",
    impute_missing_size: bool = True
) -> pd.DataFrame:
    """
    Load Customer_Order.csv.

    Columns:
        codCustomer, orderNumber, orderToCollect, Reference,
        Size (US), quantity (units), creationDate, waveNumber, operator
    """
    df = _read_csv_robust(file_path, sep=";")
    for col in ["codCustomer", "Reference", "operator"]:
        if col in df.columns:
            df[col] = clean_string_column(df[col])

    if impute_missing_size and "Size (US)" in df.columns:
        # 16 records have missing size; impute using product median or global median
        median_size = df["Size (US)"].median()
        df["Size (US)"] = df["Size (US)"].fillna(median_size)

    return df


def load_picking_waves(file_path: str = "data/raw/Picking_Wave.csv") -> pd.DataFrame:
    """
    Load Picking_Wave.csv.

    Columns:
        waveNumber, reference, Size (US), quantityToPick (units), locations, operator
    """
    df = _read_csv_robust(file_path, sep=";")
    for col in ["reference", "locations", "operator"]:
        if col in df.columns:
            df[col] = clean_string_column(df[col])
    return df


def load_storage_matrix(
    policy: Literal["random", "class_based", "dedicated", "hybrid"] = "random",
    file_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load storage matrix for the specified policy.

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

    loc_col = "originalLocation" if "originalLocation" in df.columns else ("Location" if "Location" in df.columns else None)
    if loc_col:
        df[loc_col] = clean_string_column(df[loc_col])

    return df
