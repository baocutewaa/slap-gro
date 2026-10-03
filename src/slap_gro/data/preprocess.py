"""Preprocessing and parsing utilities for SLAP-GRO datasets."""

import re
from typing import Optional, Tuple
import pandas as pd


_POINT_REGEX = re.compile(
    r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
)


def parse_slot_value(cell_value: str | float | None) -> Optional[Tuple[str, float]]:
    """
    Parse a storage slot cell formatted as 'product_code;quantity'.

    Example:
        '8551FLX;15.0' -> ('8551FLX', 15.0)
        'TQBVRI;7' -> ('TQBVRI', 7.0)
        None / NaN / '' -> None
    """
    if cell_value is None or pd.isna(cell_value):
        return None
    val_str = str(cell_value).strip().strip('"').strip("'")
    if not val_str:
        return None

    if ";" in val_str:
        parts = val_str.split(";")
        sku = parts[0].strip()
        try:
            qty = float(parts[1].strip())
        except (ValueError, IndexError):
            qty = 1.0
        return sku, qty
    else:
        # Just SKU without quantity
        return val_str, 1.0


def parse_coordinate_tuple(tuple_str: str | None) -> Tuple[float, float, float]:
    """
    Parse string representation of 3D point tuple.

    Example:
        '(66.0, -29.0, 1.0)' -> (66.0, -29.0, 1.0)
    """
    if not tuple_str or pd.isna(tuple_str):
        raise ValueError("Cannot parse empty coordinate string")
    clean = str(tuple_str).strip()
    match = _POINT_REGEX.match(clean)
    if not match:
        raise ValueError(f"Invalid coordinate tuple format: {tuple_str}")
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def parse_position_string(pos_str: str | None) -> Tuple[float, float, float]:
    """
    Parse comma-separated coordinate position string.

    Example:
        '368, 0, 1' -> (368.0, 0.0, 1.0)
    """
    if not pos_str or pd.isna(pos_str):
        raise ValueError("Cannot parse empty position string")
    parts = [p.strip().strip('"') for p in str(pos_str).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 coordinate components, got {len(parts)} in '{pos_str}'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def clean_string_column(series: pd.Series) -> pd.Series:
    """Strip whitespace and quotation marks from a pandas series of strings."""
    return series.astype(str).str.strip().str.strip('"').str.strip("'")


def normalize_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Product DataFrame:
    - Trims string columns ('Reference', 'ABCCOD', 'Sector').
    - Ensures ABCCOD and Sector are upper-case.
    """
    clean_df = df.copy()
    for col in ["Reference", "ABCCOD", "Sector"]:
        if col in clean_df.columns:
            clean_df[col] = clean_string_column(clean_df[col])
    if "ABCCOD" in clean_df.columns:
        clean_df["ABCCOD"] = clean_df["ABCCOD"].str.upper()
    if "Sector" in clean_df.columns:
        clean_df["Sector"] = clean_df["Sector"].str.upper()
    return clean_df


def normalize_storage_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Storage_Location DataFrame:
    - Trims 'originalLocation' and 'position'.
    - Casts 'x' and 'y' to float64, 'z' to int64.
    - Adds parsed 'block', 'aisle', and 'bay' columns if not already present.
    """
    clean_df = df.copy()
    if "originalLocation" in clean_df.columns:
        clean_df["originalLocation"] = clean_string_column(clean_df["originalLocation"])
    if "position" in clean_df.columns:
        clean_df["position"] = clean_string_column(clean_df["position"])

    if "x" in clean_df.columns:
        clean_df["x"] = pd.to_numeric(clean_df["x"], errors="coerce").astype(float)
    if "y" in clean_df.columns:
        clean_df["y"] = pd.to_numeric(clean_df["y"], errors="coerce").astype(float)
    if "z" in clean_df.columns:
        clean_df["z"] = pd.to_numeric(clean_df["z"], errors="coerce").astype(int)

    # Derive block, aisle, bay if location format is Block-Aisle-Bay (e.g. A-14-11)
    if "originalLocation" in clean_df.columns and "block" not in clean_df.columns:
        parts = clean_df["originalLocation"].str.split("-", expand=True)
        if parts.shape[1] >= 1:
            clean_df["block"] = parts[0].str.strip()
        if parts.shape[1] >= 2:
            clean_df["aisle"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
        if parts.shape[1] >= 3:
            clean_df["bay"] = pd.to_numeric(parts[2], errors="coerce").astype("Int64")

    return clean_df


def normalize_navigation_points(
    df: pd.DataFrame,
    parse_coordinates: bool = False,
) -> pd.DataFrame:
    """
    Normalize Support_Points_Navigation DataFrame:
    - Trims 'labels' and 'points_specified'.
    - Optionally parses tuple coordinates into numeric 'x', 'y', 'z' float columns.
    """
    clean_df = df.copy()
    if "labels" in clean_df.columns:
        clean_df["labels"] = clean_string_column(clean_df["labels"])
    if "points_specified" in clean_df.columns:
        clean_df["points_specified"] = clean_string_column(clean_df["points_specified"])
        if parse_coordinates:
            parsed = clean_df["points_specified"].apply(parse_coordinate_tuple)
            clean_df["x"] = parsed.apply(lambda p: p[0]).astype(float)
            clean_df["y"] = parsed.apply(lambda p: p[1]).astype(float)
            clean_df["z"] = parsed.apply(lambda p: p[2]).astype(float)
    return clean_df


def normalize_customer_orders(
    df: pd.DataFrame,
    impute_missing_size: bool = True,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Normalize Customer_Order DataFrame:
    - Trims strings: 'codCustomer', 'Reference', 'operator'.
    - Casts integer identifiers: 'orderNumber', 'orderToCollect', 'waveNumber', 'quantity (units)'.
    - Casts 'Size (US)' to float and optionally imputes missing values with median size.
    - Parses 'creationDate' to datetime64.
    """
    clean_df = df.copy()
    for col in ["codCustomer", "Reference", "operator"]:
        if col in clean_df.columns:
            clean_df[col] = clean_string_column(clean_df[col])

    for col in ["orderNumber", "orderToCollect", "waveNumber", "quantity (units)"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").astype(int)

    if "Size (US)" in clean_df.columns:
        clean_df["Size (US)"] = pd.to_numeric(clean_df["Size (US)"], errors="coerce")
        if impute_missing_size:
            median_size = clean_df["Size (US)"].median()
            clean_df["Size (US)"] = clean_df["Size (US)"].fillna(median_size)

    if parse_dates and "creationDate" in clean_df.columns:
        clean_df["creationDate"] = pd.to_datetime(
            clean_df["creationDate"], format="%d/%m/%Y %H:%M", errors="coerce"
        )

    return clean_df


normalize_orders = normalize_customer_orders


def normalize_picking_waves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Picking_Wave DataFrame:
    - Trims strings: 'reference', 'locations', 'operator'.
    - Casts integers: 'waveNumber', 'quantityToPick (units)'.
    - Casts float: 'Size (US)'.
    """
    clean_df = df.copy()
    for col in ["reference", "locations", "operator"]:
        if col in clean_df.columns:
            clean_df[col] = clean_string_column(clean_df[col])

    for col in ["waveNumber", "quantityToPick (units)"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").astype(int)

    if "Size (US)" in clean_df.columns:
        clean_df["Size (US)"] = pd.to_numeric(clean_df["Size (US)"], errors="coerce").astype(float)

    return clean_df


def normalize_storage_matrix(
    df: pd.DataFrame,
    policy: str = "random",
) -> pd.DataFrame:
    """
    Normalize Storage Matrix DataFrame (random, class_based, dedicated, hybrid):
    - Trims location column ('originalLocation' or 'Location').
    - Standardizes / trims any categorical columns ('ABCCOD', 'XYZCOD').
    - Trims all slot values.
    """
    clean_df = df.copy()
    loc_col = "originalLocation" if "originalLocation" in clean_df.columns else (
        "Location" if "Location" in clean_df.columns else None
    )
    if loc_col:
        clean_df[loc_col] = clean_string_column(clean_df[loc_col])

    for cat_col in ["ABCCOD", "XYZCOD"]:
        if cat_col in clean_df.columns:
            clean_df[cat_col] = clean_string_column(clean_df[cat_col]).str.upper()

    # Identify 18 slot columns
    non_slot = {loc_col, "ABCCOD", "XYZCOD"}
    slot_cols = [c for c in clean_df.columns if c not in non_slot]
    for c in slot_cols:
        clean_df[c] = clean_string_column(clean_df[c])

    return clean_df

