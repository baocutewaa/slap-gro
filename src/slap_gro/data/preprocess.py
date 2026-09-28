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
