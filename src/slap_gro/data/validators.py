"""Data validation routines for SLAP-GRO datasets."""

from typing import Any, Dict, Optional
import pandas as pd

from slap_gro.data.preprocess import parse_coordinate_tuple, parse_slot_value


def validate_products(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Product.csv dataframe against contract requirements."""
    errors = []
    warnings = []

    required_cols = ["Reference", "ABCCOD", "Sector"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Clean strings
    ref_col = df["Reference"].astype(str).str.strip()
    abc_col = df["ABCCOD"].astype(str).str.strip()
    sec_col = df["Sector"].astype(str).str.strip()

    # Nulls
    if df[required_cols].isnull().any().any():
        errors.append(f"Null values found in columns: {df[required_cols].isnull().sum().to_dict()}")

    # Uniqueness
    if ref_col.duplicated().any():
        errors.append(f"Duplicate product references found: {ref_col[ref_col.duplicated()].tolist()}")

    # Value checks
    invalid_abc = set(abc_col.unique()) - {"A", "B", "C"}
    if invalid_abc:
        errors.append(f"Invalid ABCCOD values: {invalid_abc}")

    invalid_sec = set(sec_col.unique()) - {"PF"}
    if invalid_sec:
        warnings.append(f"Unexpected Sector values: {invalid_sec}")

    return {
        "valid": len(errors) == 0,
        "row_count": len(df),
        "unique_products": ref_col.nunique(),
        "errors": errors,
        "warnings": warnings,
        "abc_distribution": abc_col.value_counts().to_dict(),
    }


def validate_storage_locations(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Storage_Location.csv dataframe."""
    errors = []
    warnings = []

    required_cols = ["originalLocation", "x", "y", "z"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    loc_col = df["originalLocation"].astype(str).str.strip()

    if df[required_cols].isnull().any().any():
        errors.append("Null values found in storage location coordinates")

    if loc_col.duplicated().any():
        errors.append(f"Duplicate location IDs found: {loc_col[loc_col.duplicated()].tolist()}")

    # Coordinate ranges
    z_vals = set(df["z"].unique())
    if not z_vals.issubset({1, 2, 3, 4}):
        errors.append(f"Unexpected z coordinate levels: {z_vals - {1, 2, 3, 4}}")

    if (df["x"] < 0).any() or (df["y"] < 0).any():
        errors.append("Negative coordinates detected in x or y")

    return {
        "valid": len(errors) == 0,
        "row_count": len(df),
        "unique_locations": loc_col.nunique(),
        "z_distribution": df["z"].value_counts().sort_index().to_dict(),
        "x_range": [float(df["x"].min()), float(df["x"].max())],
        "y_range": [float(df["y"].min()), float(df["y"].max())],
        "errors": errors,
        "warnings": warnings,
    }


def validate_navigation_points(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Support_Points_Navigation.csv dataframe."""
    errors = []
    warnings = []

    required_cols = ["points_specified", "labels"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    labels_col = df["labels"].astype(str).str.strip()

    if labels_col.duplicated().any():
        errors.append(f"Duplicate navigation labels found: {labels_col[labels_col.duplicated()].tolist()}")

    # Check coordinate parseability
    parse_errors = []
    for idx, raw_pt in enumerate(df["points_specified"]):
        try:
            parse_coordinate_tuple(str(raw_pt))
        except Exception as e:
            parse_errors.append((idx, str(raw_pt), str(e)))

    if parse_errors:
        errors.append(f"Failed to parse {len(parse_errors)} navigation coordinate points")

    return {
        "valid": len(errors) == 0,
        "row_count": len(df),
        "unique_points": labels_col.nunique(),
        "errors": errors,
        "warnings": warnings,
    }


def validate_customer_orders(df: pd.DataFrame, prod_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Validate Customer_Order.csv dataframe."""
    errors = []
    warnings = []

    required_cols = [
        "codCustomer", "orderNumber", "orderToCollect", "Reference",
        "Size (US)", "quantity (units)", "creationDate", "waveNumber", "operator"
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check nulls
    null_counts = df[required_cols].isnull().sum().to_dict()
    size_nulls = null_counts.get("Size (US)", 0)
    if size_nulls > 0:
        # Known discrepancy: 16 rows have missing Size (US)
        warnings.append(f"Known issue: 'Size (US)' has {size_nulls} null values (reconciliation required)")
    
    other_nulls = {k: v for k, v in null_counts.items() if k != "Size (US)" and v > 0}
    if other_nulls:
        errors.append(f"Unexpected null values in columns: {other_nulls}")

    # Check Foreign Key with Product.csv
    if prod_df is not None and "Reference" in prod_df.columns:
        prod_refs = set(prod_df["Reference"].astype(str).str.strip())
        order_refs = set(df["Reference"].astype(str).str.strip())
        missing_refs = order_refs - prod_refs
        if missing_refs:
            errors.append(f"{len(missing_refs)} product references in orders are not in Product.csv")

    return {
        "valid": len(errors) == 0,
        "row_count": len(df),
        "unique_orders": df["orderNumber"].nunique(),
        "unique_waves": df["waveNumber"].nunique(),
        "unique_products": df["Reference"].astype(str).str.strip().nunique(),
        "null_counts": null_counts,
        "errors": errors,
        "warnings": warnings,
    }


def validate_picking_waves(
    df: pd.DataFrame,
    prod_df: Optional[pd.DataFrame] = None,
    loc_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Validate Picking_Wave.csv dataframe."""
    errors = []
    warnings = []

    required_cols = ["waveNumber", "reference", "Size (US)", "quantityToPick (units)", "locations", "operator"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check granularity: is quantityToPick always 1?
    q_vals = set(df["quantityToPick (units)"].dropna().unique())
    if q_vals != {1}:
        warnings.append(f"quantityToPick has non-unit values: {q_vals}")

    # Check FK with Product.csv
    if prod_df is not None and "Reference" in prod_df.columns:
        prod_refs = set(prod_df["Reference"].astype(str).str.strip())
        wave_refs = set(df["reference"].astype(str).str.strip())
        missing_refs = wave_refs - prod_refs
        if missing_refs:
            errors.append(f"{len(missing_refs)} product references in waves are not in Product.csv")

    # Check locations against Storage_Location.csv
    unmatched_locs = []
    if loc_df is not None and "originalLocation" in loc_df.columns:
        valid_locs = set(loc_df["originalLocation"].astype(str).str.strip())
        wave_locs = set(df["locations"].astype(str).str.strip())
        unmatched_locs = sorted(list(wave_locs - valid_locs))
        if unmatched_locs:
            warnings.append(
                f"Known issue: {len(unmatched_locs)} picking locations are not in Storage_Location.csv "
                f"(e.g. cross-dock/reception/aisle 25-26): {unmatched_locs[:5]}..."
            )

    return {
        "valid": len(errors) == 0,
        "row_count": len(df),
        "unique_waves": df["waveNumber"].nunique(),
        "unique_products": df["reference"].astype(str).str.strip().nunique(),
        "unique_locations": df["locations"].astype(str).str.strip().nunique(),
        "unmatched_locations_count": len(unmatched_locs),
        "errors": errors,
        "warnings": warnings,
    }


def validate_storage_matrix(
    df: pd.DataFrame,
    policy_name: str,
    loc_df: Optional[pd.DataFrame] = None,
    prod_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Validate a storage policy matrix dataframe (18 slots)."""
    errors = []
    warnings = []

    # Identify location column
    loc_col = "originalLocation" if "originalLocation" in df.columns else ("Location" if "Location" in df.columns else None)
    if not loc_col:
        errors.append(f"Missing location column in {policy_name}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Find 18 slot columns
    non_slot = {loc_col, "ABCCOD", "XYZCOD"}
    slot_cols = [c for c in df.columns if c not in non_slot]
    if len(slot_cols) != 18:
        errors.append(f"Expected 18 slot columns in {policy_name}, found {len(slot_cols)}")

    # Check parseability of slots
    skus_in_slots = set()
    slot_parse_errors = 0
    for col in slot_cols:
        for val in df[col].dropna():
            parsed = parse_slot_value(val)
            if parsed:
                skus_in_slots.add(parsed[0])
            else:
                slot_parse_errors += 1

    if slot_parse_errors > 0:
        warnings.append(f"{slot_parse_errors} empty or invalid slot values in {policy_name}")

    # Check locations vs Storage_Location
    extra_locations = []
    if loc_df is not None and "originalLocation" in loc_df.columns:
        valid_locs = set(loc_df["originalLocation"].astype(str).str.strip())
        matrix_locs = set(df[loc_col].astype(str).str.strip())
        extra_locations = sorted(list(matrix_locs - valid_locs))
        if extra_locations:
            warnings.append(
                f"Known discrepancy: {len(extra_locations)} locations in {policy_name} "
                f"are not in Storage_Location.csv"
            )

    # Check SKUs vs Product.csv
    extra_skus = []
    if prod_df is not None and "Reference" in prod_df.columns:
        prod_skus = set(prod_df["Reference"].astype(str).str.strip())
        extra_skus = sorted(list(skus_in_slots - prod_skus))
        if extra_skus:
            warnings.append(
                f"Known discrepancy: {len(extra_skus)} SKUs in {policy_name} slots "
                f"are not in Product.csv"
            )

    return {
        "valid": len(errors) == 0,
        "policy": policy_name,
        "row_count": len(df),
        "slot_columns_count": len(slot_cols),
        "unique_skus_in_slots": len(skus_in_slots),
        "extra_locations_count": len(extra_locations),
        "extra_skus_count": len(extra_skus),
        "errors": errors,
        "warnings": warnings,
    }


validate_orders = validate_customer_orders
