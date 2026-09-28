"""Tests for SLAP-GRO Phase 0: Data layer, loaders, schemas, and validators."""

import pytest
import pandas as pd

from slap_gro.config.loader import load_data_contract, load_paper_config, load_real_config
from slap_gro.data.loaders import (
    load_customer_orders,
    load_navigation_points,
    load_picking_waves,
    load_products,
    load_storage_locations,
    load_storage_matrix,
)
from slap_gro.data.preprocess import (
    parse_coordinate_tuple,
    parse_position_string,
    parse_slot_value,
)
from slap_gro.data.validators import (
    validate_customer_orders,
    validate_navigation_points,
    validate_picking_waves,
    validate_products,
    validate_storage_locations,
    validate_storage_matrix,
)


# ==============================================================================
# 1. Config & Contract Tests
# ==============================================================================

def test_data_contract_loading():
    """Verify data_contract.yaml loads and contains paper and real modes."""
    contract = load_data_contract()
    assert "paper_mode" in contract
    assert "real_mode" in contract
    assert "datasets" in contract
    assert "reconciliation" in contract

    paper = contract["paper_mode"]
    assert paper["n_storage_units"] == 847
    assert paper["slots_per_unit"] == 18
    assert paper["total_storage_slots"] == 15246
    assert paper["n_products"] == 2000
    assert paper["n_waves"] == 50
    assert paper["items_per_wave"] == 25

    real = contract["real_mode"]
    assert real["expected_locations_count"] == 2292
    assert real["expected_products_count"] == 208
    assert real["expected_waves_count"] == 9707


def test_mode_configs_loading():
    """Verify individual mode config YAML files load correctly."""
    paper_cfg = load_paper_config()
    assert paper_cfg["mode"] == "paper_replication"
    assert paper_cfg["gro"]["population_size"] == 1000
    assert paper_cfg["gro"]["crossover_rate"] == 0.90

    real_cfg = load_real_config()
    assert real_cfg["mode"] == "real_data_replication"
    assert real_cfg["gro"]["mutation_rate"] == 0.08


# ==============================================================================
# 2. Parsing & Preprocessing Unit Tests
# ==============================================================================

def test_parse_slot_value():
    """Test slot parsing with various input formats."""
    assert parse_slot_value("8551FLX;15.0") == ("8551FLX", 15.0)
    assert parse_slot_value("TQBVRI;7") == ("TQBVRI", 7.0)
    assert parse_slot_value("ONLYSKU") == ("ONLYSKU", 1.0)
    assert parse_slot_value("") is None
    assert parse_slot_value(None) is None


def test_parse_coordinate_tuple():
    """Test 3D coordinate tuple parsing."""
    coords = parse_coordinate_tuple("(66.0, -29.0, 1.0)")
    assert coords == (66.0, -29.0, 1.0)

    coords_neg = parse_coordinate_tuple("(-12.5, 45.0, 3.0)")
    assert coords_neg == (-12.5, 45.0, 3.0)

    with pytest.raises(ValueError):
        parse_coordinate_tuple("invalid")


def test_parse_position_string():
    """Test comma-separated position string parsing."""
    coords = parse_position_string("368, 0, 1")
    assert coords == (368.0, 0.0, 1.0)


# ==============================================================================
# 3. Data Loader & Schema Conformity Tests
# ==============================================================================

def test_load_products():
    """Verify loading and schema conformity for Product.csv."""
    df = load_products()
    assert len(df) == 208
    assert set(df.columns) == {"Reference", "ABCCOD", "Sector"}
    assert set(df["ABCCOD"].unique()) == {"A", "B", "C"}
    assert set(df["Sector"].unique()) == {"PF"}
    assert df["Reference"].nunique() == 208

    val = validate_products(df)
    assert val["valid"] is True
    assert val["errors"] == []


def test_load_storage_locations():
    """Verify loading and schema conformity for Storage_Location.csv."""
    df = load_storage_locations()
    assert len(df) == 2292
    assert "originalLocation" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns
    assert df["originalLocation"].nunique() == 2292
    assert set(df["z"].unique()) == {1, 2, 3, 4}

    val = validate_storage_locations(df)
    assert val["valid"] is True
    assert val["errors"] == []


def test_load_navigation_points():
    """Verify loading and schema conformity for Support_Points_Navigation.csv."""
    df = load_navigation_points()
    assert len(df) == 44
    assert set(df.columns) == {"points_specified", "labels"}
    assert df["labels"].nunique() == 44

    val = validate_navigation_points(df)
    assert val["valid"] is True
    assert val["errors"] == []


def test_load_customer_orders():
    """Verify loading Customer_Order.csv and missing size handling."""
    # Test with imputation
    df_imputed = load_customer_orders(impute_missing_size=True)
    assert len(df_imputed) == 122370
    assert df_imputed["Size (US)"].isnull().sum() == 0

    # Test without imputation
    df_raw = load_customer_orders(impute_missing_size=False)
    assert df_raw["Size (US)"].isnull().sum() == 16

    val = validate_customer_orders(df_imputed)
    assert val["valid"] is True


def test_load_picking_waves():
    """Verify loading Picking_Wave.csv and unit-pick granularity."""
    df = load_picking_waves()
    assert len(df) == 215192
    assert df["waveNumber"].nunique() == 9707
    # All unit picks
    assert (df["quantityToPick (units)"] == 1).all()

    val = validate_picking_waves(df)
    assert val["valid"] is True


@pytest.mark.parametrize("policy", ["random", "class_based", "dedicated", "hybrid"])
def test_load_storage_matrices(policy):
    """Verify all 4 storage policies load with 18 slot columns."""
    df = load_storage_matrix(policy)
    loc_col = "originalLocation" if "originalLocation" in df.columns else "Location"
    assert loc_col in df.columns

    non_slot = {loc_col, "ABCCOD", "XYZCOD"}
    slot_cols = [c for c in df.columns if c not in non_slot]
    assert len(slot_cols) == 18

    val = validate_storage_matrix(df, policy)
    assert val["valid"] is True


# ==============================================================================
# 4. Cross-Table Integrity & Discrepancies Tests
# ==============================================================================

def test_product_foreign_keys():
    """Verify all order and picking wave products are defined in Product.csv."""
    prod_df = load_products()
    co_df = load_customer_orders()
    pw_df = load_picking_waves()

    prod_refs = set(prod_df["Reference"].str.strip())
    co_refs = set(co_df["Reference"].str.strip())
    pw_refs = set(pw_df["reference"].str.strip())

    # Every ordered item must exist in Product catalog
    assert co_refs.issubset(prod_refs)
    # Every picked item must exist in Product catalog
    assert pw_refs.issubset(prod_refs)


def test_known_discrepancies_reconciliation():
    """Explicitly verify documented dataset discrepancies are correctly observed."""
    sl_df = load_storage_locations()
    rs_df = load_storage_matrix("random")
    cb_df = load_storage_matrix("class_based")
    ds_df = load_storage_matrix("dedicated")
    hs_df = load_storage_matrix("hybrid")
    pw_df = load_picking_waves()

    sl_locs = set(sl_df["originalLocation"].str.strip())
    rs_locs = set(rs_df["originalLocation"].str.strip())
    cb_locs = set(cb_df["Location"].str.strip())
    ds_locs = set(ds_df["Location"].str.strip())
    hs_locs = set(hs_df["Location"].str.strip())
    pw_locs = set(pw_df["locations"].str.strip())

    # DISC-01 & DISC-02: Row counts
    assert len(sl_locs) == 2292
    assert len(rs_locs) == 2292
    assert len(cb_locs) == 2341
    assert len(ds_locs) == 2340
    assert len(hs_locs) == 2340

    # Dedicated & Hybrid have exactly 48 extra locations (aisles 25/26)
    assert len(ds_locs - sl_locs) == 48
    assert len(hs_locs - sl_locs) == 48
    assert (ds_locs - sl_locs) == (hs_locs - sl_locs)

    # Class-Based has 49 extra locations (the 48 from aisles 25/26 + 'Q-26-20')
    assert len(cb_locs - sl_locs) == 49
    assert (cb_locs - ds_locs) == {"Q-26-20"}

    # DISC-07: Picking Wave has 22 non-storage locations
    assert len(pw_locs - sl_locs) == 22
