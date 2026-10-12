"""Unit tests for SLAP-GRO Storage Matrix & Storage Policies."""

import pytest
import pandas as pd

from slap_gro.data.product import ProductCatalog
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.class_based import ClassBasedStoragePolicy
from slap_gro.storage.dedicated_storage import DedicatedStoragePolicy
from slap_gro.storage.hybrid_storage import HybridStoragePolicy
from slap_gro.storage.random_storage import RandomStoragePolicy
from slap_gro.storage.storage_matrix import (
    StorageMatrix,
    StorageUnit,
    StoredItem,
    parse_slot,
)
from slap_gro.storage import get_storage_policy
from slap_gro.warehouse.warehouse_layout import SyntheticWarehouse, build_warehouse


# ==============================================================================
# 1. Storage Matrix & Slot Parsing Tests
# ==============================================================================

def test_stored_item_and_parse_slot():
    """Verify StoredItem model and parse_slot parser behavior."""
    item = StoredItem(reference="  8551FLX ", quantity=15.0)
    assert item.reference == "8551FLX"
    assert item.quantity == 15.0
    assert item.is_empty is False

    empty_item = StoredItem(reference="", quantity=0.0)
    assert empty_item.is_empty is True

    # Parsing tests
    assert parse_slot("8551FLX;15.0") == StoredItem(reference="8551FLX", quantity=15.0)
    assert parse_slot("TQBVRI;7") == StoredItem(reference="TQBVRI", quantity=7.0)
    assert parse_slot("PLAIN_SKU") == StoredItem(reference="PLAIN_SKU", quantity=1.0)
    assert parse_slot("") is None
    assert parse_slot(None) is None
    assert parse_slot(float("nan")) is None


def test_storage_unit_methods():
    """Verify StorageUnit operations on its 18 slots."""
    slots = [
        StoredItem(reference="SKU_A", quantity=5.0),
        StoredItem(reference="", quantity=0.0),
        StoredItem(reference="SKU_B", quantity=10.0),
    ]
    unit = StorageUnit(location_id="A-01-01", slots=slots)
    assert unit.total_slots == 3
    assert unit.occupied_slots_count == 2
    assert unit.is_empty is False
    assert unit.contains_sku("SKU_A") is True
    assert unit.contains_sku("SKU_NON_EXISTENT") is False
    assert unit.get_quantity("SKU_B") == 10.0
    assert len(unit.get_items()) == 2


def test_storage_matrix_from_csv_and_indexing():
    """Verify loading real Random_Storage.csv matrix and fast SKU lookup indexing."""
    matrix = StorageMatrix.from_csv(policy="random")
    assert len(matrix) == 2292
    sample_loc = matrix.location_ids[0]
    assert sample_loc in matrix
    assert matrix.policy == "random"


    # Verify SKU index
    skus = matrix.stored_skus
    assert len(skus) > 0
    sample_sku = list(skus)[0]
    locs = matrix.get_locations_for_sku(sample_sku)
    assert len(locs) > 0
    assert matrix[locs[0]].contains_sku(sample_sku)

    # Export to DataFrame
    df = matrix.to_dataframe()
    assert len(df) == 2292
    assert "Location" in df.columns
    assert "slot_1" in df.columns
    assert "slot_18" in df.columns


# ==============================================================================
# 2. Storage Policies (Observed Mode)
# ==============================================================================

@pytest.mark.parametrize("policy_name", ["random", "class_based", "dedicated", "hybrid"])
def test_observed_storage_policies(policy_name):
    """Verify all 4 storage policies load observed dataset matrix via factory."""
    policy = get_storage_policy(policy_name, source="observed")
    assert isinstance(policy, StoragePolicy)
    assert policy.source == "observed"
    assert len(policy.matrix) >= 2292
    occupied = policy.get_all_occupied_locations()
    assert len(occupied) > 0


# ==============================================================================
# 3. Storage Policies (Generated Mode for Mode A)
# ==============================================================================

@pytest.fixture(scope="module")
def synthetic_env():
    warehouse = build_warehouse(mode="paper")
    catalog = ProductCatalog.generate_synthetic(n_products=100, seed=42)
    return warehouse, catalog


def test_generated_random_storage(synthetic_env):
    """Verify programmatic Random storage assignment."""
    warehouse, catalog = synthetic_env
    policy = get_storage_policy(
        "random",
        source="generated",
        warehouse=warehouse,
        catalog=catalog,
        seed=42,
    )
    assert policy.source == "generated"
    assert len(policy.matrix) == len(warehouse.get_all_locations())
    assert policy.matrix.total_occupied_slots > 0


def test_generated_class_based_storage(synthetic_env):
    """Verify programmatic Class-Based storage assignment with ABC zones."""
    warehouse, catalog = synthetic_env
    policy = get_storage_policy(
        "class_based",
        source="generated",
        warehouse=warehouse,
        catalog=catalog,
        seed=42,
    )
    assert policy.source == "generated"
    matrix = policy.matrix
    assert len(matrix) == len(warehouse.get_all_locations())

    # Check category codes in units
    categories = {u.category_code for u in matrix}
    assert categories == {"A", "B", "C"}


def test_generated_dedicated_storage(synthetic_env):
    """Verify programmatic Dedicated storage assignment ranking by demand."""
    warehouse, catalog = synthetic_env
    policy = get_storage_policy(
        "dedicated",
        source="generated",
        warehouse=warehouse,
        catalog=catalog,
        seed=42,
    )
    assert policy.source == "generated"
    assert len(policy.matrix) == len(warehouse.get_all_locations())


def test_generated_hybrid_storage(synthetic_env):
    """Verify programmatic Hybrid storage assignment."""
    warehouse, catalog = synthetic_env
    policy = get_storage_policy(
        "hybrid",
        source="generated",
        warehouse=warehouse,
        catalog=catalog,
        seed=42,
    )
    assert policy.source == "generated"
    matrix = policy.matrix
    assert len(matrix) == len(warehouse.get_all_locations())
    categories = {u.category_code for u in matrix}
    assert "A" in categories
    assert "B" in categories
    assert "C" in categories
