"""Unit tests for SLAP-GRO Picking Layer & Route Targets."""

import pytest
import pandas as pd

from slap_gro.data.loaders import load_customer_orders, load_picking_waves
from slap_gro.data.product import ProductCatalog
from slap_gro.picking.order import batch_customer_orders
from slap_gro.picking.picking_points import (
    build_gro_route_targets,
    build_wave_tour_sequence,
    get_wave_picking_locations,
)
from slap_gro.picking.wave import (
    PickingItem,
    PickingWave,
    PickingWaveDataset,
)
from slap_gro.storage import get_storage_policy
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.warehouse_layout import build_warehouse


# ==============================================================================
# 1. PickingItem & PickingWave Entity Tests
# ==============================================================================

def test_picking_item_initialization():
    """Verify PickingItem attributes and data trimming."""
    item = PickingItem(reference="  8551FLX ", quantity=1, location=" A-14-11 ", size_us=9.5)
    assert item.reference == "8551FLX"
    assert item.quantity == 1
    assert item.location == "A-14-11"
    assert item.size_us == 9.5


def test_picking_wave_aggregation():
    """Verify PickingWave calculates total units and unique locations correctly."""
    wave = PickingWave(wave_id=1)
    wave.add_item(PickingItem(reference="SKU_1", quantity=1, location="A-01-01"))
    wave.add_item(PickingItem(reference="SKU_1", quantity=1, location="A-01-01"))  # duplicate location
    wave.add_item(PickingItem(reference="SKU_2", quantity=2, location="A-02-05"))

    assert wave.total_picks == 3
    assert wave.total_units == 4
    assert wave.unique_skus == {"SKU_1", "SKU_2"}
    assert wave.unique_locations == ["A-01-01", "A-02-05"]


# ==============================================================================
# 2. Picking Wave Dataset Loading & Generation
# ==============================================================================

def test_picking_wave_dataset_from_sample():
    """Verify loading real picking waves into PickingWaveDataset from sample."""
    df_sample = load_picking_waves().head(100)
    dataset = PickingWaveDataset.from_dataframe(df_sample)
    assert len(dataset) > 0
    first_id = dataset.wave_ids[0]
    wave = dataset[first_id]
    assert isinstance(wave, PickingWave)
    assert len(wave.items) > 0


def test_batch_customer_orders():
    """Verify batching Customer_Order records into consolidated waves."""
    df_sample = load_customer_orders().head(100)
    dataset = batch_customer_orders(df_sample)
    assert len(dataset) > 0
    first_id = dataset.wave_ids[0]
    wave = dataset[first_id]
    assert wave.total_units > 0


def test_synthetic_picking_wave_generation():
    """Verify Mode A synthetic wave generator (50 waves x 25 items)."""
    warehouse = build_warehouse(mode="paper")
    catalog = ProductCatalog.generate_synthetic(n_products=100, seed=42)
    policy = get_storage_policy("random", source="generated", warehouse=warehouse, catalog=catalog)

    dataset = PickingWaveDataset.generate_synthetic(
        catalog=catalog,
        storage_policy=policy,
        n_waves=5,
        items_per_wave=10,
        seed=42,
    )
    assert len(dataset) == 5
    for w in dataset:
        assert w.total_picks == 10
        assert len(w.unique_locations) > 0


# ==============================================================================
# 3. Picking Points Extraction for GRO (Picking -> Routing bridge)
# ==============================================================================

def test_picking_points_decoupled_from_skus():
    """
    Verify that build_gro_route_targets produces pure (depot, [locations])
    without SKU dependencies.
    """
    warehouse = build_warehouse(mode="real")
    wave = PickingWave(wave_id=99)
    wave.add_item(PickingItem(reference="SKU_1", quantity=1, location="A-14-11"))
    wave.add_item(PickingItem(reference="SKU_2", quantity=1, location="A-14-12"))

    depot, pick_locs = build_gro_route_targets(wave, warehouse)
    assert isinstance(depot, Location)
    assert len(pick_locs) == 2
    assert all(isinstance(loc, Location) for loc in pick_locs)

    # Tour sequence: [depot, L1, L2, depot]
    tour = build_wave_tour_sequence(wave, warehouse)
    assert len(tour) == 4
    assert tour[0] == depot
    assert tour[-1] == depot
