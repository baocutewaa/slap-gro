"""Unit and integration tests for Phase 2: Warehouse Layer."""

import math
import pytest

from slap_gro.warehouse.geometry import (
    BoundingBox3D,
    calculate_block_bounding_boxes,
    calculate_bounding_box,
    chebyshev_distance_3d,
    euclidean_distance_3d,
    manhattan_distance_3d,
)
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.navigation import (
    NavigationPoint,
    find_nearest_navigation_point,
    load_csv_navigation_points,
)
from slap_gro.warehouse.warehouse_layout import (
    CSVWarehouse,
    SyntheticWarehouse,
    build_warehouse,
)


# ==============================================================================
# 1. Location Model Tests
# ==============================================================================

def test_location_attributes_and_properties():
    """Verify Location properties, coordinates, and immutability."""
    loc = Location(
        id="A-14-11",
        x=368.0,
        y=0.0,
        z=1.0,
        capacity=18,
        block="A",
        aisle=14,
        side="L",
        level=1,
    )
    assert loc.id == "A-14-11"
    assert loc.coordinates == (368.0, 0.0, 1.0)
    assert loc.capacity == 18
    assert loc.block == "A"
    assert loc.aisle == 14
    assert loc.side == "L"
    assert loc.level == 1


def test_location_distances():
    """Verify 3D Euclidean and Manhattan distance calculations."""
    loc1 = Location(id="L1", x=0.0, y=0.0, z=0.0)
    loc2 = Location(id="L2", x=3.0, y=4.0, z=12.0)

    # 3D Euclidean: sqrt(3^2 + 4^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13.0
    assert math.isclose(loc1.distance_to(loc2), 13.0)
    assert math.isclose(loc1.distance_to((3.0, 4.0, 12.0)), 13.0)

    # Manhattan: 3 + 4 + 12 = 19.0
    assert math.isclose(loc1.manhattan_distance_to(loc2), 19.0)


def test_location_from_csv_row():
    """Verify parsing location naming convention from CSV."""
    loc = Location.from_csv_row(loc_id="A-14-11", x=368.0, y=0.0, z=1.0)
    assert loc.id == "A-14-11"
    assert loc.block == "A"
    assert loc.aisle == 14
    assert loc.level == 1
    assert loc.side == "L"  # odd slot -> Left


# ==============================================================================
# 2. Geometry Module Tests
# ==============================================================================

def test_bounding_box_3d():
    """Verify BoundingBox3D dimensions and containment checks."""
    bbox = BoundingBox3D(min_x=10.0, max_x=50.0, min_y=20.0, max_y=100.0, min_z=1.0, max_z=5.0)
    assert bbox.width == 40.0
    assert bbox.length == 80.0
    assert bbox.height == 4.0
    assert bbox.center == (30.0, 60.0, 3.0)
    assert bbox.contains(25.0, 50.0, 2.0) is True
    assert bbox.contains(5.0, 50.0, 2.0) is False


def test_geometry_distances():
    """Verify standalone distance metric functions."""
    p1 = (0.0, 0.0, 0.0)
    p2 = (6.0, 8.0, 0.0)
    assert math.isclose(euclidean_distance_3d(p1, p2), 10.0)
    assert math.isclose(manhattan_distance_3d(p1, p2), 14.0)
    assert math.isclose(chebyshev_distance_3d(p1, p2), 8.0)


def test_calculate_bounding_box():
    """Verify bounding box computation over collection of locations."""
    locs = [
        Location(id="1", x=10.0, y=20.0, z=1.0),
        Location(id="2", x=50.0, y=5.0, z=4.0),
        Location(id="3", x=30.0, y=90.0, z=2.0),
    ]
    bbox = calculate_bounding_box(locs)
    assert bbox.min_x == 10.0
    assert bbox.max_x == 50.0
    assert bbox.min_y == 5.0
    assert bbox.max_y == 90.0
    assert bbox.min_z == 1.0
    assert bbox.max_z == 4.0


# ==============================================================================
# 3. Navigation Module Tests
# ==============================================================================

def test_load_csv_navigation_points():
    """Verify loading 44 navigation points from Support_Points_Navigation.csv."""
    points = load_csv_navigation_points()
    assert len(points) == 44
    assert "LC-01" in points
    assert "CC-01" in points
    assert "RC-08" in points

    pt = points["LC-01"]
    assert math.isclose(pt.x, 66.0)
    assert math.isclose(pt.y, -29.0)
    assert math.isclose(pt.z, 1.0)


def test_find_nearest_navigation_point():
    """Verify lookup of nearest navigation waypoint."""
    points = [
        NavigationPoint(id="P1", x=10.0, y=10.0, z=1.0),
        NavigationPoint(id="P2", x=100.0, y=100.0, z=1.0),
    ]
    target = Location(id="T", x=12.0, y=11.0, z=1.0)
    nearest = find_nearest_navigation_point(target, points)
    assert nearest.id == "P1"


# ==============================================================================
# 4. Mode A: SyntheticWarehouse Tests
# ==============================================================================

def test_synthetic_warehouse_structure_and_contracts():
    """Verify paper mode SyntheticWarehouse matches paper requirements."""
    wh = SyntheticWarehouse(n_storage_units=847, slots_per_unit=18)

    # Core contracts
    assert wh.total_locations == 847
    assert wh.total_capacity == 15246  # 847 * 18

    # Blocks: A, B, C
    blocks = wh.get_blocks()
    assert blocks == ["A", "B", "C"]

    # Aisles: 25 aisles total (1 to 25)
    aisles = wh.get_aisles()
    assert len(aisles) == 25
    assert aisles == list(range(1, 26))

    # Levels: 4 levels
    levels = wh.get_levels()
    assert levels == [1, 2, 3, 4]

    # Required API checks
    first_loc_id = wh.get_all_locations()[0].id
    loc = wh.get_location(first_loc_id)
    assert loc is not None
    assert wh.get_capacity(first_loc_id) == 18

    depot = wh.get_depot()
    assert depot is not None
    assert depot.id == "DEPOT"

    # Query helpers
    locs_a = wh.get_locations_by_block("A")
    assert len(locs_a) > 0
    locs_aisle1 = wh.get_locations_by_aisle(1)
    assert len(locs_aisle1) > 0
    locs_lvl1 = wh.get_locations_by_level(1)
    assert len(locs_lvl1) > 0


# ==============================================================================
# 5. Mode B: CSVWarehouse Tests
# ==============================================================================

def test_csv_warehouse_structure_and_contracts():
    """Verify real mode CSVWarehouse matches Storage_Location.csv."""
    wh = CSVWarehouse()

    # Core contracts
    assert wh.total_locations == 2292
    assert wh.total_capacity == 41256  # 2292 * 18

    # Levels distribution
    assert len(wh.get_locations_by_level(1)) == 846
    assert len(wh.get_locations_by_level(2)) == 846
    assert len(wh.get_locations_by_level(3)) == 301
    assert len(wh.get_locations_by_level(4)) == 299

    # Required API checks
    loc = wh.get_location("A-14-11")
    assert loc is not None
    assert loc.coordinates == (368.0, 0.0, 1.0)
    assert wh.get_capacity("A-14-11") == 18

    depot = wh.get_depot()
    assert depot is not None
    assert depot.id == "DEPOT"

    # Navigation points
    assert len(wh.navigation_points) == 44


# ==============================================================================
# 6. Factory Function Tests
# ==============================================================================

def test_build_warehouse_factory():
    """Verify build_warehouse creates appropriate backends."""
    wh_paper = build_warehouse("paper")
    assert isinstance(wh_paper, SyntheticWarehouse)
    assert wh_paper.total_locations == 847

    wh_real = build_warehouse("real")
    assert isinstance(wh_real, CSVWarehouse)
    assert wh_real.total_locations == 2292

    with pytest.raises(ValueError):
        build_warehouse("invalid_mode")
