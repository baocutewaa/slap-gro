"""Unit and integration tests for Phase 3: Navigation Layer & Routing Engine."""

import math
import numpy as np
import pytest

from slap_gro.warehouse.location import Location
from slap_gro.warehouse.navigation import NavigationPoint, load_csv_navigation_points
from slap_gro.warehouse.warehouse_layout import build_warehouse
from slap_gro.routing.distance import (
    DistanceCache,
    compute_distance_matrix,
    distance,
    get_default_navigation_graph,
)
from slap_gro.routing.graph import NavigationGraph
from slap_gro.routing.route import (
    Route,
    build_route,
    evaluate_route,
    route_distance,
)


# ==============================================================================
# 1. Core Distance API Contracts
# ==============================================================================

def test_distance_identity_and_non_negativity():
    """Verify core mathematical invariants: dist(A, A) == 0 and dist(A, B) >= 0."""
    loc_a = Location(id="LOC-A", x=100.0, y=50.0, z=1.0)
    loc_b = Location(id="LOC-B", x=200.0, y=150.0, z=2.0)

    for method in ["euclidean", "manhattan", "chebyshev", "network"]:
        # Invariant 1: Identity
        assert distance(loc_a, loc_a, method=method) == 0.0
        # Invariant 2: Non-negativity
        d_ab = distance(loc_a, loc_b, method=method)
        assert d_ab > 0.0
        # Invariant 3: Symmetry
        d_ba = distance(loc_b, loc_a, method=method)
        assert math.isclose(d_ab, d_ba, rel_tol=1e-5)


def test_distance_methods_numeric_accuracy():
    """Verify distance accuracy for Euclidean, Manhattan, Chebyshev."""
    p1 = Location(id="P1", x=0.0, y=0.0, z=1.0)
    p2 = Location(id="P2", x=30.0, y=40.0, z=1.0)

    # 2D on same z: dx=30, dy=40 -> Euclidean = 50.0, Manhattan = 70.0, Chebyshev = 40.0
    assert math.isclose(distance(p1, p2, method="euclidean"), 50.0)
    assert math.isclose(distance(p1, p2, method="manhattan"), 70.0)
    assert math.isclose(distance(p1, p2, method="chebyshev"), 40.0)


def test_triangle_inequality():
    """Verify triangle inequality holds for Euclidean and network distance."""
    a = Location(id="A", x=10.0, y=20.0, z=1.0)
    b = Location(id="B", x=50.0, y=80.0, z=1.0)
    c = Location(id="C", x=90.0, y=30.0, z=1.0)

    for method in ["euclidean", "manhattan", "network"]:
        d_ac = distance(a, c, method=method)
        d_ab = distance(a, b, method=method)
        d_bc = distance(b, c, method=method)
        assert d_ac <= d_ab + d_bc + 1e-6


# ==============================================================================
# 2. Navigation Graph Tests
# ==============================================================================

def test_navigation_graph_construction_from_support_points():
    """Verify NavigationGraph connects the 44 warehouse support points."""
    pts = load_csv_navigation_points()
    assert len(pts) == 44

    graph = NavigationGraph()
    graph.build_from_support_points(pts)

    assert graph.node_count == 44
    assert graph.edge_count > 0

    # Test pathfinding between two distinct cross-aisle points (e.g. LC-01 to CC-17)
    path = graph.shortest_path("LC-01", "CC-17")
    assert len(path) > 1
    assert path[0] == "LC-01"
    assert path[-1] == "CC-17"

    d = graph.shortest_path_distance("LC-01", "CC-17")
    assert d > 0.0


def test_navigation_graph_with_warehouse():
    """Verify building navigation graph directly for a WarehouseLayout."""
    wh_real = build_warehouse("real")
    graph = NavigationGraph()
    graph.build_for_warehouse(wh_real)

    # Depot and locations attached
    assert graph.graph.has_node("DEPOT")
    assert graph.graph.has_node("A-14-11")

    # Shortest path from depot to a storage location
    path = graph.shortest_path("DEPOT", "A-14-11")
    assert len(path) >= 2
    assert path[0] == "DEPOT"
    assert path[-1] == "A-14-11"

    dist = graph.shortest_path_distance("DEPOT", "A-14-11")
    assert dist > 0.0


# ==============================================================================
# 3. Route Distance & Tour Evaluation Tests
# ==============================================================================

def test_route_distance_contracts():
    """Verify route distance invariants required by Plan.md."""
    depot = Location(id="DEPOT", x=0.0, y=0.0, z=1.0)
    loc1 = Location(id="L1", x=10.0, y=20.0, z=1.0)
    loc2 = Location(id="L2", x=30.0, y=50.0, z=1.0)

    # Plan requirement: assert route_distance([depot, depot]) == 0
    assert route_distance([depot, depot]) == 0.0
    assert route_distance([depot]) == 0.0
    assert route_distance([]) == 0.0

    # Multi-stop route
    tour = [depot, loc1, loc2, depot]
    dist = route_distance(tour, method="euclidean")
    assert dist > 0.0

    # Verify cumulative calculation
    d1 = distance(depot, loc1, method="euclidean")
    d2 = distance(loc1, loc2, method="euclidean")
    d3 = distance(loc2, depot, method="euclidean")
    assert math.isclose(dist, d1 + d2 + d3)


def test_build_route_and_evaluate_route():
    """Verify build_route and evaluate_route constructs valid closed tours."""
    depot = Location(id="DEPOT", x=0.0, y=0.0, z=1.0)
    picks = [
        Location(id="P1", x=15.0, y=10.0, z=1.0),
        Location(id="P2", x=25.0, y=30.0, z=1.0),
        Location(id="P3", x=40.0, y=20.0, z=1.0),
    ]

    route_list = build_route(depot, picks, return_to_depot=True)
    assert len(route_list) == 5
    assert route_list[0].id == "DEPOT"
    assert route_list[-1].id == "DEPOT"

    route_eval = evaluate_route(route_list, method="euclidean")
    assert isinstance(route_eval, Route)
    assert route_eval.is_closed is True
    assert route_eval.stops_count == 3
    assert route_eval.total_distance > 0.0


# ==============================================================================
# 4. Distance Matrix and Cache Tests
# ==============================================================================

def test_compute_distance_matrix():
    """Verify pairwise symmetric distance matrix generation."""
    locs = [
        Location(id="L1", x=0.0, y=0.0, z=1.0),
        Location(id="L2", x=10.0, y=0.0, z=1.0),
        Location(id="L3", x=0.0, y=20.0, z=1.0),
    ]

    matrix = compute_distance_matrix(locs, method="euclidean")
    assert matrix.shape == (3, 3)

    # Diagonal is 0
    assert np.all(np.diag(matrix) == 0.0)

    # Symmetric
    assert np.allclose(matrix, matrix.T)

    # Values
    assert math.isclose(matrix[0, 1], 10.0)
    assert math.isclose(matrix[0, 2], 20.0)


def test_distance_cache():
    """Verify DistanceCache caches evaluations and speeds up lookups."""
    cache = DistanceCache(method="euclidean")
    loc_a = Location(id="A", x=10.0, y=10.0, z=1.0)
    loc_b = Location(id="B", x=40.0, y=50.0, z=1.0)

    d1 = cache.get_distance(loc_a, loc_b)
    d2 = cache.get_distance(loc_b, loc_a)  # Symmetric lookup

    assert math.isclose(d1, 50.0)
    assert math.isclose(d1, d2)
    assert len(cache._cache) == 1  # Only 1 unique key stored

    cache.clear()
    assert len(cache._cache) == 0
