"""Routing, distance calculation, and baseline heuristics package for SLAP-GRO."""

from slap_gro.routing.baseline import (
    BaseRoutingHeuristic,
    BaselineResult,
    NearestNeighborBaseline,
    OriginalCompanyRouteBaseline,
    ReconstructedFIFOBaseline,
    SShapeBaseline,
    evaluate_all_baselines,
    evaluate_baseline,
    get_baseline_heuristic,
)
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
    evaluate_wave_route,
    route_distance,
)

__all__ = [
    # Routing & Graph
    "NavigationGraph",
    "distance",
    "compute_distance_matrix",
    "DistanceCache",
    "get_default_navigation_graph",
    "Route",
    "route_distance",
    "build_route",
    "evaluate_route",
    "evaluate_wave_route",
    # Baselines
    "BaselineResult",
    "BaseRoutingHeuristic",
    "OriginalCompanyRouteBaseline",
    "ReconstructedFIFOBaseline",
    "NearestNeighborBaseline",
    "SShapeBaseline",
    "get_baseline_heuristic",
    "evaluate_baseline",
    "evaluate_all_baselines",
]
