"""Distance computation metrics and pairwise matrix calculation for warehouse routing."""

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

from slap_gro.warehouse.geometry import (
    chebyshev_distance_3d,
    euclidean_distance_3d,
    manhattan_distance_3d,
)
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.navigation import NavigationPoint
from slap_gro.routing.graph import NavigationGraph


PointLike = Location | NavigationPoint | Tuple[float, float, float]
DistanceMethod = Literal["network", "euclidean", "manhattan", "chebyshev"]

# Default singleton graph for quick routing lookups
_GLOBAL_DEFAULT_GRAPH: Optional[NavigationGraph] = None


def get_default_navigation_graph() -> NavigationGraph:
    """Retrieve or lazily initialize the default warehouse navigation graph."""
    global _GLOBAL_DEFAULT_GRAPH
    if _GLOBAL_DEFAULT_GRAPH is None:
        g = NavigationGraph()
        g.build_from_support_points()
        _GLOBAL_DEFAULT_GRAPH = g
    return _GLOBAL_DEFAULT_GRAPH


def distance(
    point_a: PointLike,
    point_b: PointLike,
    method: DistanceMethod = "network",
    graph: Optional[NavigationGraph] = None,
    z_weight: float = 1.0,
) -> float:
    """
    Calculate distance between two points, locations, or waypoints.

    Guarantees:
        - distance(A, A) == 0.0
        - distance(A, B) >= 0.0
        - distance(A, B) == distance(B, A) (symmetric)

    Args:
        point_a: First location, waypoint, or 3D coordinate tuple.
        point_b: Second location, waypoint, or 3D coordinate tuple.
        method: Distance calculation mode:
            - 'network': Distance through the warehouse aisle navigation graph.
            - 'euclidean': 3D Euclidean straight-line distance.
            - 'manhattan': 3D Manhattan (rectilinear) distance.
            - 'chebyshev': 3D Chebyshev distance.
        graph: Optional custom NavigationGraph. If None, default graph is used.
        z_weight: Penalty weight for vertical elevation travel.

    Returns:
        Travel distance in meters (float >= 0.0).
    """
    # 1. Identity check
    if point_a is point_b:
        return 0.0

    if isinstance(point_a, (Location, NavigationPoint)) and isinstance(point_b, (Location, NavigationPoint)):
        if point_a.id == point_b.id:
            return 0.0

    # 2. Coordinate extraction
    c_a = point_a.coordinates if isinstance(point_a, (Location, NavigationPoint)) else point_a
    c_b = point_b.coordinates if isinstance(point_b, (Location, NavigationPoint)) else point_b

    if c_a == c_b:
        return 0.0

    # 3. Method execution
    if method == "network":
        nav_graph = graph or get_default_navigation_graph()
        return nav_graph.shortest_path_distance(point_a, point_b)
    elif method == "euclidean":
        return euclidean_distance_3d(c_a, c_b, z_weight=z_weight)
    elif method == "manhattan":
        return manhattan_distance_3d(c_a, c_b, z_weight=z_weight)
    elif method == "chebyshev":
        return chebyshev_distance_3d(c_a, c_b)
    else:
        raise ValueError(f"Unknown distance method: '{method}'. Expected 'network', 'euclidean', 'manhattan', or 'chebyshev'.")


def compute_distance_matrix(
    locations: List[Location | NavigationPoint],
    method: DistanceMethod = "network",
    graph: Optional[NavigationGraph] = None,
    z_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute pairwise symmetric distance matrix for a list of locations.

    Args:
        locations: List of N locations (including depot if desired).
        method: Distance metric ('network', 'euclidean', 'manhattan', 'chebyshev').
        graph: Optional navigation graph instance.
        z_weight: Factor weighting vertical travel against horizontal travel.

    Returns:
        N x N NumPy array of float distances where matrix[i, j] = dist(locations[i], locations[j]).
    """
    n = len(locations)
    matrix = np.zeros((n, n), dtype=np.float64)

    if n <= 1:
        return matrix

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(
                locations[i],
                locations[j],
                method=method,
                graph=graph,
                z_weight=z_weight,
            )
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


class DistanceCache:
    """
    Fast memoized pairwise distance cache for repeated route evaluation in GRO.
    """

    def __init__(
        self,
        method: DistanceMethod = "network",
        graph: Optional[NavigationGraph] = None,
    ) -> None:
        self.method = method
        self.graph = graph
        self._cache: Dict[Tuple[str, str], float] = {}

    def get_distance(self, loc_a: Location | NavigationPoint, loc_b: Location | NavigationPoint) -> float:
        """Retrieve distance with symmetric hash caching."""
        id_a, id_b = loc_a.id, loc_b.id
        if id_a == id_b:
            return 0.0

        key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
        if key not in self._cache:
            d = distance(loc_a, loc_b, method=self.method, graph=self.graph)
            self._cache[key] = d
        return self._cache[key]

    def clear(self) -> None:
        """Clear cached distances."""
        self._cache.clear()
