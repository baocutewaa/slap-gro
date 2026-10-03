"""Route modeling, route distance calculation, and picker tour evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional

from slap_gro.warehouse.location import Location
from slap_gro.warehouse.navigation import NavigationPoint
from slap_gro.routing.distance import DistanceCache, DistanceMethod, distance
from slap_gro.routing.graph import NavigationGraph


PointType = Location | NavigationPoint


@dataclass
class Route:
    """
    Represents an ordered picking route through warehouse locations.

    Attributes:
        stops: Sequence of locations/waypoints visited by the picker.
        total_distance: Total travel distance in meters.
        stops_count: Number of pick locations visited (excluding depot returns).
        is_closed: Whether the route starts and ends at the same location (depot).
    """

    stops: List[PointType] = field(default_factory=list)
    total_distance: float = 0.0
    stops_count: int = 0
    is_closed: bool = False

    def __len__(self) -> int:
        return len(self.stops)


def route_distance(
    route: List[PointType],
    method: DistanceMethod = "network",
    graph: Optional[NavigationGraph] = None,
    distance_cache: Optional[DistanceCache] = None,
    z_weight: float = 1.0,
) -> float:
    """
    Calculate the total travel distance of an ordered route.

    Guarantees:
        - route_distance([]) == 0.0
        - route_distance([depot]) == 0.0
        - route_distance([depot, depot]) == 0.0
        - route_distance(route) >= 0.0

    Args:
        route: Ordered list of Locations / NavigationPoints to visit.
        method: Distance calculation mode ('network', 'euclidean', 'manhattan').
        graph: Optional NavigationGraph instance.
        distance_cache: Optional DistanceCache for accelerated GRO evaluations.
        z_weight: Penalty weight for vertical travel.

    Returns:
        Cumulative distance in meters (float >= 0.0).
    """
    if len(route) <= 1:
        return 0.0

    total = 0.0
    for i in range(len(route) - 1):
        p1 = route[i]
        p2 = route[i + 1]

        if distance_cache is not None:
            d = distance_cache.get_distance(p1, p2)
        else:
            d = distance(
                p1,
                p2,
                method=method,
                graph=graph,
                z_weight=z_weight,
            )
        total += d

    return float(total)


def build_route(
    depot: Location,
    pick_locations: List[Location],
    return_to_depot: bool = True,
) -> List[Location]:
    """
    Construct a complete picker tour starting at depot, visiting all pick locations,
    and returning to depot.

    Args:
        depot: Warehouse I/O depot location.
        pick_locations: Ordered list of item storage locations to pick.
        return_to_depot: If True, appends depot to the end of the route.

    Returns:
        List of Location objects: [depot, L1, L2, ..., Ln, depot]
    """
    route = [depot] + list(pick_locations)
    if return_to_depot:
        route.append(depot)
    return route


def evaluate_route(
    route_stops: List[PointType],
    method: DistanceMethod = "network",
    graph: Optional[NavigationGraph] = None,
    distance_cache: Optional[DistanceCache] = None,
) -> Route:
    """
    Evaluate a sequence of stops and return a populated Route object.

    Args:
        route_stops: List of locations forming the tour.
        method: Distance metric.
        graph: Navigation graph.
        distance_cache: Optional cache.

    Returns:
        Route instance with total distance and stop statistics.
    """
    total_dist = route_distance(
        route_stops,
        method=method,
        graph=graph,
        distance_cache=distance_cache,
    )

    is_closed = len(route_stops) >= 2 and route_stops[0].id == route_stops[-1].id

    # Stop count excluding depot start and finish
    effective_stops = len(route_stops) - 2 if is_closed else max(0, len(route_stops) - 1)

    return Route(
        stops=list(route_stops),
        total_distance=round(total_dist, 3),
        stops_count=effective_stops,
        is_closed=is_closed,
    )
