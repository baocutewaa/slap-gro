"""Navigation Graph implementation using NetworkX for warehouse aisle topology and shortest paths."""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math
import networkx as nx

from slap_gro.warehouse.geometry import euclidean_distance_3d
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.navigation import NavigationPoint, load_csv_navigation_points
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


class NavigationGraph:
    """
    Warehouse navigation network graph.

    Represents aisles, cross-aisles, support points, and depots as an undirected
    weighted graph to compute accurate picker travel distances across warehouse corridors.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self._node_coords: Dict[str, Tuple[float, float, float]] = {}
        self._navigation_points: Dict[str, NavigationPoint] = {}

    @property
    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        return self.graph.number_of_edges()

    def add_navigation_point(self, pt: NavigationPoint) -> None:
        """Add a navigation point node to the graph."""
        self._navigation_points[pt.id] = pt
        self._node_coords[pt.id] = (pt.x, pt.y, pt.z)
        self.graph.add_node(
            pt.id,
            x=pt.x,
            y=pt.y,
            z=pt.z,
            node_type="navigation_point",
        )

    def add_location_node(self, loc: Location) -> None:
        """Add a location node to the graph and attach it to the network."""
        self._node_coords[loc.id] = (loc.x, loc.y, loc.z)
        self.graph.add_node(
            loc.id,
            x=loc.x,
            y=loc.y,
            z=loc.z,
            node_type="location",
            block=loc.block,
            aisle=loc.aisle,
        )

    def add_edge(self, u: str, v: str, weight: Optional[float] = None) -> None:
        """Add an undirected edge between node u and node v with Euclidean weight."""
        if u not in self._node_coords or v not in self._node_coords:
            raise KeyError(f"Both nodes '{u}' and '{v}' must exist in graph before connecting")

        if weight is None:
            p1 = self._node_coords[u]
            p2 = self._node_coords[v]
            weight = euclidean_distance_3d(p1, p2)

        self.graph.add_edge(u, v, weight=float(weight))

    def build_from_support_points(
        self,
        points: Optional[Dict[str, NavigationPoint]] = None,
        depot: Optional[Location] = None,
    ) -> None:
        """
        Build grid topology from support navigation points (44 points for CSV warehouse).

        Connects points along the same cross-aisle (vertical columns with same X)
        and cross-connections across aisles (horizontal rows with matching Y).
        """
        if points is None:
            points = load_csv_navigation_points()

        # 1. Add all points as nodes
        for pt in points.values():
            self.add_navigation_point(pt)

        # 2. Connect points sharing the same X (vertical cross-aisles: LC, CC, RC)
        by_x: Dict[float, List[NavigationPoint]] = {}
        for pt in points.values():
            by_x.setdefault(round(pt.x, 1), []).append(pt)

        for _, col_points in by_x.items():
            sorted_col = sorted(col_points, key=lambda p: p.y)
            for i in range(len(sorted_col) - 1):
                self.add_edge(sorted_col[i].id, sorted_col[i + 1].id)

        # 3. Connect points sharing the same Y (horizontal cross-aisle passages)
        by_y: Dict[float, List[NavigationPoint]] = {}
        for pt in points.values():
            by_y.setdefault(round(pt.y, 1), []).append(pt)

        for _, row_points in by_y.items():
            if len(row_points) > 1:
                sorted_row = sorted(row_points, key=lambda p: p.x)
                for i in range(len(sorted_row) - 1):
                    self.add_edge(sorted_row[i].id, sorted_row[i + 1].id)

        # 4. Connect depot to nearest support point if provided
        if depot is not None:
            self.add_location_node(depot)
            nearest_pt = self.find_nearest_navigation_point(depot)
            if nearest_pt:
                self.add_edge(depot.id, nearest_pt.id)

    def build_for_warehouse(self, warehouse: WarehouseLayout) -> None:
        """
        Build navigation graph tailored for a specific WarehouseLayout.
        """
        self.graph.clear()
        self._node_coords.clear()
        self._navigation_points.clear()

        # Build support point backbone
        depot = warehouse.get_depot()
        if warehouse.navigation_points:
            self.build_from_support_points(warehouse.navigation_points, depot=depot)
        else:
            # Fallback: connect depot to first available support point
            self.build_from_support_points(depot=depot)

        # Connect all warehouse locations to their nearest navigation waypoint
        nav_pts = list(self._navigation_points.values())
        if nav_pts:
            for loc in warehouse.get_all_locations():
                self.add_location_node(loc)
                # Find nearest navigation waypoint
                nearest = min(nav_pts, key=lambda p: loc.distance_to(p))
                self.add_edge(loc.id, nearest.id)

    def find_nearest_navigation_point(
        self,
        target: Location | Tuple[float, float, float],
    ) -> Optional[NavigationPoint]:
        """Find the closest navigation point to a target location or coordinates."""
        if not self._navigation_points:
            return None

        t_coords = target.coordinates if isinstance(target, Location) else target
        return min(
            self._navigation_points.values(),
            key=lambda p: euclidean_distance_3d(p.coordinates, t_coords),
        )

    def shortest_path(
        self,
        source: str | Location | NavigationPoint,
        target: str | Location | NavigationPoint,
    ) -> List[str]:
        """
        Compute shortest path node sequence between source and target using Dijkstra's algorithm.

        Args:
            source: Source node ID, Location, or NavigationPoint.
            target: Target node ID, Location, or NavigationPoint.

        Returns:
            List of node IDs from source to target.
        """
        s_id = source if isinstance(source, str) else source.id
        t_id = target if isinstance(target, str) else target.id

        if s_id == t_id:
            return [s_id]

        # If node not yet in graph, attach temporarily
        temp_nodes: Set[str] = set()
        try:
            if s_id not in self.graph:
                if isinstance(source, (Location, NavigationPoint)):
                    if isinstance(source, Location):
                        self.add_location_node(source)
                    else:
                        self.add_navigation_point(source)
                    nearest_s = self.find_nearest_navigation_point(source)
                    if nearest_s and nearest_s.id != s_id:
                        self.add_edge(s_id, nearest_s.id)
                    temp_nodes.add(s_id)
                else:
                    raise KeyError(f"Source node '{s_id}' not found in navigation graph")

            if t_id not in self.graph:
                if isinstance(target, (Location, NavigationPoint)):
                    if isinstance(target, Location):
                        self.add_location_node(target)
                    else:
                        self.add_navigation_point(target)
                    nearest_t = self.find_nearest_navigation_point(target)
                    if nearest_t and nearest_t.id != t_id:
                        self.add_edge(t_id, nearest_t.id)
                    temp_nodes.add(t_id)
                else:
                    raise KeyError(f"Target node '{t_id}' not found in navigation graph")

            return nx.shortest_path(self.graph, source=s_id, target=t_id, weight="weight")
        finally:
            for tn in temp_nodes:
                if tn in self.graph:
                    self.graph.remove_node(tn)
                self._node_coords.pop(tn, None)

    def shortest_path_distance(
        self,
        source: str | Location | NavigationPoint,
        target: str | Location | NavigationPoint,
    ) -> float:
        """
        Compute shortest travel distance through the warehouse navigation graph.

        Args:
            source: Source node ID, Location, or NavigationPoint.
            target: Target node ID, Location, or NavigationPoint.

        Returns:
            Shortest travel distance in meters.
        """
        s_id = source if isinstance(source, str) else source.id
        t_id = target if isinstance(target, str) else target.id

        if s_id == t_id:
            return 0.0

        temp_nodes: Set[str] = set()
        try:
            if s_id not in self.graph:
                if isinstance(source, (Location, NavigationPoint)):
                    if isinstance(source, Location):
                        self.add_location_node(source)
                    else:
                        self.add_navigation_point(source)
                    nearest_s = self.find_nearest_navigation_point(source)
                    if nearest_s and nearest_s.id != s_id:
                        self.add_edge(s_id, nearest_s.id)
                    temp_nodes.add(s_id)
                else:
                    raise KeyError(f"Source node '{s_id}' not found in navigation graph")

            if t_id not in self.graph:
                if isinstance(target, (Location, NavigationPoint)):
                    if isinstance(target, Location):
                        self.add_location_node(target)
                    else:
                        self.add_navigation_point(target)
                    nearest_t = self.find_nearest_navigation_point(target)
                    if nearest_t and nearest_t.id != t_id:
                        self.add_edge(t_id, nearest_t.id)
                    temp_nodes.add(t_id)
                else:
                    raise KeyError(f"Target node '{t_id}' not found in navigation graph")

            return float(nx.shortest_path_length(self.graph, source=s_id, target=t_id, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            p1 = source.coordinates if isinstance(source, (Location, NavigationPoint)) else self._node_coords.get(s_id, (0, 0, 0))
            p2 = target.coordinates if isinstance(target, (Location, NavigationPoint)) else self._node_coords.get(t_id, (0, 0, 0))
            return euclidean_distance_3d(p1, p2)
        finally:
            for tn in temp_nodes:
                if tn in self.graph:
                    self.graph.remove_node(tn)
                self._node_coords.pop(tn, None)
