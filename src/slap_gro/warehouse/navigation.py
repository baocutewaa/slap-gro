"""Navigation support points and waypoint mapping for warehouse layouts."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

from slap_gro.data.loaders import load_navigation_points
from slap_gro.data.preprocess import parse_coordinate_tuple
from slap_gro.warehouse.location import Location


@dataclass(frozen=True)
class NavigationPoint:
    """
    Represents a navigation support point / waypoint in the warehouse aisle network.

    Attributes:
        id: Identifier label (e.g. 'LC-01', 'CC-08', 'RC-15').
        x: X-coordinate in meters.
        y: Y-coordinate in meters.
        z: Z-coordinate (floor level, default 1.0).
    """

    id: str
    x: float
    y: float
    z: float = 1.0

    @property
    def coordinates(self) -> Tuple[float, float, float]:
        """Return 3D coordinates (x, y, z) as a tuple."""
        return (self.x, self.y, self.z)

    def distance_to(
        self,
        other: "NavigationPoint | Location | Tuple[float, float, float]",
        z_weight: float = 1.0,
    ) -> float:
        """Calculate 3D Euclidean distance to another point or location."""
        if isinstance(other, (NavigationPoint, Location)):
            ox, oy, oz = other.x, other.y, other.z
        else:
            ox, oy, oz = other

        dx = self.x - ox
        dy = self.y - oy
        dz = (self.z - oz) * z_weight
        return math.sqrt(dx * dx + dy * dy + dz * dz)


def load_csv_navigation_points(
    csv_path: Optional[str] = None,
) -> Dict[str, NavigationPoint]:
    """
    Load navigation support points from Support_Points_Navigation.csv.

    Returns:
        Dictionary mapping label -> NavigationPoint.
    """
    df = load_navigation_points(csv_path) if csv_path else load_navigation_points()
    points: Dict[str, NavigationPoint] = {}

    for _, row in df.iterrows():
        label = str(row["labels"]).strip()
        coords_str = str(row["points_specified"]).strip()
        x, y, z = parse_coordinate_tuple(coords_str)
        points[label] = NavigationPoint(id=label, x=x, y=y, z=z)

    return points


def find_nearest_navigation_point(
    target: Location | Tuple[float, float, float],
    navigation_points: Iterable[NavigationPoint],
) -> NavigationPoint:
    """
    Find the closest navigation support point to a given storage location.

    Args:
        target: Location or (x, y, z) tuple.
        navigation_points: Collection of candidate NavigationPoints.

    Returns:
        Nearest NavigationPoint.
    """
    pt_list = list(navigation_points)
    if not pt_list:
        raise ValueError("Navigation points list cannot be empty")

    return min(pt_list, key=lambda np: np.distance_to(target))


def generate_synthetic_navigation_points(
    block_bounds: Dict[str, Tuple[float, float, float, float]],
    cross_aisle_positions: List[float],
) -> Dict[str, NavigationPoint]:
    """
    Generate synthetic cross-aisle navigation support points for paper mode warehouse.

    Args:
        block_bounds: Dict of block -> (min_x, max_x, min_y, max_y).
        cross_aisle_positions: Y coordinates of cross-aisles (e.g. front, middle, back).

    Returns:
        Dict of point_id -> NavigationPoint.
    """
    points: Dict[str, NavigationPoint] = {}
    idx = 1

    for blk_name, (min_x, max_x, _, _) in block_bounds.items():
        for y_pos in cross_aisle_positions:
            # Left and right side of aisle blocks
            pt_left = NavigationPoint(id=f"NAV-{blk_name}-L{idx}", x=min_x - 2.0, y=y_pos, z=1.0)
            pt_right = NavigationPoint(id=f"NAV-{blk_name}-R{idx}", x=max_x + 2.0, y=y_pos, z=1.0)
            points[pt_left.id] = pt_left
            points[pt_right.id] = pt_right
            idx += 1

    return points
