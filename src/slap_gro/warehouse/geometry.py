"""Geometric calculations, bounding boxes, and distance metrics for 3D warehouse layouts."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

from slap_gro.warehouse.location import Location


@dataclass(frozen=True)
class BoundingBox3D:
    """Represents a 3D bounding box for a warehouse region, block, or floor."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        """Dimension along X-axis."""
        return self.max_x - self.min_x

    @property
    def length(self) -> float:
        """Dimension along Y-axis."""
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        """Dimension along Z-axis."""
        return self.max_z - self.min_z

    @property
    def center(self) -> Tuple[float, float, float]:
        """Center coordinates of the bounding box."""
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within the bounding box."""
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )


def euclidean_distance_3d(
    p1: Tuple[float, float, float] | Location,
    p2: Tuple[float, float, float] | Location,
    z_weight: float = 1.0,
) -> float:
    """
    Calculate 3D Euclidean distance between two points or locations.

    Args:
        p1: Point or Location 1.
        p2: Point or Location 2.
        z_weight: Factor weighting vertical travel against horizontal travel.
    """
    c1 = p1.coordinates if isinstance(p1, Location) else p1
    c2 = p2.coordinates if isinstance(p2, Location) else p2

    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dz = (c1[2] - c2[2]) * z_weight
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def manhattan_distance_3d(
    p1: Tuple[float, float, float] | Location,
    p2: Tuple[float, float, float] | Location,
    z_weight: float = 1.0,
) -> float:
    """Calculate 3D Manhattan (rectilinear) distance."""
    c1 = p1.coordinates if isinstance(p1, Location) else p1
    c2 = p2.coordinates if isinstance(p2, Location) else p2

    return (
        abs(c1[0] - c2[0])
        + abs(c1[1] - c2[1])
        + abs(c1[2] - c2[2]) * z_weight
    )


def chebyshev_distance_3d(
    p1: Tuple[float, float, float] | Location,
    p2: Tuple[float, float, float] | Location,
) -> float:
    """Calculate 3D Chebyshev distance (maximum coordinate difference)."""
    c1 = p1.coordinates if isinstance(p1, Location) else p1
    c2 = p2.coordinates if isinstance(p2, Location) else p2

    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), abs(c1[2] - c2[2]))


def calculate_bounding_box(locations: Iterable[Location]) -> BoundingBox3D:
    """Compute the overall 3D bounding box enclosing a collection of Locations."""
    loc_list = list(locations)
    if not loc_list:
        return BoundingBox3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    xs = [loc.x for loc in loc_list]
    ys = [loc.y for loc in loc_list]
    zs = [loc.z for loc in loc_list]

    return BoundingBox3D(
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
        min_z=min(zs),
        max_z=max(zs),
    )


def calculate_block_bounding_boxes(
    locations: Iterable[Location],
) -> Dict[str, BoundingBox3D]:
    """Compute 3D bounding boxes grouped by block name."""
    by_block: Dict[str, List[Location]] = {}
    for loc in locations:
        blk = loc.block or "UNKNOWN"
        by_block.setdefault(blk, []).append(loc)

    return {blk: calculate_bounding_box(locs) for blk, locs in by_block.items()}
