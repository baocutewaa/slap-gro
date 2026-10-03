"""Location model and domain representation for warehouse storage units."""

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class Location:
    """
    Represents a storage unit / location in the warehouse.

    Attributes:
        id: Unique location identifier (e.g., 'A-14-11' or 'SYN-A-01-01-1').
        x: X-coordinate in meters (or layout units).
        y: Y-coordinate in meters (or layout units).
        z: Z-coordinate / level in meters or integer level.
        capacity: Number of slots in this storage unit (default 18).
        block: Block identifier (e.g., 'A', 'B', 'C').
        aisle: Aisle number within the block or warehouse.
        side: Shelf side ('L' for left, 'R' for right, or None).
        level: Vertical shelf level (typically 1 to 4).
    """

    id: str
    x: float
    y: float
    z: float
    capacity: int = 18
    block: Optional[str] = None
    aisle: Optional[int] = None
    side: Optional[str] = None
    level: int = 1

    @property
    def coordinates(self) -> Tuple[float, float, float]:
        """Return 3D coordinates (x, y, z) as a tuple."""
        return (self.x, self.y, self.z)

    def distance_to(
        self,
        other: Any,
        z_weight: float = 1.0,
    ) -> float:
        """
        Calculate 3D Euclidean distance to another Location, NavigationPoint, or coordinate tuple.
        """
        if hasattr(other, "x") and hasattr(other, "y") and hasattr(other, "z"):
            ox, oy, oz = other.x, other.y, other.z
        elif hasattr(other, "coordinates"):
            ox, oy, oz = other.coordinates
        else:
            ox, oy, oz = other

        dx = self.x - ox
        dy = self.y - oy
        dz = (self.z - oz) * z_weight
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def manhattan_distance_to(
        self,
        other: Any,
        z_weight: float = 1.0,
    ) -> float:
        """Calculate 3D Manhattan (rectilinear) distance."""
        if hasattr(other, "x") and hasattr(other, "y") and hasattr(other, "z"):
            ox, oy, oz = other.x, other.y, other.z
        elif hasattr(other, "coordinates"):
            ox, oy, oz = other.coordinates
        else:
            ox, oy, oz = other

        return abs(self.x - ox) + abs(self.y - oy) + abs(self.z - oz) * z_weight

    @classmethod
    def from_csv_row(
        cls,
        loc_id: str,
        x: float,
        y: float,
        z: float,
        capacity: int = 18,
    ) -> "Location":
        """
        Construct a Location from a row in Storage_Location.csv.

        Parses standard warehouse naming like 'A-14-11':
        - Block: 'A'
        - Aisle: 14
        - Level: derived from z coordinate (int)
        - Side: derived from slot index if applicable
        """
        clean_id = loc_id.strip()
        parts = clean_id.split("-")
        block = parts[0] if len(parts) > 0 else None
        aisle = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        level = int(z)

        # Slot encoding often uses the last digits (e.g., 11: level 1, slot 1; 21: level 2, slot 1)
        side = None
        if len(parts) > 2 and parts[2].isdigit():
            slot_num = int(parts[2])
            side = "L" if slot_num % 2 == 1 else "R"

        return cls(
            id=clean_id,
            x=float(x),
            y=float(y),
            z=float(z),
            capacity=capacity,
            block=block,
            aisle=aisle,
            side=side,
            level=level,
        )
