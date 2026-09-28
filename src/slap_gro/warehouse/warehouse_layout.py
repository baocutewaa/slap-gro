"""Warehouse layout abstraction supporting both Synthetic (paper mode) and CSV (real mode) backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from slap_gro.config.loader import load_paper_config, load_real_config
from slap_gro.data.loaders import load_storage_locations
from slap_gro.warehouse.geometry import BoundingBox3D, calculate_bounding_box
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.navigation import (
    NavigationPoint,
    generate_synthetic_navigation_points,
    load_csv_navigation_points,
)


class WarehouseLayout(ABC):
    """
    Abstract Base Class for 3D warehouse layouts.

    Defines standard required API across paper replication and real-data modes:
    - get_location(id)
    - get_capacity(id)
    - get_all_locations()
    - get_depot()
    """

    def __init__(self, mode: str, name: str) -> None:
        self.mode = mode
        self.name = name
        self._locations: Dict[str, Location] = {}
        self._depot: Optional[Location] = None
        self._navigation_points: Dict[str, NavigationPoint] = {}

    @abstractmethod
    def build(self) -> None:
        """Construct the warehouse layout locations and geometry."""
        pass

    def get_location(self, loc_id: str) -> Optional[Location]:
        """Retrieve a Location by its unique identifier."""
        return self._locations.get(str(loc_id).strip())

    def get_capacity(self, loc_id: str) -> int:
        """Return the slot capacity of a given location (default 18)."""
        loc = self.get_location(loc_id)
        return loc.capacity if loc else 0

    def get_all_locations(self) -> List[Location]:
        """Return a list of all storage locations in the warehouse."""
        return list(self._locations.values())

    def get_depot(self) -> Location:
        """Return the I/O depot location (starting and ending point for picking routes)."""
        if self._depot is None:
            self._depot = Location(
                id="DEPOT",
                x=0.0,
                y=0.0,
                z=1.0,
                capacity=0,
                block="DEPOT",
                aisle=0,
                level=1,
            )
        return self._depot

    @property
    def total_locations(self) -> int:
        """Total number of storage locations."""
        return len(self._locations)

    @property
    def total_capacity(self) -> int:
        """Total storage slot capacity (sum of slot capacities)."""
        return sum(loc.capacity for loc in self._locations.values())

    def get_locations_by_block(self, block: str) -> List[Location]:
        """Return all locations belonging to a specific block (e.g., 'A', 'B', 'C')."""
        return [loc for loc in self._locations.values() if loc.block == block]

    def get_locations_by_aisle(self, aisle: int) -> List[Location]:
        """Return all locations belonging to a specific aisle number."""
        return [loc for loc in self._locations.values() if loc.aisle == aisle]

    def get_locations_by_level(self, level: int) -> List[Location]:
        """Return all locations located on a specific vertical floor/level."""
        return [loc for loc in self._locations.values() if loc.level == level]

    def get_blocks(self) -> List[str]:
        """Return sorted list of distinct block identifiers."""
        return sorted({loc.block for loc in self._locations.values() if loc.block is not None})

    def get_aisles(self) -> List[int]:
        """Return sorted list of distinct aisle numbers."""
        return sorted({loc.aisle for loc in self._locations.values() if loc.aisle is not None})

    def get_levels(self) -> List[int]:
        """Return sorted list of distinct vertical levels."""
        return sorted({loc.level for loc in self._locations.values()})

    @property
    def bounding_box(self) -> BoundingBox3D:
        """Compute the 3D bounding box enclosing all locations in the layout."""
        return calculate_bounding_box(self._locations.values())

    @property
    def navigation_points(self) -> Dict[str, NavigationPoint]:
        """Return dictionary of navigation support points."""
        return self._navigation_points


# ==============================================================================
# 1. Mode A: Synthetic Warehouse (Paper Replication)
# ==============================================================================

class SyntheticWarehouse(WarehouseLayout):
    """
    Synthetic warehouse layout replicating the literature setup.

    Specifications from paper:
    - 3 Blocks: Block A (7 aisles), Block B (9 aisles), Block C (9 aisles). Total: 25 aisles.
    - 4 vertical levels (z = 1, 2, 3, 4).
    - 847 total storage units, 18 slots per unit -> 15,246 slots.
    - Depot located at (0.0, 0.0, 0.0) or (0.0, 0.0, 1.0).
    """

    def __init__(
        self,
        n_storage_units: int = 847,
        slots_per_unit: int = 18,
        levels: int = 4,
        depot_coords: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(mode="paper_replication", name="Synthetic_Warehouse_847")
        self.target_units = n_storage_units
        self.slots_per_unit = slots_per_unit
        self.levels_count = levels
        self.depot_coords = depot_coords
        self.config = config or {}
        self.build()

    def build(self) -> None:
        """Generate the synthetic 847 storage locations across 3 blocks and 25 aisles."""
        self._locations.clear()

        # Block definitions: A: 7 aisles, B: 9 aisles, C: 9 aisles
        block_specs = [
            ("A", 1, 7, 0.0),       # Block A: aisles 1-7, x_offset 0
            ("B", 8, 16, 80.0),     # Block B: aisles 8-16, x_offset 80m
            ("C", 17, 25, 180.0),   # Block C: aisles 17-25, x_offset 180m
        ]
        total_aisles = sum(end - start + 1 for _, start, end, _ in block_specs)  # 25

        # Distribute target units (847) across the 25 aisles evenly:
        # 847 // 25 = 33 base units per aisle, 847 % 25 = 22 aisles get +1 (34 units)
        base_per_aisle = self.target_units // total_aisles
        remainder = self.target_units % total_aisles

        aisle_idx = 0
        block_bounds: Dict[str, Tuple[float, float, float, float]] = {}

        aisle_width = 3.0       # width between shelf faces in meters
        aisle_spacing = 8.0     # pitch from one aisle center to next
        bay_pitch = 1.5         # distance between adjacent bays along the aisle
        level_height = 1.5      # height per vertical level in meters

        for blk_name, start_aisle, end_aisle, x_block_offset in block_specs:
            blk_xs = []
            blk_ys = []

            for aisle_num in range(start_aisle, end_aisle + 1):
                units_in_aisle = base_per_aisle + (1 if aisle_idx < remainder else 0)
                aisle_idx += 1

                aisle_x = x_block_offset + (aisle_num - start_aisle) * aisle_spacing

                # Generate bays along the aisle, alternating left and right sides and levels
                for u in range(units_in_aisle):
                    bay_index = u // (2 * self.levels_count)
                    level_idx = (u % self.levels_count) + 1
                    is_left = ((u // self.levels_count) % 2) == 0

                    side = "L" if is_left else "R"
                    x_pos = aisle_x - (aisle_width / 2.0) if is_left else aisle_x + (aisle_width / 2.0)
                    y_pos = 10.0 + bay_index * bay_pitch  # 10m front cross-aisle clearance
                    z_pos = float(level_idx) * level_height

                    loc_id = f"SYN-{blk_name}-{aisle_num:02d}-{u+1:03d}"
                    loc = Location(
                        id=loc_id,
                        x=round(x_pos, 2),
                        y=round(y_pos, 2),
                        z=round(z_pos, 2),
                        capacity=self.slots_per_unit,
                        block=blk_name,
                        aisle=aisle_num,
                        side=side,
                        level=level_idx,
                    )
                    self._locations[loc_id] = loc
                    blk_xs.append(x_pos)
                    blk_ys.append(y_pos)

            if blk_xs and blk_ys:
                block_bounds[blk_name] = (min(blk_xs), max(blk_xs), min(blk_ys), max(blk_ys))

        # Setup Depot
        self._depot = Location(
            id="DEPOT",
            x=self.depot_coords[0],
            y=self.depot_coords[1],
            z=self.depot_coords[2],
            capacity=0,
            block="DEPOT",
            aisle=0,
            level=int(self.depot_coords[2]),
        )

        # Generate synthetic navigation points for cross-aisles
        cross_aisle_y = [5.0, 30.0, 55.0]
        self._navigation_points = generate_synthetic_navigation_points(block_bounds, cross_aisle_y)


# ==============================================================================
# 2. Mode B: CSV Warehouse (Real Dataset)
# ==============================================================================

class CSVWarehouse(WarehouseLayout):
    """
    Real-world warehouse layout built from Storage_Location.csv.

    Specifications from real dataset:
    - 2,292 locations with coordinates (x, y, z).
    - 4 vertical levels (z = 1: 846, z = 2: 846, z = 3: 301, z = 4: 299).
    - 18 slots per location -> 41,256 total slot capacity.
    - 44 navigation support points from Support_Points_Navigation.csv.
    """

    def __init__(
        self,
        storage_locations_path: Optional[str] = None,
        navigation_path: Optional[str] = None,
        depot_coords: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        capacity_per_location: int = 18,
    ) -> None:
        super().__init__(mode="real_data_replication", name="CSV_Warehouse_2292")
        self.storage_locations_path = storage_locations_path
        self.navigation_path = navigation_path
        self.depot_coords = depot_coords
        self.capacity_per_location = capacity_per_location
        self.build()

    def build(self) -> None:
        """Load and parse real storage locations from Storage_Location.csv."""
        self._locations.clear()
        df = (
            load_storage_locations(self.storage_locations_path)
            if self.storage_locations_path
            else load_storage_locations()
        )

        for _, row in df.iterrows():
            loc_id = str(row["originalLocation"]).strip()
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])

            loc = Location.from_csv_row(
                loc_id=loc_id,
                x=x,
                y=y,
                z=z,
                capacity=self.capacity_per_location,
            )
            self._locations[loc.id] = loc

        # Setup Depot
        self._depot = Location(
            id="DEPOT",
            x=self.depot_coords[0],
            y=self.depot_coords[1],
            z=self.depot_coords[2],
            capacity=0,
            block="DEPOT",
            aisle=0,
            level=int(self.depot_coords[2]),
        )

        # Load navigation support points
        try:
            self._navigation_points = load_csv_navigation_points(self.navigation_path)
        except Exception:
            self._navigation_points = {}


# ==============================================================================
# 3. Factory Function
# ==============================================================================

def build_warehouse(
    mode: str = "real",
    config: Optional[Dict[str, Any]] = None,
) -> WarehouseLayout:
    """
    Factory function to instantiate the warehouse layout for the requested mode.

    Args:
        mode: 'paper' | 'paper_replication' for SyntheticWarehouse (847 units).
              'real' | 'real_data' for CSVWarehouse (2,292 locations).
        config: Optional configuration dictionary. If None, loaded from YAML.

    Returns:
        Instance of WarehouseLayout (SyntheticWarehouse or CSVWarehouse).
    """
    mode_clean = mode.lower().strip()

    if mode_clean in ("paper", "paper_replication", "synthetic", "mode_a"):
        if config is None:
            try:
                config = load_paper_config()
            except Exception:
                config = {}
        wh_cfg = config.get("warehouse", {})
        n_units = wh_cfg.get("n_storage_units", 847)
        slots = wh_cfg.get("slots_per_unit", 18)
        depot_dict = wh_cfg.get("depot", {"x": 0.0, "y": 0.0, "z": 1.0})
        depot_coords = (float(depot_dict.get("x", 0.0)), float(depot_dict.get("y", 0.0)), float(depot_dict.get("z", 1.0)))

        return SyntheticWarehouse(
            n_storage_units=n_units,
            slots_per_unit=slots,
            depot_coords=depot_coords,
            config=config,
        )

    elif mode_clean in ("real", "real_data", "real_data_replication", "csv", "mode_b"):
        if config is None:
            try:
                config = load_real_config()
            except Exception:
                config = {}
        paths = config.get("paths", {})
        loc_path = paths.get("storage_location_csv")
        nav_path = paths.get("navigation_points_csv")

        return CSVWarehouse(
            storage_locations_path=loc_path,
            navigation_path=nav_path,
        )

    else:
        raise ValueError(
            f"Unknown warehouse mode: '{mode}'. Expected 'paper' or 'real'."
        )
