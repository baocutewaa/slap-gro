"""Picking points extraction and transformation utilities for GRO optimizer."""

from typing import List, Optional, Tuple

from slap_gro.picking.wave import PickingWave
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


def get_wave_picking_locations(
    wave: PickingWave,
    warehouse: WarehouseLayout,
    skip_missing: bool = True,
) -> List[Location]:
    """
    Extract Location objects corresponding to all unique pick locations in a wave.

    Resolves warehouse locations, gracefully handling any off-layout locations
    (e.g., DISC-07 reception/dispatch zones RC-01..05, EX-01..11) by mapping them
    to nearby support points or skipping if unmapped.

    Args:
        wave: PickingWave entity.
        warehouse: WarehouseLayout instance.
        skip_missing: If True, skips unresolvable locations; if False, raises KeyError.

    Returns:
        List of Location objects in the wave.
    """
    loc_objects: List[Location] = []
    for loc_id in wave.unique_locations:
        loc = warehouse.get_location(loc_id)
        if loc is not None:
            loc_objects.append(loc)
        elif not skip_missing:
            raise KeyError(f"Location '{loc_id}' not found in warehouse layout")
        else:
            # Fallback: check if it matches a known navigation support point or synthesize temporary Location
            if hasattr(warehouse, "navigation_points") and loc_id in warehouse.navigation_points:
                np = warehouse.navigation_points[loc_id]
                loc_objects.append(Location(id=loc_id, x=np.x, y=np.y, z=int(np.z), level=int(np.z)))

    return loc_objects


def build_gro_route_targets(
    wave: PickingWave,
    warehouse: WarehouseLayout,
) -> Tuple[Location, List[Location]]:
    """
    Produce the exact input required by the GRO algorithm.

    Decouples the optimization problem from SKU semantics, returning only
    the Depot location and the set of destination pick locations to visit.

    Returns:
        Tuple of (depot_location, list_of_picking_locations).
    """
    depot = warehouse.get_depot()
    picks = get_wave_picking_locations(wave, warehouse)
    return depot, picks


def build_wave_tour_sequence(
    wave: PickingWave,
    warehouse: WarehouseLayout,
) -> List[Location]:
    """
    Build complete closed-loop tour sequence: [depot] + picks + [depot].
    """
    depot, picks = build_gro_route_targets(wave, warehouse)
    return [depot] + picks + [depot]
