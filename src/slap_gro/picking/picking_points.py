from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from slap_gro.picking.wave import PickingWave
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.warehouse_layout import WarehouseLayout

if TYPE_CHECKING:
    from slap_gro.storage.base import StoragePolicy


def get_wave_picking_locations(
    wave: PickingWave,
    warehouse: WarehouseLayout,
    storage_policy: Optional["StoragePolicy"] = None,
    skip_missing: bool = True,
) -> List[Location]:
    """
    Extract Location objects corresponding to all unique pick locations in a wave.

    If a storage_policy is provided, resolves the wave's SKUs to the physical
    locations assigned by that policy. Otherwise, uses the wave's pre-assigned locations.

    Args:
        wave: PickingWave entity.
        warehouse: WarehouseLayout instance.
        storage_policy: Optional StoragePolicy to lookup SKU locations.
        skip_missing: If True, skips unresolvable locations; if False, raises KeyError.

    Returns:
        List of Location objects in the wave.
    """
    loc_objects: List[Location] = []

    if storage_policy is not None:
        seen_locs: Set[str] = set()
        for sku in wave.unique_skus:
            candidate_locations = storage_policy.get_locations_for_sku(sku)
            if candidate_locations:
                chosen_loc = candidate_locations[0]
                if chosen_loc not in seen_locs:
                    seen_locs.add(chosen_loc)
                    loc = warehouse.get_location(chosen_loc)
                    if loc is not None:
                        loc_objects.append(loc)
        return loc_objects

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
    storage_policy: Optional["StoragePolicy"] = None,
) -> Tuple[Location, List[Location]]:
    """
    Produce the exact input required by the GRO algorithm.

    Decouples the optimization problem from SKU semantics, returning only
    the Depot location and the set of destination pick locations to visit.

    Returns:
        Tuple of (depot_location, list_of_picking_locations).
    """
    depot = warehouse.get_depot()
    picks = get_wave_picking_locations(wave, warehouse, storage_policy=storage_policy)
    return depot, picks


def build_wave_tour_sequence(
    wave: PickingWave,
    warehouse: WarehouseLayout,
    storage_policy: Optional["StoragePolicy"] = None,
) -> List[Location]:
    """
    Build complete closed-loop tour sequence: [depot] + picks + [depot].
    """
    depot, picks = build_gro_route_targets(wave, warehouse, storage_policy=storage_policy)
    return [depot] + picks + [depot]
