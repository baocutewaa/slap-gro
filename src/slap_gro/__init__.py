"""
SLAP-GRO Package: Storage Location Assignment Problem & Genetic Route Optimization.
"""

from slap_gro.data.product import Product, ProductCatalog, load_product_catalog
from slap_gro.picking.picking_points import (
    build_gro_route_targets,
    build_wave_tour_sequence,
    get_wave_picking_locations,
)
from slap_gro.picking.wave import PickingItem, PickingWave, PickingWaveDataset
from slap_gro.routing.distance import (
    DistanceCache,
    compute_distance_matrix,
    distance,
)
from slap_gro.routing.graph import NavigationGraph
from slap_gro.routing.route import (
    Route,
    build_route,
    evaluate_route,
    evaluate_wave_route,
    route_distance,
)
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.class_based import ClassBasedStoragePolicy
from slap_gro.storage.dedicated_storage import DedicatedStoragePolicy
from slap_gro.storage.hybrid_storage import HybridStoragePolicy
from slap_gro.storage.random_storage import RandomStoragePolicy
from slap_gro.storage.storage_matrix import (
    StorageMatrix,
    StorageUnit,
    StoredItem,
    parse_slot,
)
from slap_gro.storage import get_storage_policy
from slap_gro.gro import (
    Chromosome,
    FitnessEvaluator,
    GROConfig,
    GROOptimizer,
    GROResult,
    crossover,
    initialize_population,
    mutate,
    order_crossover,
    pmx_crossover,
    select_parents,
    single_point_crossover_repair,
    tournament_selection,
)

__all__ = [
    # Product Layer (Phase 5)
    "Product",
    "ProductCatalog",
    "load_product_catalog",
    # Storage Matrix & Policies (Phase 6 & 7)
    "StoredItem",
    "StorageUnit",
    "StorageMatrix",
    "parse_slot",
    "StoragePolicy",
    "RandomStoragePolicy",
    "ClassBasedStoragePolicy",
    "DedicatedStoragePolicy",
    "HybridStoragePolicy",
    "get_storage_policy",
    # Picking Layer (Phase 8)
    "PickingItem",
    "PickingWave",
    "PickingWaveDataset",
    "get_wave_picking_locations",
    "build_gro_route_targets",
    "build_wave_tour_sequence",
    # Routing Engine (Phase 9)
    "NavigationGraph",
    "distance",
    "compute_distance_matrix",
    "DistanceCache",
    "Route",
    "route_distance",
    "build_route",
    "evaluate_route",
    "evaluate_wave_route",
    # GRO Optimization Engine (Phase 10)
    "Chromosome",
    "FitnessEvaluator",
    "GROConfig",
    "GROResult",
    "GROOptimizer",
    "initialize_population",
    "tournament_selection",
    "select_parents",
    "crossover",
    "order_crossover",
    "pmx_crossover",
    "single_point_crossover_repair",
    "mutate",
]
