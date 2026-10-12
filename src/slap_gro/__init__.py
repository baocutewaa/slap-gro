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
from slap_gro.routing.baseline import (
    BaseRoutingHeuristic,
    BaselineResult,
    NearestNeighborBaseline,
    OriginalCompanyRouteBaseline,
    ReconstructedFIFOBaseline,
    SShapeBaseline,
    evaluate_all_baselines,
    evaluate_baseline,
    get_baseline_heuristic,
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

from slap_gro.pipeline import (
    ExperimentResult,
    ExperimentRunner,
    SingleRunResult,
    build_paper_storage_policies,
    build_real_storage_policies,
    run_paper_experiment,
    run_real_experiment,
)

from slap_gro.evaluation import (
    DistanceMetrics,
    DistributionSummary,
    compare_policies_pairwise,
    compute_confidence_interval,
    compute_distance_metrics,
    compute_distribution_summary,
    compute_improvement,
    compute_policy_metrics,
    compute_relative_improvements,
    evaluate_experiment_run,
    export_metrics_table,
    plot_convergence_curve,
    plot_distance_boxplot,
    plot_distance_scatter,
    plot_picking_route_2d,
    plot_policy_comparison_bar,
    run_one_way_anova,
    run_tukey_hsd,
    test_normality_shapiro,
)

__all__ = [
    # Product Layer
    "Product",
    "ProductCatalog",
    "load_product_catalog",
    # Storage Matrix & Policies
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
    # Picking Layer
    "PickingItem",
    "PickingWave",
    "PickingWaveDataset",
    "get_wave_picking_locations",
    "build_gro_route_targets",
    "build_wave_tour_sequence",
    # Routing Engine
    "NavigationGraph",
    "distance",
    "compute_distance_matrix",
    "DistanceCache",
    "Route",
    "route_distance",
    "build_route",
    "evaluate_route",
    "evaluate_wave_route",
    # GRO Optimization Engine
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
    # Pipeline & Experiments
    "SingleRunResult",
    "ExperimentResult",
    "ExperimentRunner",
    "build_paper_storage_policies",
    "run_paper_experiment",
    "build_real_storage_policies",
    "run_real_experiment",
    # Baseline Routing Engine
    "BaselineResult",
    "BaseRoutingHeuristic",
    "OriginalCompanyRouteBaseline",
    "ReconstructedFIFOBaseline",
    "NearestNeighborBaseline",
    "SShapeBaseline",
    "get_baseline_heuristic",
    "evaluate_baseline",
    "evaluate_all_baselines",
    # Evaluation Layer
    "DistanceMetrics",
    "compute_distance_metrics",
    "compute_improvement",
    "compute_policy_metrics",
    "evaluate_experiment_run",
    "export_metrics_table",
    "DistributionSummary",
    "compute_distribution_summary",
    "compute_confidence_interval",
    "compute_relative_improvements",
    "compare_policies_pairwise",
    "test_normality_shapiro",
    "run_one_way_anova",
    "run_tukey_hsd",
    "plot_distance_boxplot",
    "plot_distance_scatter",
    "plot_policy_comparison_bar",
    "plot_convergence_curve",
    "plot_picking_route_2d",
]


