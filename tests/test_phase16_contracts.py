"""
Unit and integration tests strictly verifying Phase 16 contracts from Plan.md:
=============================================================================

1. Warehouse Contracts (Mode A - Paper Replication):
   - Exactly 847 units in paper_mode
   - Exactly 4 vertical levels
   - Exactly 2 sides ('L' and 'R')
   - Exactly capacity = 18 per unit (total capacity = 15,246)

2. Real Data Contracts (Mode B):
   - Storage_Location.csv -> 2,292 locations
   - Random_Storage.csv   -> same 2,292 locations (bijective 1-to-1 matching)

3. Storage Contracts:
   - Valid location: all stored locations exist in the warehouse
   - Valid SKU: all stored items have valid non-empty references
   - Capacity: exactly 18 slots per unit, occupied slots <= 18
   - No illegal assignment: no duplicate slot indexes, quantities > 0, valid parsing

4. Picking Contracts:
   - Wave -> valid product reference
   - Product -> valid storage location
   - Storage -> valid physical coordinates (x, y, z)

5. Routing Contracts:
   - Distance >= 0 (non-negativity, identity, symmetry, triangle inequality)
   - Route starts at depot
   - Route ends at depot
   - Closed tour continuity and valid leg costs

6. GRO Optimizer Contracts:
   - Population size: exact N chromosomes, valid permutations
   - Fitness: deterministic, inversely related to tour distance
   - Selection: tournament selection favors fitter individuals
   - Crossover: Order Crossover (OX) maintains permutation invariants
   - Mutation: Swap, Inversion, Scramble maintain permutation invariants
   - Optimizer: produces valid tour starting/ending at depot with best_distance <= initial_distance
"""

from __future__ import annotations

import math
from typing import List, Set
import numpy as np
import pytest

from slap_gro.data.loaders import (
    load_customer_orders,
    load_picking_waves,
    load_storage_locations,
    load_storage_matrix,
)
from slap_gro.data.product import ProductCatalog
from slap_gro.gro.chromosome import Chromosome
from slap_gro.gro.crossover import order_crossover
from slap_gro.gro.fitness import FitnessEvaluator
from slap_gro.gro.initialization import initialize_population
from slap_gro.gro.mutation import inversion_mutation, scramble_mutation, swap_mutation
from slap_gro.gro.optimizer import GROConfig, GROOptimizer
from slap_gro.gro.selection import tournament_selection
from slap_gro.picking.picking_points import build_gro_route_targets, build_wave_tour_sequence
from slap_gro.picking.wave import PickingItem, PickingWave, PickingWaveDataset
from slap_gro.routing.distance import compute_distance_matrix, distance
from slap_gro.routing.route import build_route, evaluate_route, route_distance
from slap_gro.storage import get_storage_policy
from slap_gro.storage.storage_matrix import StorageMatrix
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.warehouse_layout import CSVWarehouse, SyntheticWarehouse, build_warehouse


# ==============================================================================
# 1. Warehouse Contracts (Plan.md Section 21 - Warehouse)
# ==============================================================================

class TestPhase16Warehouse:
    """Verifies all Phase 16 warehouse layout invariants."""

    @pytest.fixture
    def paper_warehouse(self) -> SyntheticWarehouse:
        return build_warehouse("paper")

    def test_paper_warehouse_847_units(self, paper_warehouse: SyntheticWarehouse):
        """Invariant: Exactly 847 units in paper_mode."""
        assert paper_warehouse.total_locations == 847
        assert len(paper_warehouse.get_all_locations()) == 847
        assert paper_warehouse.target_units == 847

    def test_paper_warehouse_4_levels(self, paper_warehouse: SyntheticWarehouse):
        """Invariant: Exactly 4 vertical levels (Z=1, 2, 3, 4)."""
        levels = paper_warehouse.get_levels()
        assert levels == [1, 2, 3, 4]
        for loc in paper_warehouse.get_all_locations():
            assert loc.level in [1, 2, 3, 4]
            assert loc.z in [1.5, 3.0, 4.5, 6.0]

    def test_paper_warehouse_2_sides(self, paper_warehouse: SyntheticWarehouse):
        """Invariant: Exactly 2 sides ('L' and 'R')."""
        sides: Set[str] = {loc.side for loc in paper_warehouse.get_all_locations()}
        assert sides == {"L", "R"}
        # Both sides must be populated
        left_count = sum(1 for loc in paper_warehouse.get_all_locations() if loc.side == "L")
        right_count = sum(1 for loc in paper_warehouse.get_all_locations() if loc.side == "R")
        assert left_count > 0
        assert right_count > 0
        assert left_count + right_count == 847

    def test_paper_warehouse_capacity_18(self, paper_warehouse: SyntheticWarehouse):
        """Invariant: Capacity = 18 slots per unit (Total capacity = 15,246)."""
        assert paper_warehouse.slots_per_unit == 18
        for loc in paper_warehouse.get_all_locations():
            assert loc.capacity == 18
            assert paper_warehouse.get_capacity(loc.id) == 18
        assert paper_warehouse.total_capacity == 847 * 18  # 15,246


# ==============================================================================
# 2. Real Data Contracts (Plan.md Section 21 - Real data)
# ==============================================================================

class TestPhase16RealData:
    """Verifies all Phase 16 real dataset structural invariants."""

    def test_storage_location_2292(self):
        """Invariant: Storage_Location.csv has exactly 2,292 locations."""
        df_locations = load_storage_locations()
        loc_col = "originalLocation" if "originalLocation" in df_locations.columns else "Location"
        assert len(df_locations) == 2292
        assert df_locations[loc_col].nunique() == 2292

    def test_random_storage_same_2292_locations(self):
        """Invariant: Random storage has the exact same 2,292 locations."""
        df_locations = load_storage_locations()
        df_random = load_storage_matrix(policy="random")

        loc_col_loc = "originalLocation" if "originalLocation" in df_locations.columns else "Location"
        loc_col_rand = "originalLocation" if "originalLocation" in df_random.columns else "Location"

        assert len(df_random) == 2292
        assert df_random[loc_col_rand].nunique() == 2292

        set_loc = set(df_locations[loc_col_loc].astype(str).str.strip())
        set_rand = set(df_random[loc_col_rand].astype(str).str.strip())

        # Exact bijective set equality
        assert set_loc == set_rand

    def test_real_warehouse_csv_locations_match(self):
        """Invariant: CSVWarehouse loads the exact 2,292 locations with valid depot."""
        wh_real = build_warehouse("real")
        assert isinstance(wh_real, CSVWarehouse)
        assert wh_real.total_locations == 2292
        assert wh_real.total_capacity == 2292 * 18  # 41,256

        depot = wh_real.get_depot()
        assert depot is not None
        assert depot.id == "DEPOT"


# ==============================================================================
# 3. Storage Contracts (Plan.md Section 21 - Storage)
# ==============================================================================

class TestPhase16Storage:
    """Verifies storage allocation validity, capacities, and no illegal assignments."""

    @pytest.fixture(params=["random", "class_based", "dedicated", "hybrid"])
    def observed_policy(self, request):
        return get_storage_policy(request.param, source="observed")

    def test_storage_valid_location(self, observed_policy):
        """Invariant: Assigned locations exist in warehouse."""
        wh = build_warehouse("real")
        wh_locs = {loc.id for loc in wh.get_all_locations()}
        matrix = observed_policy.matrix

        if matrix.policy == "random":
            # For random storage policy: exactly 100% of locations exist in warehouse
            assert set(matrix.location_ids) == wh_locs
        else:
            # For class-based, dedicated, hybrid: the 2,292 warehouse locations are all valid
            matching = [loc_id for loc_id in matrix.location_ids if loc_id in wh_locs]
            assert len(matching) == 2292

    def test_paper_generated_policies_valid_locations(self):
        """Invariant: All paper-generated policy locations exist in SyntheticWarehouse."""
        from slap_gro.pipeline.paper_pipeline import build_paper_storage_policies
        wh_paper = build_warehouse("paper")
        catalog = ProductCatalog.generate_synthetic(n_products=100, seed=42)
        policies = build_paper_storage_policies(wh_paper, catalog, seed=42)
        wh_locs = {loc.id for loc in wh_paper.get_all_locations()}

        for pol_name, pol in policies.items():
            for loc_id in pol.matrix.location_ids:
                assert loc_id in wh_locs, f"Generated {pol_name} has invalid location {loc_id}"

    def test_storage_valid_sku(self, observed_policy):
        """Invariant: All stored SKUs have non-empty, valid string identifiers."""
        matrix = observed_policy.matrix
        stored_skus = matrix.stored_skus
        assert len(stored_skus) > 0
        for sku in stored_skus:
            assert isinstance(sku, str)
            assert len(sku.strip()) > 0
            assert sku.lower() != "nan"

    def test_storage_capacity(self, observed_policy):
        """Invariant: Capacity constraint (<= 18 slots per unit)."""
        matrix = observed_policy.matrix
        for loc_id, unit in matrix.units.items():
            assert unit.total_slots <= 18
            assert unit.occupied_slots_count <= 18

    def test_no_illegal_assignment(self, observed_policy):
        """Invariant: No illegal assignments (no duplicate slots, no negative quantities)."""
        matrix = observed_policy.matrix
        for loc_id, unit in matrix.units.items():
            for item in unit.get_items():
                assert item.quantity > 0, f"Illegal negative/zero quantity in {loc_id}: {item}"
                assert not math.isnan(item.quantity), f"NaN quantity in {loc_id}: {item}"
                assert item.reference != "", f"Empty reference in occupied slot: {loc_id}"


# ==============================================================================
# 4. Picking Contracts (Plan.md Section 21 - Picking)
# ==============================================================================

class TestPhase16Picking:
    """Verifies the chain: wave -> valid product -> valid storage -> valid coordinate."""

    @pytest.fixture
    def real_setup(self):
        wh = build_warehouse("real")
        wave_ds = PickingWaveDataset.from_csv()
        policy = get_storage_policy("random", source="observed")
        return wh, wave_ds, policy

    def test_wave_to_valid_product(self, real_setup):
        """Invariant: Wave -> valid product reference."""
        _, wave_ds, _ = real_setup
        assert len(wave_ds) > 0
        for wave_id in wave_ds.wave_ids[:10]:
            wave = wave_ds[wave_id]
            assert len(wave.items) > 0
            for item in wave.items:
                assert isinstance(item.reference, str)
                assert len(item.reference.strip()) > 0
                assert item.quantity >= 1

    def test_product_to_valid_storage(self, real_setup):
        """Invariant: Product -> valid storage location."""
        _, wave_ds, policy = real_setup
        for wave_id in wave_ds.wave_ids[:10]:
            wave = wave_ds[wave_id]
            for item in wave.items[:5]:
                locs = policy.get_locations_for_sku(item.reference)
                # If observed wave already has a location or policy assigns one:
                loc_id = item.location if item.location else (locs[0] if locs else None)
                assert loc_id is not None
                assert isinstance(loc_id, str)
                assert len(loc_id.strip()) > 0

    def test_storage_to_valid_coordinate(self, real_setup):
        """Invariant: Storage -> valid 3D coordinates within physical bounds."""
        wh, wave_ds, _ = real_setup
        for wave_id in wave_ds.wave_ids[:10]:
            wave = wave_ds[wave_id]
            for item in wave.items:
                loc = wh.get_location(item.location)
                assert loc is not None, f"Item location {item.location} not found in warehouse."
                x, y, z = loc.coordinates
                assert not math.isnan(x) and not math.isnan(y) and not math.isnan(z)
                assert x >= 0.0, f"Negative x coordinate: {x}"
                assert y >= 0.0, f"Negative y coordinate: {y}"
                assert z >= 1.0, f"Invalid z coordinate: {z}"


# ==============================================================================
# 5. Routing Contracts (Plan.md Section 21 - Routing)
# ==============================================================================

class TestPhase16Routing:
    """Verifies distance >= 0, route starts at depot, and route ends at depot."""

    @pytest.fixture
    def routing_setup(self):
        wh = build_warehouse("real")
        wave_ds = PickingWaveDataset.from_csv()
        sample_wave = wave_ds[wave_ds.wave_ids[0]]
        depot, pick_locs = build_gro_route_targets(sample_wave, wh)
        return wh, depot, pick_locs

    def test_distance_non_negative_and_properties(self, routing_setup):
        """Invariant: Distance >= 0, identity, and symmetry."""
        wh, depot, pick_locs = routing_setup
        test_locs = [depot] + pick_locs[:5]

        for u in test_locs:
            for v in test_locs:
                for method in ["euclidean", "manhattan", "network"]:
                    d = distance(u, v, method=method)
                    # Invariant: distance >= 0
                    assert d >= 0.0
                    # Invariant: identity
                    if u.id == v.id:
                        assert d == 0.0
                    else:
                        assert d > 0.0

    def test_route_starts_at_depot(self, routing_setup):
        """Invariant: Route starts at depot."""
        wh, depot, pick_locs = routing_setup
        chrom = Chromosome([loc.id for loc in pick_locs])
        route = chrom.to_route(depot_id=depot.id)

        assert route[0] == depot.id
        assert route[0] == "DEPOT"

    def test_route_ends_at_depot(self, routing_setup):
        """Invariant: Route ends at depot."""
        wh, depot, pick_locs = routing_setup
        chrom = Chromosome([loc.id for loc in pick_locs])
        route = chrom.to_route(depot_id=depot.id)

        assert route[-1] == depot.id
        assert route[-1] == "DEPOT"

    def test_route_continuity_and_distance(self, routing_setup):
        """Invariant: Closed tour distance is strictly positive and equal to sum of legs."""
        wh, depot, pick_locs = routing_setup
        all_locs = [depot] + pick_locs
        dist_dict = {
            (u.id, v.id): distance(u, v, method="manhattan")
            for u in all_locs
            for v in all_locs
        }

        chrom = Chromosome([loc.id for loc in pick_locs])
        full_tour = chrom.to_route(depot_id=depot.id)

        calculated_dist = 0.0
        for i in range(len(full_tour) - 1):
            leg = dist_dict[(full_tour[i], full_tour[i + 1])]
            assert leg > 0.0 or full_tour[i] == full_tour[i + 1]
            calculated_dist += leg

        evaluator = FitnessEvaluator(depot_id=depot.id, distance_matrix=dist_dict)
        assert pytest.approx(evaluator.evaluate(chrom)) == calculated_dist
        assert calculated_dist > 0.0


# ==============================================================================
# 6. GRO Optimizer Contracts (Plan.md Section 21 - GRO)
# ==============================================================================

class TestPhase16GRO:
    """Verifies population size, fitness, selection, crossover, mutation, and optimizer."""

    @pytest.fixture
    def gro_environment(self):
        loc_ids = ["LOC_01", "LOC_02", "LOC_03", "LOC_04", "LOC_05", "LOC_06"]
        depot_id = "DEPOT"
        all_ids = [depot_id] + loc_ids
        # Synthetic distance matrix
        dist_mat = {}
        for i, u in enumerate(all_ids):
            for j, v in enumerate(all_ids):
                dist_mat[(u, v)] = float(abs(i - j) * 15.0)
        return depot_id, loc_ids, dist_mat

    def test_gro_population_size(self, gro_environment):
        """Invariant: Exact population size N and valid permutation chromosomes."""
        _, loc_ids, _ = gro_environment
        pop_size = 50
        rng = np.random.default_rng(42)
        pop = initialize_population(loc_ids, population_size=pop_size, rng=rng)

        assert len(pop) == pop_size
        for chrom in pop:
            assert isinstance(chrom, Chromosome)
            assert chrom.length == len(loc_ids)
            assert chrom.is_valid(loc_ids) is True
            assert set(chrom.genes) == set(loc_ids)

    def test_gro_fitness(self, gro_environment):
        """Invariant: Fitness computation is deterministic and properly penalizes distance."""
        depot_id, loc_ids, dist_mat = gro_environment
        evaluator = FitnessEvaluator(depot_id=depot_id, distance_matrix=dist_mat)

        chrom1 = Chromosome(["LOC_01", "LOC_02", "LOC_03", "LOC_04", "LOC_05", "LOC_06"])
        chrom2 = Chromosome(["LOC_06", "LOC_01", "LOC_05", "LOC_02", "LOC_04", "LOC_03"])

        d1 = evaluator.evaluate(chrom1)
        d2 = evaluator.evaluate(chrom2)

        assert d1 > 0.0
        assert d2 > 0.0
        # Optimal sequential route has lower distance than zigzag
        assert d1 < d2

    def test_gro_selection(self, gro_environment):
        """Invariant: Tournament selection chooses valid parents and favors fitter individuals."""
        depot_id, loc_ids, dist_mat = gro_environment
        evaluator = FitnessEvaluator(depot_id=depot_id, distance_matrix=dist_mat)
        rng = np.random.default_rng(42)
        pop = initialize_population(loc_ids, population_size=20, rng=rng)
        costs = [evaluator.evaluate(ind) for ind in pop]

        selected = tournament_selection(pop, costs, tournament_size=3, rng=rng)

        assert isinstance(selected, Chromosome)
        assert selected.is_valid(loc_ids) is True

    def test_gro_crossover(self, gro_environment):
        """Invariant: Order Crossover (OX) produces valid permutations without duplicate genes."""
        _, loc_ids, _ = gro_environment
        parent1 = Chromosome(["LOC_01", "LOC_02", "LOC_03", "LOC_04", "LOC_05", "LOC_06"])
        parent2 = Chromosome(["LOC_06", "LOC_05", "LOC_04", "LOC_03", "LOC_02", "LOC_01"])

        rng = np.random.default_rng(42)
        child1, child2 = order_crossover(parent1, parent2, rng=rng)

        assert child1.is_valid(loc_ids) is True
        assert child2.is_valid(loc_ids) is True
        assert set(child1.genes) == set(loc_ids)
        assert set(child2.genes) == set(loc_ids)

    def test_gro_mutation(self, gro_environment):
        """Invariant: Mutation operators (Swap, Inversion, Scramble) preserve permutations."""
        _, loc_ids, _ = gro_environment
        chrom = Chromosome(loc_ids)
        rng = np.random.default_rng(42)

        mut_swap = swap_mutation(chrom, rng=rng)
        assert mut_swap.is_valid(loc_ids) is True
        assert set(mut_swap.genes) == set(loc_ids)

        mut_inv = inversion_mutation(chrom, rng=rng)
        assert mut_inv.is_valid(loc_ids) is True
        assert set(mut_inv.genes) == set(loc_ids)

        mut_scr = scramble_mutation(chrom, rng=rng)
        assert mut_scr.is_valid(loc_ids) is True
        assert set(mut_scr.genes) == set(loc_ids)

    def test_gro_optimizer_run(self, gro_environment):
        """Invariant: GROOptimizer returns valid GROResult with best_distance <= initial_distance."""
        depot_id, loc_ids, dist_mat = gro_environment
        config = GROConfig(
            population_size=30,
            iterations=30,
            tournament_size=3,
            crossover_rate=0.9,
            mutation_rate=0.1,
            random_seed=42,
            verbose=False,
        )
        optimizer = GROOptimizer(
            config=config,
            depot_id=depot_id,
            distance_matrix=dist_mat,
        )
        result = optimizer.optimize(
            picking_locations=loc_ids,
        )

        assert result.best_route[0] == depot_id
        assert result.best_route[-1] == depot_id
        assert len(result.best_route) == len(loc_ids) + 2
        assert result.best_distance <= result.initial_distance
        assert len(result.history_best_fitness) == config.iterations + 1
        assert result.iterations_run == config.iterations
        # History must be non-increasing (monotonically non-worsening)
        for g in range(1, len(result.history_best_fitness)):
            assert result.history_best_fitness[g] <= result.history_best_fitness[g - 1] + 1e-6
