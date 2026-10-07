"""
Tests for GRO (Genetic Route Optimization) module - Phase 10.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from slap_gro.gro.chromosome import Chromosome
from slap_gro.gro.crossover import (
    crossover,
    order_crossover,
    pmx_crossover,
    single_point_crossover_repair,
)
from slap_gro.gro.fitness import FitnessEvaluator
from slap_gro.gro.initialization import initialize_population
from slap_gro.gro.mutation import (
    inversion_mutation,
    mutate,
    scramble_mutation,
    swap_mutation,
)
from slap_gro.gro.optimizer import GROConfig, GROOptimizer, GROResult
from slap_gro.gro.selection import select_parents, tournament_selection
from slap_gro.picking.picking_points import build_gro_route_targets
from slap_gro.picking.wave import PickingItem, PickingWave
from slap_gro.routing.distance import distance
from slap_gro.warehouse.warehouse_layout import SyntheticWarehouse


class TestChromosome:
    def test_chromosome_initialization_and_properties(self):
        locs = ["L1", "L2", "L3", "L4"]
        chrom = Chromosome(locs)

        assert chrom.length == 4
        assert len(chrom) == 4
        assert chrom.genes == ("L1", "L2", "L3", "L4")
        assert chrom[0] == "L1"
        assert chrom[1:3] == ("L2", "L3")
        assert list(chrom) == locs

    def test_chromosome_to_route(self):
        chrom = Chromosome(["A", "B", "C"])
        route = chrom.to_route(depot_id="DEPOT")
        assert route == ["DEPOT", "A", "B", "C", "DEPOT"]

    def test_chromosome_validation(self):
        chrom = Chromosome(["A", "B", "C"])
        assert chrom.is_valid(["A", "B", "C"]) is True
        assert chrom.is_valid(["C", "A", "B"]) is True
        assert chrom.is_valid(["A", "B"]) is False
        assert chrom.is_valid(["A", "B", "C", "D"]) is False

        # Chromosome with duplicates
        duplicate_chrom = Chromosome(["A", "A", "B"])
        assert duplicate_chrom.is_valid(["A", "B"]) is False

    def test_chromosome_immutability_and_hashing(self):
        chrom1 = Chromosome(["A", "B"])
        chrom2 = Chromosome(["A", "B"])
        chrom3 = Chromosome(["B", "A"])

        assert chrom1 == chrom2
        assert chrom1 != chrom3
        assert hash(chrom1) == hash(chrom2)
        s = {chrom1, chrom2, chrom3}
        assert len(s) == 2


class TestFitnessEvaluator:
    @pytest.fixture
    def dummy_distance_matrix(self):
        # 1D line: DEPOT=0, A=10, B=25, C=40
        coords = {"DEPOT": 0.0, "A": 10.0, "B": 25.0, "C": 40.0}
        mat = {}
        for u, u_x in coords.items():
            for v, v_x in coords.items():
                mat[(u, v)] = abs(u_x - v_x)
        return mat

    def test_fitness_evaluation_closed_tour(self, dummy_distance_matrix):
        evaluator = FitnessEvaluator(depot_id="DEPOT", distance_matrix=dummy_distance_matrix)

        # Tour 1: DEPOT -> A (10) -> B (15) -> C (15) -> DEPOT (40) = 80
        chrom1 = Chromosome(["A", "B", "C"])
        assert evaluator.evaluate(chrom1) == 80.0

        # Tour 2: DEPOT -> C (40) -> B (15) -> A (15) -> DEPOT (10) = 80
        chrom2 = Chromosome(["C", "B", "A"])
        assert evaluator.evaluate(chrom2) == 80.0

        # Tour 3: DEPOT -> B (25) -> A (15) -> C (30) -> DEPOT (40) = 110 (worse)
        chrom3 = Chromosome(["B", "A", "C"])
        assert evaluator.evaluate(chrom3) == 110.0

    def test_fitness_evaluation_edge_cases(self, dummy_distance_matrix):
        evaluator = FitnessEvaluator(depot_id="DEPOT", distance_matrix=dummy_distance_matrix)

        # 0 locations
        assert evaluator.evaluate(Chromosome([])) == 0.0

        # 1 location: DEPOT -> A (10) -> DEPOT (10) = 20
        assert evaluator.evaluate(Chromosome(["A"])) == 20.0

    def test_evaluate_route_object(self, dummy_distance_matrix):
        evaluator = FitnessEvaluator(depot_id="DEPOT", distance_matrix=dummy_distance_matrix)
        route_obj = evaluator.evaluate_route(Chromosome(["A", "B"]))

        assert route_obj.stops == ["DEPOT", "A", "B", "DEPOT"]
        assert route_obj.is_closed is True
        assert route_obj.stops_count == 2
        assert route_obj.total_distance == 10.0 + 15.0 + 25.0


class TestInitialization:
    def test_initialize_population_shape_and_validity(self):
        locs = ["L1", "L2", "L3", "L4", "L5"]
        pop_size = 50
        rng = np.random.default_rng(123)

        pop = initialize_population(locs, population_size=pop_size, rng=rng, include_canonical=True)

        assert len(pop) == pop_size
        # First individual is canonical order
        assert pop[0].genes == tuple(locs)

        # All individuals must be valid permutations of locs
        for ind in pop:
            assert ind.is_valid(locs) is True

    def test_initialize_population_reproducibility(self):
        locs = ["L1", "L2", "L3"]
        pop1 = initialize_population(locs, 10, rng=np.random.default_rng(42))
        pop2 = initialize_population(locs, 10, rng=np.random.default_rng(42))

        assert [ind.genes for ind in pop1] == [ind.genes for ind in pop2]


class TestSelection:
    def test_tournament_selection_minimization(self):
        pop = [
            Chromosome(["A", "B"]),  # idx 0: dist 100
            Chromosome(["B", "A"]),  # idx 1: dist 50
            Chromosome(["C", "D"]),  # idx 2: dist 80
            Chromosome(["D", "C"]),  # idx 3: dist 30
        ]
        fitness = [100.0, 50.0, 80.0, 30.0]

        # If tournament size == 4, candidate with 30.0 must always win
        rng = np.random.default_rng(42)
        winner = tournament_selection(pop, fitness, tournament_size=4, rng=rng)
        assert winner.genes == ("D", "C")

    def test_select_parents(self):
        pop = [Chromosome(["A", "B"]), Chromosome(["B", "A"]), Chromosome(["A", "C"])]
        fitness = [10.0, 20.0, 30.0]
        parents = select_parents(pop, fitness, n_pairs=5, tournament_size=2)

        assert len(parents) == 5
        for p1, p2 in parents:
            assert isinstance(p1, Chromosome)
            assert isinstance(p2, Chromosome)


class TestCrossover:
    @pytest.fixture
    def parent_pair(self):
        p1 = Chromosome(["1", "2", "3", "4", "5", "6", "7", "8"])
        p2 = Chromosome(["8", "7", "6", "5", "4", "3", "2", "1"])
        return p1, p2

    def test_order_crossover_mode_b(self, parent_pair):
        p1, p2 = parent_pair
        rng = np.random.default_rng(42)
        c1, c2 = order_crossover(p1, p2, rng)

        assert c1.is_valid(p1.genes) is True
        assert c2.is_valid(p2.genes) is True
        assert len(c1) == len(p1)
        assert len(c2) == len(p2)

    def test_pmx_crossover_mode_b(self, parent_pair):
        p1, p2 = parent_pair
        rng = np.random.default_rng(42)
        c1, c2 = pmx_crossover(p1, p2, rng)

        assert c1.is_valid(p1.genes) is True
        assert c2.is_valid(p2.genes) is True

    def test_single_point_crossover_repair_mode_a(self, parent_pair):
        p1, p2 = parent_pair
        rng = np.random.default_rng(42)
        c1, c2 = single_point_crossover_repair(p1, p2, rng)

        assert c1.is_valid(p1.genes) is True
        assert c2.is_valid(p2.genes) is True

    def test_crossover_probability(self, parent_pair):
        p1, p2 = parent_pair
        # With Pc=0.0, parents returned unchanged
        rng = np.random.default_rng(42)
        c1, c2 = crossover(p1, p2, pc=0.0, rng=rng)
        assert c1.genes == p1.genes
        assert c2.genes == p2.genes


class TestMutation:
    def test_swap_mutation(self):
        chrom = Chromosome(["1", "2", "3", "4", "5"])
        rng = np.random.default_rng(42)
        mutated = swap_mutation(chrom, rng)

        assert mutated.is_valid(chrom.genes) is True
        # Check that exactly 2 elements swapped
        diff_count = sum(1 for a, b in zip(chrom.genes, mutated.genes) if a != b)
        assert diff_count == 2

    def test_inversion_mutation(self):
        chrom = Chromosome(["1", "2", "3", "4", "5"])
        rng = np.random.default_rng(42)
        mutated = inversion_mutation(chrom, rng)

        assert mutated.is_valid(chrom.genes) is True

    def test_scramble_mutation(self):
        chrom = Chromosome(["1", "2", "3", "4", "5", "6"])
        rng = np.random.default_rng(42)
        mutated = scramble_mutation(chrom, rng)

        assert mutated.is_valid(chrom.genes) is True

    def test_mutation_probability(self):
        chrom = Chromosome(["1", "2", "3"])
        # With Pm=0.0, chromosome is identical
        rng = np.random.default_rng(42)
        mutated = mutate(chrom, pm=0.0, rng=rng)
        assert mutated.genes == chrom.genes


class TestGROOptimizer:
    @pytest.fixture
    def circle_tsp_problem(self):
        """
        8 points arranged on a circle of radius 100 centered at (0, 0), depot at (0, 0).
        Optimal tour visits points in angular order!
        """
        n = 8
        depot = "DEPOT"
        points = {}
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = 100.0 * math.cos(angle)
            y = 100.0 * math.sin(angle)
            points[f"P{i}"] = (x, y)

        points[depot] = (0.0, 0.0)

        # Distance matrix
        mat = {}
        for u, (ux, uy) in points.items():
            for v, (vx, vy) in points.items():
                mat[(u, v)] = math.hypot(ux - vx, uy - vy)

        # Scrambled order as input
        scrambled_locs = ["P0", "P4", "P1", "P5", "P2", "P6", "P3", "P7"]
        return depot, scrambled_locs, mat

    def test_gro_optimization_improvement(self, circle_tsp_problem):
        depot, locs, mat = circle_tsp_problem

        config = GROConfig(
            population_size=100,
            iterations=150,
            tournament_size=3,
            crossover_rate=0.90,
            mutation_rate=0.10,
            crossover_method="ox",
            mutation_method="inversion",
            elitism=True,
            random_seed=42,
        )

        optimizer = GROOptimizer(config=config, distance_matrix=mat, depot_id=depot)
        result = optimizer.optimize(locs)

        assert isinstance(result, GROResult)
        assert result.best_route[0] == depot
        assert result.best_route[-1] == depot
        assert len(result.best_route) == len(locs) + 2
        # Must show significant improvement over the scrambled initial order
        assert result.best_distance < result.initial_distance
        assert result.improvement_meters > 0
        assert result.improvement_pct > 10.0
        assert len(result.history_best_fitness) == result.iterations_run + 1

    def test_gro_mode_a_single_point_repair(self, circle_tsp_problem):
        depot, locs, mat = circle_tsp_problem

        config = GROConfig(
            population_size=50,
            iterations=50,
            crossover_method="single_point_repair",
            random_seed=42,
        )

        optimizer = GROOptimizer(config=config, distance_matrix=mat, depot_id=depot)
        result = optimizer.optimize(locs)

        assert result.best_chromosome.is_valid(locs) is True
        assert result.best_distance <= result.initial_distance

    def test_gro_config_loading_from_yaml(self):
        cfg_paper = GROConfig.from_yaml("configs/paper_replication.yaml")
        assert cfg_paper.population_size == 1000
        assert cfg_paper.iterations == 1000
        assert cfg_paper.tournament_size == 3
        assert cfg_paper.crossover_rate == 0.90
        assert cfg_paper.mutation_rate == 0.08
        assert cfg_paper.elitism is True

    def test_gro_edge_cases(self):
        depot = "DEPOT"
        mat = {("DEPOT", "A"): 15.0, ("A", "DEPOT"): 15.0}
        optimizer = GROOptimizer(distance_matrix=mat, depot_id=depot)

        # 0 locations
        res0 = optimizer.optimize([])
        assert res0.best_distance == 0.0
        assert res0.best_route == [depot, depot]

        # 1 location
        res1 = optimizer.optimize(["A"])
        assert res1.best_distance == 30.0
        assert res1.best_route == [depot, "A", depot]

    def test_gro_integration_with_synthetic_warehouse(self):
        wh = SyntheticWarehouse()
        # Take 5 locations from warehouse
        loc_keys = [loc.id for loc in wh.get_all_locations()[:5]]

        # Distance function using distance(u, v) with depot support
        def dist_fn(u: str, v: str) -> float:
            loc_u = wh.get_depot() if u == "DEPOT" else wh.get_location(u)
            loc_v = wh.get_depot() if v == "DEPOT" else wh.get_location(v)
            return distance(loc_u, loc_v, method="euclidean")

        config = GROConfig(
            population_size=30,
            iterations=30,
            random_seed=42,
        )

        optimizer = GROOptimizer(config=config, distance_fn=dist_fn, depot_id="DEPOT")
        result = optimizer.optimize(loc_keys)

        assert result.best_chromosome.is_valid(loc_keys) is True
        assert result.best_route[0] == "DEPOT"
        assert result.best_route[-1] == "DEPOT"
        assert result.best_distance > 0.0
