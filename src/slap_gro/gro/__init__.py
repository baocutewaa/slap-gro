"""
GRO (Genetic Route Optimization) module.

Triển khai đầy đủ thuật toán di truyền tối ưu hóa hành trình nhặt hàng theo đặc tả giải thuật GRO.
"""

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

__all__ = [
    # Chromosome representation
    "Chromosome",
    # Fitness evaluation
    "FitnessEvaluator",
    # Initialization
    "initialize_population",
    # Selection operators
    "tournament_selection",
    "select_parents",
    # Crossover operators (Mode A & Mode B)
    "crossover",
    "order_crossover",
    "pmx_crossover",
    "single_point_crossover_repair",
    # Mutation operators
    "mutate",
    "swap_mutation",
    "inversion_mutation",
    "scramble_mutation",
    # Optimizer engine & configs
    "GROConfig",
    "GROResult",
    "GROOptimizer",
]
