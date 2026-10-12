"""
Population initialization module for GRO.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from slap_gro.gro.chromosome import Chromosome


def initialize_population(
    locations: Sequence[str],
    population_size: int,
    rng: np.random.Generator | None = None,
    include_canonical: bool = True,
) -> list[Chromosome]:
    """
    Khởi tạo quần thể ban đầu gồm `population_size` cá thể hoán vị.
    
    Args:
        locations: Danh sách các vị trí nhặt hàng cần ghé thăm.
        population_size: Kích thước quần thể (số lượng cá thể).
        rng: Bộ sinh số ngẫu nhiên NumPy Generator (tuân thủ nguyên tắc reproducibility).
        include_canonical: Nếu True, cá thể đầu tiên là thứ tự ban đầu của danh sách;
                           các cá thể còn lại được xáo trộn ngẫu nhiên.
                           
    Returns:
        Danh sách các đối tượng Chromosome.
    """
    if population_size <= 0:
        raise ValueError(f"population_size phải là số nguyên dương, nhận: {population_size}")

    cleaned_locations = [str(loc).strip() for loc in locations]
    n_locs = len(cleaned_locations)

    if rng is None:
        rng = np.random.default_rng(42)

    population: list[Chromosome] = []

    if n_locs == 0:
        return [Chromosome(()) for _ in range(population_size)]

    if n_locs == 1:
        return [Chromosome(cleaned_locations) for _ in range(population_size)]

    # Cá thể đầu tiên (canonical) nếu được yêu cầu
    start_idx = 0
    if include_canonical:
        population.append(Chromosome(cleaned_locations))
        start_idx = 1

    # Khởi tạo các cá thể còn lại bằng hoán vị ngẫu nhiên
    for _ in range(start_idx, population_size):
        permuted = rng.permutation(cleaned_locations).tolist()
        population.append(Chromosome(permuted))

    return population
