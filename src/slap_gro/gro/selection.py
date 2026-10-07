"""
Selection operators for GRO.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from slap_gro.gro.chromosome import Chromosome


def tournament_selection(
    population: Sequence[Chromosome],
    fitness_scores: Sequence[float],
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> Chromosome:
    """
    Chọn lọc giải đấu (Tournament Selection).
    
    Chọn ngẫu nhiên `tournament_size` cá thể từ quần thể, cá thể có
    fitness nhỏ nhất (quãng đường ngắn nhất) sẽ chiến thắng.
    
    Args:
        population: Danh sách các cá thể trong quần thể hiện tại.
        fitness_scores: Điểm fitness tương ứng (tổng quãng đường di chuyển).
        tournament_size: Kích thước giải đấu (paper quy định T=3).
        rng: Bộ sinh số ngẫu nhiên NumPy Generator.
        
    Returns:
        Cá thể chiến thắng giải đấu.
    """
    pop_size = len(population)
    if pop_size == 0:
        raise ValueError("Quần thể rỗng, không thể thực hiện chọn lọc.")
    if len(fitness_scores) != pop_size:
        raise ValueError(f"Số lượng fitness ({len(fitness_scores)}) khác kích thước quần thể ({pop_size}).")

    if rng is None:
        rng = np.random.default_rng(42)

    actual_t_size = min(max(1, tournament_size), pop_size)
    candidates_idx = rng.choice(pop_size, size=actual_t_size, replace=False)

    best_idx = candidates_idx[0]
    best_fitness = fitness_scores[best_idx]

    for idx in candidates_idx[1:]:
        if fitness_scores[idx] < best_fitness:
            best_fitness = fitness_scores[idx]
            best_idx = idx

    return population[best_idx]


def select_parents(
    population: Sequence[Chromosome],
    fitness_scores: Sequence[float],
    n_pairs: int,
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> list[tuple[Chromosome, Chromosome]]:
    """
    Chọn lọc `n_pairs` cặp cha mẹ bằng Tournament Selection để chuẩn bị lai ghép.
    
    Returns:
        Danh sách các bộ đôi (parent1, parent2).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    parents_list: list[tuple[Chromosome, Chromosome]] = []
    for _ in range(n_pairs):
        p1 = tournament_selection(population, fitness_scores, tournament_size, rng)
        p2 = tournament_selection(population, fitness_scores, tournament_size, rng)
        parents_list.append((p1, p2))

    return parents_list
