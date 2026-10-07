"""
Mutation operators for GRO.
"""

from __future__ import annotations

import numpy as np

from slap_gro.gro.chromosome import Chromosome


def swap_mutation(chromosome: Chromosome, rng: np.random.Generator) -> Chromosome:
    """
    Đột biến hoán đổi vị trí (Swap Mutation).
    
    Chọn ngẫu nhiên 2 chỉ số phân biệt và đổi chỗ 2 gen.
    """
    n = len(chromosome)
    if n <= 1:
        return chromosome.copy()

    idx1, idx2 = rng.choice(n, size=2, replace=False)
    genes = list(chromosome.genes)
    genes[idx1], genes[idx2] = genes[idx2], genes[idx1]

    return Chromosome(genes)


def inversion_mutation(chromosome: Chromosome, rng: np.random.Generator) -> Chromosome:
    """
    Đột biến đảo ngược đoạn (Inversion / 2-opt Mutation).
    
    Chọn 2 điểm cắt và đảo ngược thứ tự các gen nằm giữa 2 điểm cắt.
    Rất hiệu quả trong việc gỡ bỏ các đoạn giao cắt trong hành trình TSP.
    """
    n = len(chromosome)
    if n <= 1:
        return chromosome.copy()

    idx1, idx2 = rng.choice(n, size=2, replace=False)
    c1, c2 = min(idx1, idx2), max(idx1, idx2)

    genes = list(chromosome.genes)
    genes[c1 : c2 + 1] = reversed(genes[c1 : c2 + 1])

    return Chromosome(genes)


def scramble_mutation(chromosome: Chromosome, rng: np.random.Generator) -> Chromosome:
    """
    Đột biến xáo trộn đoạn (Scramble Mutation).
    
    Chọn 2 điểm cắt và xáo trộn ngẫu nhiên thứ tự các gen nằm giữa 2 điểm cắt.
    """
    n = len(chromosome)
    if n <= 1:
        return chromosome.copy()

    idx1, idx2 = rng.choice(n, size=2, replace=False)
    c1, c2 = min(idx1, idx2), max(idx1, idx2)

    genes = list(chromosome.genes)
    sub = genes[c1 : c2 + 1]
    shuffled_sub = rng.permutation(sub).tolist()
    genes[c1 : c2 + 1] = shuffled_sub

    return Chromosome(genes)


def mutate(
    chromosome: Chromosome,
    pm: float = 0.08,
    method: str = "swap",
    rng: np.random.Generator | None = None,
) -> Chromosome:
    """
    Hàm đột biến tổng quát cho GRO.
    
    Args:
        chromosome: Cá thể cần đột biến.
        pm: Xác suất đột biến (Paper quy định Pm = 0.08).
        method: Phương pháp đột biến ('swap', 'inversion', 'scramble').
        rng: Bộ sinh số ngẫu nhiên NumPy.
        
    Returns:
        Cá thể sau khi xem xét đột biến.
    """
    if len(chromosome) <= 1:
        return chromosome.copy()

    if rng is None:
        rng = np.random.default_rng(42)

    # Kiểm tra xác suất đột biến Pm
    if rng.random() > pm:
        return chromosome.copy()

    norm_method = method.strip().lower()

    if norm_method == "swap":
        return swap_mutation(chromosome, rng)
    elif norm_method in ("inversion", "2opt", "reverse"):
        return inversion_mutation(chromosome, rng)
    elif norm_method == "scramble":
        return scramble_mutation(chromosome, rng)
    else:
        raise ValueError(f"Không hỗ trợ phương thức đột biến '{method}'. Các lựa chọn: 'swap', 'inversion', 'scramble'.")
