"""
Crossover operators for GRO.

Tuân thủ mục 15 của Plan.md:
- Mode B: Permutation-safe crossover (OX - Order Crossover, PMX - Partially Mapped Crossover)
- Mode A: Faithful replication with repair (Single-point crossover with conflict repair)
"""

from __future__ import annotations

import numpy as np

from slap_gro.gro.chromosome import Chromosome


def order_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
) -> tuple[Chromosome, Chromosome]:
    """
    Toán tử lai ghép Order Crossover (OX) - Mode B (Permutation-Safe).
    
    Bảo toàn thứ tự tương đối của các vị trí nhặt hàng mà không sinh ra
    điểm trùng lặp hoặc thiếu sót.
    """
    n = len(parent1)
    if n <= 2:
        return parent1.copy(), parent2.copy()

    # Chọn 2 điểm cắt ngẫu nhiên: 0 <= c1 < c2 < n
    idx1, idx2 = rng.choice(n, size=2, replace=False)
    c1, c2 = min(idx1, idx2), max(idx1, idx2)
    # Nếu chọn bao phủ toàn bộ, thu nhỏ lại 1 đơn vị
    if c1 == 0 and c2 == n - 1:
        c2 = n - 2

    p1 = list(parent1.genes)
    p2 = list(parent2.genes)

    def _create_child(p_main: list[str], p_other: list[str]) -> list[str]:
        child = [None] * n
        # Sao chép đoạn slice từ cha mẹ chính
        slice_set = set()
        for i in range(c1, c2 + 1):
            child[i] = p_main[i]
            slice_set.add(p_main[i])

        # Điền các vị trí còn lại từ cha mẹ còn lại theo vòng xoay từ c2 + 1
        current_pos = (c2 + 1) % n
        for i in range(n):
            candidate = p_other[(c2 + 1 + i) % n]
            if candidate not in slice_set:
                child[current_pos] = candidate
                current_pos = (current_pos + 1) % n

        return child

    child1_genes = _create_child(p1, p2)
    child2_genes = _create_child(p2, p1)

    return Chromosome(child1_genes), Chromosome(child2_genes)


def pmx_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
) -> tuple[Chromosome, Chromosome]:
    """
    Toán tử lai ghép Partially Mapped Crossover (PMX) - Mode B (Permutation-Safe).
    """
    n = len(parent1)
    if n <= 2:
        return parent1.copy(), parent2.copy()

    idx1, idx2 = rng.choice(n, size=2, replace=False)
    c1, c2 = min(idx1, idx2), max(idx1, idx2)
    if c1 == 0 and c2 == n - 1:
        c2 = n - 2

    p1 = list(parent1.genes)
    p2 = list(parent2.genes)

    def _pmx_child(p_main: list[str], p_other: list[str]) -> list[str]:
        child = [None] * n
        child[c1 : c2 + 1] = p_main[c1 : c2 + 1]

        # Mapping: p_main slice -> p_other slice
        mapping = {}
        for i in range(c1, c2 + 1):
            mapping[p_main[i]] = p_other[i]

        slice_set = set(p_main[c1 : c2 + 1])

        for i in list(range(0, c1)) + list(range(c2 + 1, n)):
            val = p_other[i]
            while val in slice_set:
                val = mapping[val]
            child[i] = val

        return child

    child1 = _pmx_child(p1, p2)
    child2 = _pmx_child(p2, p1)

    return Chromosome(child1), Chromosome(child2)


def single_point_crossover_repair(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
) -> tuple[Chromosome, Chromosome]:
    """
    Toán tử lai ghép 1 điểm cắt có cơ chế sửa lỗi (Repair) - Mode A (Replication).
    
    Lấy tiền tố [0:cut] từ cha mẹ chính, sau đó bổ sung tuần tự các gen còn thiếu
    theo đúng thứ tự xuất hiện ở cha mẹ còn lại.
    """
    n = len(parent1)
    if n <= 2:
        return parent1.copy(), parent2.copy()

    cut = int(rng.integers(1, n))
    p1 = list(parent1.genes)
    p2 = list(parent2.genes)

    def _make_child(head: list[str], other_parent: list[str]) -> list[str]:
        child = list(head)
        used = set(head)
        for gene in other_parent:
            if gene not in used:
                child.append(gene)
                used.add(gene)
        return child

    c1_genes = _make_child(p1[:cut], p2)
    c2_genes = _make_child(p2[:cut], p1)

    return Chromosome(c1_genes), Chromosome(c2_genes)


def crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    pc: float = 0.90,
    method: str = "ox",
    rng: np.random.Generator | None = None,
) -> tuple[Chromosome, Chromosome]:
    """
    Hàm lai ghép tổng quát cho GRO.
    
    Args:
        parent1: Cá thể cha/mẹ 1.
        parent2: Cá thể cha/mẹ 2.
        pc: Xác suất lai ghép (Paper quy định Pc = 0.90).
        method: Phương pháp lai ghép ('ox', 'pmx', 'single_point_repair', 'mode_a', 'mode_b').
        rng: Bộ sinh số ngẫu nhiên NumPy.
        
    Returns:
        Bộ đôi cá thể con (child1, child2).
    """
    if len(parent1) <= 1:
        return parent1.copy(), parent2.copy()

    if rng is None:
        rng = np.random.default_rng(42)

    # Kiểm tra xác suất lai ghép Pc
    if rng.random() > pc:
        return parent1.copy(), parent2.copy()

    norm_method = method.strip().lower()

    if norm_method in ("ox", "order", "mode_b"):
        return order_crossover(parent1, parent2, rng)
    elif norm_method == "pmx":
        return pmx_crossover(parent1, parent2, rng)
    elif norm_method in ("single_point_repair", "single_point", "mode_a"):
        return single_point_crossover_repair(parent1, parent2, rng)
    else:
        raise ValueError(f"Không hỗ trợ phương thức lai ghép '{method}'. Các lựa chọn: 'ox', 'pmx', 'single_point_repair'.")
