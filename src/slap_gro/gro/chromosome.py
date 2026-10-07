"""
Chromosome representation for GRO (Genetic Route Optimization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Chromosome:
    """
    Biểu diễn một nhiễm sắc thể trong thuật toán GRO.
    
    Gen (genes) là một hoán vị của các vị trí nhặt hàng (picking locations)
    thuộc một Picking Wave.
    
    Ví dụ:
        genes = ('A-14-11', 'B-12-05', 'C-08-01')
        route = ['DEPOT', 'A-14-11', 'B-12-05', 'C-08-01', 'DEPOT']
    """

    genes: tuple[str, ...]

    def __init__(self, genes: Sequence[str] | Iterable[str]) -> None:
        object.__setattr__(self, "genes", tuple(str(g).strip() for g in genes))

    @property
    def length(self) -> int:
        """Số lượng điểm nhặt hàng trong nhiễm sắc thể."""
        return len(self.genes)

    def to_route(self, depot_id: str = "DEPOT") -> list[str]:
        """
        Chuyển đổi nhiễm sắc thể thành hành trình khép kín bắt đầu và kết thúc tại Depot.
        
        Ví dụ:
            [depot_id] + list(genes) + [depot_id]
        """
        depot = str(depot_id).strip()
        return [depot, *self.genes, depot]

    def is_valid(self, expected_locations: Collection[str]) -> bool:
        """
        Kiểm tra tính hợp lệ của nhiễm sắc thể:
        - Phải chứa đúng và đủ các điểm nhặt hàng kỳ vọng.
        - Không chứa phần tử trùng lặp.
        """
        exp_set = {str(loc).strip() for loc in expected_locations}
        if len(self.genes) != len(exp_set):
            return False
        return set(self.genes) == exp_set

    def copy(self) -> Chromosome:
        """Tạo bản sao của Chromosome."""
        return Chromosome(self.genes)

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        return self.genes[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.genes)

    def __repr__(self) -> str:
        if len(self.genes) <= 5:
            items_str = ", ".join(self.genes)
        else:
            items_str = f"{', '.join(self.genes[:3])}, ... (+{len(self.genes) - 3} more)"
        return f"Chromosome(len={len(self.genes)}, [{items_str}])"
