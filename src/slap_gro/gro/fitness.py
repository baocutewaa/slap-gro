"""
Fitness evaluation module for GRO (Genetic Route Optimization).
"""

from __future__ import annotations

from typing import Callable, Mapping

from slap_gro.gro.chromosome import Chromosome
from slap_gro.routing.route import Route


class FitnessEvaluator:
    """
    Bộ đánh giá độ thích nghi (Fitness Evaluator) cho các cá thể trong GRO.
    
    Trong bài toán tối ưu lộ trình nhặt hàng (Picking Tour TSP):
        Fitness(route) = Total Travel Distance (mét)
    Hàm mục tiêu là cực tiểu hóa (Minimization): khoảng cách càng ngắn,
    độ thích nghi càng cao (tốt hơn).
    """

    def __init__(
        self,
        depot_id: str = "DEPOT",
        distance_fn: Callable[[str, str], float] | None = None,
        distance_matrix: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        """
        Khởi tạo FitnessEvaluator.
        
        Args:
            depot_id: Mã định danh của trạm xuất phát/kết thúc (Depot).
            distance_fn: Hàm tính khoảng cách giữa 2 vị trí u và v: f(u, v) -> float.
            distance_matrix: Từ điển ma trận khoảng cách đối xứng (u, v) -> float (tùy chọn để tăng tốc).
        """
        self.depot_id = str(depot_id).strip()
        self._distance_fn = distance_fn
        self._distance_matrix = distance_matrix

        if self._distance_fn is None and self._distance_matrix is None:
            raise ValueError("Phải cung cấp ít nhất distance_fn hoặc distance_matrix cho FitnessEvaluator.")

    def get_distance(self, u: str, v: str) -> float:
        """Truy vấn khoảng cách giữa điểm u và điểm v."""
        u_str = str(u).strip()
        v_str = str(v).strip()
        if u_str == v_str:
            return 0.0

        if self._distance_matrix is not None:
            if (u_str, v_str) in self._distance_matrix:
                return float(self._distance_matrix[(u_str, v_str)])
            if (v_str, u_str) in self._distance_matrix:
                return float(self._distance_matrix[(v_str, u_str)])

        if self._distance_fn is not None:
            return float(self._distance_fn(u_str, v_str))

        raise KeyError(f"Không tìm thấy khoảng cách giữa {u_str} và {v_str}")

    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Tính tổng độ dài hành trình khép kín:
        Depot -> Gene[0] -> Gene[1] -> ... -> Gene[N-1] -> Depot.
        
        Args:
            chromosome: Cá thể cần đánh giá.
            
        Returns:
            Tổng quãng đường di chuyển (mét).
        """
        genes = chromosome.genes
        n = len(genes)

        if n == 0:
            return 0.0

        if n == 1:
            d = self.get_distance(self.depot_id, genes[0])
            return 2.0 * d

        # Đi từ Depot đến điểm đầu tiên
        total = self.get_distance(self.depot_id, genes[0])

        # Đi qua các điểm nhặt liên tiếp
        for i in range(n - 1):
            total += self.get_distance(genes[i], genes[i + 1])

        # Quay về Depot từ điểm cuối cùng
        total += self.get_distance(genes[-1], self.depot_id)

        return float(total)

    def evaluate_route(self, chromosome: Chromosome) -> Route:
        """
        Tạo đối tượng Route chuẩn hóa từ Chromosome đã đánh giá.
        """
        stops = chromosome.to_route(self.depot_id)
        dist = self.evaluate(chromosome)
        return Route(
            stops=stops,
            total_distance=dist,
            stops_count=len(chromosome.genes),
            is_closed=True,
        )
