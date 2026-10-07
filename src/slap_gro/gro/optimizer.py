"""
GRO Optimizer (Genetic Algorithm for Routing Optimization).

Tuân thủ chặt chẽ đặc tả Phase 10 trong Plan.md:
- Population = 1000 (mặc định theo paper)
- Iterations = 1000
- Tournament size = 3
- Pc = 0.90
- Pm = 0.08
- Elitism = True
- Hỗ trợ cả Mode A (Faithful replication) và Mode B (Permutation-safe OX/PMX)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from slap_gro.gro.chromosome import Chromosome
from slap_gro.gro.crossover import crossover
from slap_gro.gro.fitness import FitnessEvaluator
from slap_gro.gro.initialization import initialize_population
from slap_gro.gro.mutation import mutate
from slap_gro.gro.selection import tournament_selection


@dataclass
class GROConfig:
    """Cấu hình các siêu tham số cho thuật toán di truyền GRO."""

    population_size: int = 1000
    iterations: int = 1000
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.08
    crossover_method: str = "ox"  # 'ox', 'pmx', 'single_point_repair'
    mutation_method: str = "swap"  # 'swap', 'inversion', 'scramble'
    elitism: bool = True
    elite_count: int = 1
    random_seed: int = 42
    early_stopping_patience: int | None = None
    verbose: bool = False
    log_interval: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> GROConfig:
        """Khởi tạo GROConfig từ từ điển (ví dụ trích xuất từ file YAML)."""
        valid_keys = {
            "population_size",
            "iterations",
            "tournament_size",
            "crossover_rate",
            "mutation_rate",
            "crossover_method",
            "mutation_method",
            "elitism",
            "elite_count",
            "random_seed",
            "early_stopping_patience",
            "verbose",
            "log_interval",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, section: str = "gro") -> GROConfig:
        """Nạp cấu hình GRO từ tệp YAML."""
        path = Path(yaml_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg_dict = raw.get(section, raw)
        return cls.from_dict(cfg_dict)


@dataclass
class GROResult:
    """Kết quả đầu ra của quá trình tối ưu hóa hành trình bằng GRO."""

    best_chromosome: Chromosome
    best_distance: float
    best_route: list[str]
    initial_distance: float
    improvement_meters: float
    improvement_pct: float
    history_best_fitness: list[float] = field(default_factory=list)
    history_mean_fitness: list[float] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    iterations_run: int = 0
    config: GROConfig = field(default_factory=GROConfig)

    def summary(self) -> str:
        """Tóm tắt kết quả tối ưu bằng chuỗi định dạng."""
        return (
            f"GRO Result Summary:\n"
            f"  - Initial Distance : {self.initial_distance:.2f} m\n"
            f"  - Optimized Distance: {self.best_distance:.2f} m\n"
            f"  - Improvement       : {self.improvement_meters:.2f} m ({self.improvement_pct:.2f}%)\n"
            f"  - Route Stops       : {len(self.best_route)} stops ({len(self.best_chromosome.genes)} picks)\n"
            f"  - Iterations        : {self.iterations_run}\n"
            f"  - Execution Time    : {self.execution_time_seconds:.3f} s"
        )


class GROOptimizer:
    """
    Bộ tối ưu hóa hành trình nhặt hàng bằng thuật toán di truyền GRO.
    
    Quy trình thực thi:
      1. Khởi tạo quần thể ban đầu gồm N hoán vị.
      2. Đánh giá độ thích nghi (Fitness = Total Distance).
      3. Lặp qua các thế hệ:
         - Lưu giữ cá thể ưu tú (Elitism).
         - Chọn lọc giải đấu (Tournament Selection, size=3).
         - Lai ghép (Crossover với Pc = 0.90, Mode A/B).
         - Đột biến (Mutation với Pm = 0.08).
         - Cập nhật quần thể mới và ghi nhận hội tụ.
      4. Xuất kết quả hành trình tối ưu khép kín bắt đầu và kết thúc tại Depot.
    """

    def __init__(
        self,
        config: GROConfig | None = None,
        distance_fn: Callable[[str, str], float] | None = None,
        distance_matrix: Mapping[tuple[str, str], float] | None = None,
        depot_id: str = "DEPOT",
    ) -> None:
        self.config = config or GROConfig()
        self.depot_id = str(depot_id).strip()
        self._distance_fn = distance_fn
        self._distance_matrix = distance_matrix

        if self._distance_fn is None and self._distance_matrix is None:
            raise ValueError("Cần cung cấp ít nhất distance_fn hoặc distance_matrix cho GROOptimizer.")

        self.evaluator = FitnessEvaluator(
            depot_id=self.depot_id,
            distance_fn=self._distance_fn,
            distance_matrix=self._distance_matrix,
        )

    def optimize(
        self,
        picking_locations: Sequence[str],
        depot_id: str | None = None,
    ) -> GROResult:
        """
        Thực hiện tối ưu hóa thứ tự ghé thăm các vị trí nhặt hàng.
        
        Args:
            picking_locations: Danh sách các vị trí nhặt hàng cần ghé thăm.
            depot_id: Tùy chọn ghi đè mã định danh Depot.
            
        Returns:
            Đối tượng GROResult chứa kết quả hành trình tối ưu và lịch sử hội tụ.
        """
        start_time = time.perf_counter()
        depot = str(depot_id).strip() if depot_id is not None else self.depot_id

        # Cập nhật evaluator nếu depot thay đổi
        if depot != self.evaluator.depot_id:
            evaluator = FitnessEvaluator(
                depot_id=depot,
                distance_fn=self._distance_fn,
                distance_matrix=self._distance_matrix,
            )
        else:
            evaluator = self.evaluator

        # Chuẩn hóa danh sách các điểm nhặt hàng
        locs = [str(loc).strip() for loc in picking_locations if str(loc).strip() != depot]
        n_locs = len(locs)

        # Xử lý các trường hợp suy biến
        if n_locs == 0:
            exec_time = time.perf_counter() - start_time
            return GROResult(
                best_chromosome=Chromosome(()),
                best_distance=0.0,
                best_route=[depot, depot],
                initial_distance=0.0,
                improvement_meters=0.0,
                improvement_pct=0.0,
                execution_time_seconds=exec_time,
                iterations_run=0,
                config=self.config,
            )

        if n_locs == 1:
            exec_time = time.perf_counter() - start_time
            chrom = Chromosome(locs)
            dist = evaluator.evaluate(chrom)
            return GROResult(
                best_chromosome=chrom,
                best_distance=dist,
                best_route=[depot, locs[0], depot],
                initial_distance=dist,
                improvement_meters=0.0,
                improvement_pct=0.0,
                execution_time_seconds=exec_time,
                iterations_run=0,
                config=self.config,
            )

        # Precompute ma trận khoảng cách cục bộ cho các điểm trong đợt nhặt + Depot
        # để tăng tốc độ đánh giá fitness lên mức tối đa
        all_unique_points = list(dict.fromkeys([depot] + locs))
        local_matrix: dict[tuple[str, str], float] = {}
        for u in all_unique_points:
            for v in all_unique_points:
                if (u, v) not in local_matrix:
                    d = evaluator.get_distance(u, v)
                    local_matrix[(u, v)] = d
                    local_matrix[(v, u)] = d

        fast_evaluator = FitnessEvaluator(depot_id=depot, distance_matrix=local_matrix)

        # Khởi tạo bộ sinh số ngẫu nhiên có kiểm soát seed
        rng = np.random.default_rng(self.config.random_seed)

        # Khởi tạo quần thể ban đầu
        pop = initialize_population(
            locations=locs,
            population_size=self.config.population_size,
            rng=rng,
            include_canonical=True,
        )

        # Đánh giá quần thể thế hệ 0
        fitness_scores = [fast_evaluator.evaluate(ind) for ind in pop]

        # Ghi nhận trạng thái ban đầu từ cá thể đầu tiên (canonical order)
        initial_distance = fitness_scores[0]

        best_idx = int(np.argmin(fitness_scores))
        best_chromosome = pop[best_idx].copy()
        best_distance = fitness_scores[best_idx]

        history_best = [best_distance]
        history_mean = [float(np.mean(fitness_scores))]

        generations_no_improve = 0
        total_iterations = self.config.iterations

        # Vòng lặp các thế hệ
        for gen in range(1, total_iterations + 1):
            next_population: list[Chromosome] = []

            # 1. Elitism: Giữ lại các cá thể tốt nhất
            if self.config.elitism:
                elite_count = max(1, min(self.config.elite_count, len(pop)))
                elite_indices = np.argsort(fitness_scores)[:elite_count]
                for e_idx in elite_indices:
                    next_population.append(pop[e_idx].copy())

            # 2. Sinh các cá thể còn lại thông qua Selection -> Crossover -> Mutation
            needed = self.config.population_size - len(next_population)
            while len(next_population) < self.config.population_size:
                # Tournament Selection
                p1 = tournament_selection(
                    pop,
                    fitness_scores,
                    tournament_size=self.config.tournament_size,
                    rng=rng,
                )
                p2 = tournament_selection(
                    pop,
                    fitness_scores,
                    tournament_size=self.config.tournament_size,
                    rng=rng,
                )

                # Crossover
                c1, c2 = crossover(
                    p1,
                    p2,
                    pc=self.config.crossover_rate,
                    method=self.config.crossover_method,
                    rng=rng,
                )

                # Mutation
                c1 = mutate(
                    c1,
                    pm=self.config.mutation_rate,
                    method=self.config.mutation_method,
                    rng=rng,
                )
                c2 = mutate(
                    c2,
                    pm=self.config.mutation_rate,
                    method=self.config.mutation_method,
                    rng=rng,
                )

                next_population.append(c1)
                if len(next_population) < self.config.population_size:
                    next_population.append(c2)

            # Cập nhật thế hệ mới
            pop = next_population
            fitness_scores = [fast_evaluator.evaluate(ind) for ind in pop]

            gen_best_idx = int(np.argmin(fitness_scores))
            gen_best_fitness = fitness_scores[gen_best_idx]

            # Kiểm tra cải thiện toàn cục
            if gen_best_fitness < best_distance:
                best_distance = gen_best_fitness
                best_chromosome = pop[gen_best_idx].copy()
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            history_best.append(best_distance)
            history_mean.append(float(np.mean(fitness_scores)))

            if self.config.verbose and (gen % self.config.log_interval == 0 or gen == total_iterations):
                print(f"[GRO Gen {gen:4d}/{total_iterations}] Best: {best_distance:.2f} m, Mean: {history_mean[-1]:.2f} m")

            # Early stopping nếu được cấu hình
            if (
                self.config.early_stopping_patience is not None
                and generations_no_improve >= self.config.early_stopping_patience
            ):
                if self.config.verbose:
                    print(f"[GRO Early Stopping] Dừng tại thế hệ {gen} do không cải thiện sau {generations_no_improve} thế hệ.")
                break

        exec_time = time.perf_counter() - start_time
        improvement = initial_distance - best_distance
        improvement_pct = (improvement / initial_distance * 100.0) if initial_distance > 0 else 0.0

        return GROResult(
            best_chromosome=best_chromosome,
            best_distance=best_distance,
            best_route=best_chromosome.to_route(depot),
            initial_distance=initial_distance,
            improvement_meters=improvement,
            improvement_pct=improvement_pct,
            history_best_fitness=history_best,
            history_mean_fitness=history_mean,
            execution_time_seconds=exec_time,
            iterations_run=len(history_best) - 1,
            config=self.config,
        )
