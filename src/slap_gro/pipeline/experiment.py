"""
Experiment Runner & Results Container for SLAP-GRO Pipeline (Phase 11).

Orchestrates multi-wave, multi-policy optimization experiments, computing
travel distance improvements across different storage policies under GRO.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from slap_gro.gro.optimizer import GROConfig, GROOptimizer, GROResult
from slap_gro.picking.picking_points import build_gro_route_targets
from slap_gro.picking.wave import PickingWave
from slap_gro.routing.distance import distance
from slap_gro.storage.base import StoragePolicy
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


@dataclass
class SingleRunResult:
    """Kết quả chạy tối ưu hóa của 1 đợt nhặt hàng dưới 1 chính sách lưu trữ cụ thể."""

    wave_id: int
    policy_name: str
    initial_distance: float
    optimized_distance: float
    improvement_meters: float
    improvement_pct: float
    execution_time_seconds: float
    iterations_run: int
    n_picks: int
    mode: Optional[str] = None
    unique_locations: Optional[int] = None
    best_route: list[str] = field(default_factory=list)
    history_best_fitness: list[float] = field(default_factory=list)

    def __init__(
        self,
        wave_id: int,
        policy_name: Optional[str] = None,
        initial_distance: float = 0.0,
        optimized_distance: Optional[float] = None,
        improvement_meters: Optional[float] = None,
        improvement_pct: float = 0.0,
        execution_time_seconds: Optional[float] = None,
        iterations_run: Optional[int] = None,
        n_picks: Optional[int] = None,
        mode: Optional[str] = None,
        unique_locations: Optional[int] = None,
        best_route: Optional[list[str]] = None,
        history_best_fitness: Optional[list[float]] = None,
        # Aliases for flexibility across pipelines and tests
        policy: Optional[str] = None,
        best_distance: Optional[float] = None,
        computation_time_s: Optional[float] = None,
        iterations: Optional[int] = None,
        num_picks: Optional[int] = None,
    ):
        self.wave_id = int(wave_id)
        self.policy_name = str(policy_name or policy or "")
        self.initial_distance = float(initial_distance)
        opt_dist = (
            optimized_distance
            if optimized_distance is not None
            else (best_distance if best_distance is not None else initial_distance)
        )
        self.optimized_distance = float(opt_dist)
        if improvement_meters is not None:
            self.improvement_meters = float(improvement_meters)
        else:
            self.improvement_meters = float(self.initial_distance - self.optimized_distance)
        self.improvement_pct = float(improvement_pct)
        exec_time = (
            execution_time_seconds
            if execution_time_seconds is not None
            else (computation_time_s if computation_time_s is not None else 0.0)
        )
        self.execution_time_seconds = float(exec_time)
        n_iter = (
            iterations_run
            if iterations_run is not None
            else (iterations if iterations is not None else 0)
        )
        self.iterations_run = int(n_iter)
        picks = n_picks if n_picks is not None else (num_picks if num_picks is not None else 0)
        self.n_picks = int(picks)
        self.mode = mode
        self.unique_locations = unique_locations
        self.best_route = list(best_route) if best_route else []
        self.history_best_fitness = list(history_best_fitness) if history_best_fitness else []

    @property
    def policy(self) -> str:
        return self.policy_name

    @property
    def best_distance(self) -> float:
        return self.optimized_distance

    @property
    def num_picks(self) -> int:
        return self.n_picks

    @property
    def computation_time_s(self) -> float:
        return self.execution_time_seconds

    def to_dict(self) -> dict:
        return {
            "wave_id": self.wave_id,
            "policy": self.policy_name,
            "policy_name": self.policy_name,
            "n_picks": self.n_picks,
            "num_picks": self.n_picks,
            "unique_locations": self.unique_locations if self.unique_locations is not None else self.n_picks,
            "initial_distance": self.initial_distance,
            "optimized_distance": self.optimized_distance,
            "best_distance": self.optimized_distance,
            "improvement_meters": self.improvement_meters,
            "improvement_pct": self.improvement_pct,
            "execution_time_seconds": self.execution_time_seconds,
            "computation_time_s": self.execution_time_seconds,
            "iterations_run": self.iterations_run,
            "mode": self.mode,
            "best_route": " -> ".join(self.best_route) if self.best_route else "",
        }


@dataclass
class ExperimentResult:
    """Tập hợp kết quả của toàn bộ thực nghiệm đa chính sách & đa wave."""

    mode: str
    runs: list[SingleRunResult] = field(default_factory=list)
    gro_config: GROConfig = field(default_factory=GROConfig)
    storage_policies: list[str] = field(default_factory=list)
    total_waves: int = 0
    total_execution_time_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    def add_run(self, run: SingleRunResult) -> None:
        """Thêm kết quả của một đợt chạy đơn lẻ vào tập kết quả."""
        self.runs.append(run)
        if run.policy_name not in self.storage_policies:
            self.storage_policies.append(run.policy_name)

    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi kết quả chi tiết của tất cả các lần chạy sang DataFrame."""
        rows = [r.to_dict() for r in self.runs]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "wave_id",
                    "policy",
                    "policy_name",
                    "n_picks",
                    "initial_distance",
                    "optimized_distance",
                    "best_distance",
                    "improvement_meters",
                    "improvement_pct",
                    "execution_time_seconds",
                    "iterations_run",
                ]
            )
        df = pd.DataFrame(rows)
        return df

    def summary_statistics(self) -> pd.DataFrame:
        """
        Tính toán bảng thống kê mô tả (Mean, SD, Min, Q1, Median, Q3, Max)
        cho quãng đường tối ưu của từng chính sách lưu trữ theo chuẩn báo cáo của bài báo.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        records = []
        for policy, group in df.groupby("policy_name"):
            opt_dist = group["optimized_distance"].values
            init_dist = group["initial_distance"].values
            imp_pct = group["improvement_pct"].values

            records.append({
                "policy": policy,
                "n_waves": len(group),
                "initial_mean": float(np.mean(init_dist)),
                "mean": float(np.mean(opt_dist)),
                "std": float(np.std(opt_dist, ddof=1)) if len(opt_dist) > 1 else 0.0,
                "min": float(np.min(opt_dist)),
                "q1": float(np.percentile(opt_dist, 25)),
                "median": float(np.median(opt_dist)),
                "q3": float(np.percentile(opt_dist, 75)),
                "max": float(np.max(opt_dist)),
                "mean_improvement_pct": float(np.mean(imp_pct)),
                "total_time_s": float(group["execution_time_seconds"].sum()),
            })

        summary_df = pd.DataFrame(records)
        summary_df.sort_values(by="mean", inplace=True)
        return summary_df.reset_index(drop=True)

    def save_results(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Lưu trữ kết quả thực nghiệm ra thư mục outputs:
        - raw_experiment_results.csv
        - summary_statistics.csv
        - experiment_metadata.json
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Raw results CSV
        raw_df = self.to_dataframe()
        raw_file = out_path / "raw_experiment_results.csv"
        raw_df.to_csv(raw_file, index=False)

        # 2. Summary stats CSV
        sum_df = self.summary_statistics()
        sum_file = out_path / "summary_statistics.csv"
        sum_df.to_csv(sum_file, index=False)

        # 3. Metadata JSON
        meta_file = out_path / "experiment_metadata.json"
        meta = {
            "mode": self.mode,
            "total_runs": len(self.runs),
            "total_waves": self.total_waves,
            "policies": self.storage_policies,
            "total_execution_time_seconds": round(self.total_execution_time_seconds, 3),
            "gro_config": {
                "population_size": self.gro_config.population_size,
                "iterations": self.gro_config.iterations,
                "tournament_size": self.gro_config.tournament_size,
                "crossover_rate": self.gro_config.crossover_rate,
                "mutation_rate": self.gro_config.mutation_rate,
                "crossover_method": self.gro_config.crossover_method,
                "mutation_method": self.gro_config.mutation_method,
                "elitism": self.gro_config.elitism,
                "random_seed": self.gro_config.random_seed,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "raw": raw_file,
            "summary": sum_file,
            "metadata": meta_file,
            "detailed_csv": raw_file,
            "summary_csv": sum_file,
            "summary_json": meta_file,
        }


class ExperimentRunner:
    """
    Bộ điều phối thực nghiệm (Experiment Runner) cho bài toán SLAP-GRO.
    
    Thực thi tối ưu hóa trên ma trận thực nghiệm:
        [Số lượng chính sách lưu trữ] × [Số lượng Picking Waves] × [Thuật toán GRO]
    """

    def __init__(
        self,
        warehouse: WarehouseLayout,
        storage_policies: Union[Sequence[StoragePolicy], Mapping[str, StoragePolicy]],
        gro_config: Optional[GROConfig] = None,
        distance_fn: Optional[Callable[[str, str], float]] = None,
        depot_id: str = "DEPOT",
        mode: str = "custom",
    ) -> None:
        self.warehouse = warehouse
        self.gro_config = gro_config or GROConfig()
        self.depot_id = str(depot_id).strip()
        self.mode = mode

        # Chuẩn hóa danh sách chính sách lưu trữ thành dict name -> policy
        if isinstance(storage_policies, Mapping):
            self.policies = dict(storage_policies)
        else:
            self.policies = {p.name: p for p in storage_policies}

        if not self.policies:
            raise ValueError("At least one storage policy must be provided to ExperimentRunner")

        # Thiết lập hàm tính khoảng cách mặc định nếu chưa truyền vào
        if distance_fn is not None:
            self.distance_fn = distance_fn
        else:
            def _default_dist(u: str, v: str) -> float:
                loc_u = self.warehouse.get_depot() if u == self.depot_id else self.warehouse.get_location(u)
                loc_v = self.warehouse.get_depot() if v == self.depot_id else self.warehouse.get_location(v)
                if loc_u is None or loc_v is None:
                    return 0.0
                return distance(loc_u, loc_v, method="euclidean")

            self.distance_fn = _default_dist

    def run_single(self, wave: PickingWave, policy: StoragePolicy) -> SingleRunResult:
        """
        Thực hiện tối ưu hóa cho 1 wave cụ thể dưới 1 chính sách lưu trữ cụ thể.
        """
        # Trích xuất các vị trí cần nhặt hàng tương ứng với chính sách lưu trữ này
        depot, pick_locations = build_gro_route_targets(
            wave=wave,
            warehouse=self.warehouse,
            storage_policy=policy,
        )
        loc_ids = [loc.id for loc in pick_locations]

        # Khởi tạo optimizer
        optimizer = GROOptimizer(
            config=self.gro_config,
            distance_fn=self.distance_fn,
            depot_id=self.depot_id,
        )

        gro_res = optimizer.optimize(loc_ids, depot_id=self.depot_id)

        return SingleRunResult(
            wave_id=wave.wave_id,
            policy_name=policy.name,
            initial_distance=gro_res.initial_distance,
            optimized_distance=gro_res.best_distance,
            improvement_meters=gro_res.improvement_meters,
            improvement_pct=gro_res.improvement_pct,
            execution_time_seconds=gro_res.execution_time_seconds,
            iterations_run=gro_res.iterations_run,
            n_picks=len(loc_ids),
            mode=self.mode,
            unique_locations=len(set(loc_ids)),
            best_route=gro_res.best_route,
            history_best_fitness=gro_res.history_best_fitness,
        )

    def run(
        self,
        waves: Sequence[PickingWave],
        max_waves: Optional[int] = None,
        verbose: bool = False,
    ) -> ExperimentResult:
        """
        Chạy toàn bộ thực nghiệm trên tập hợp các waves và các chính sách lưu trữ.
        
        Args:
            waves: Danh sách các đợt nhặt hàng.
            max_waves: Giới hạn số lượng wave cần chạy (hữu ích cho debug hoặc chạy thử nghiệm phân tầng).
            verbose: In tiến độ thực hiện ra màn hình.
            
        Returns:
            Đối tượng ExperimentResult tổng hợp toàn bộ kết quả.
        """
        start_time = time.perf_counter()

        target_waves = list(waves)
        if max_waves is not None and max_waves > 0:
            target_waves = target_waves[:max_waves]

        total_runs_expected = len(self.policies) * len(target_waves)
        if verbose:
            print(
                f"[ExperimentRunner] Bắt đầu thực nghiệm ({self.mode}): "
                f"{len(self.policies)} policies x {len(target_waves)} waves = {total_runs_expected} runs."
            )

        runs: List[SingleRunResult] = []
        completed = 0

        for pol_name, policy in self.policies.items():
            for wave in target_waves:
                run_res = self.run_single(wave, policy)
                runs.append(run_res)
                completed += 1

                if verbose and (completed % 10 == 0 or completed == total_runs_expected):
                    print(
                        f"  -> Run {completed:3d}/{total_runs_expected}: "
                        f"Policy={pol_name:<11} Wave={wave.wave_id} | "
                        f"Initial: {run_res.initial_distance:.1f}m -> Opt: {run_res.optimized_distance:.1f}m "
                        f"(-{run_res.improvement_pct:.1f}%)"
                    )

        total_duration = time.perf_counter() - start_time

        return ExperimentResult(
            mode=self.mode,
            runs=runs,
            gro_config=self.gro_config,
            storage_policies=list(self.policies.keys()),
            total_waves=len(target_waves),
            total_execution_time_seconds=total_duration,
        )
