"""
Real Dataset Pipeline (Mode B) - Phase 11.

Executes optimization benchmarks on real-world warehouse dataset:
- CSV Warehouse with 2,292 locations and 4 vertical levels
- 4 Observed Storage Assignment Policies (Random, Class-Based, Dedicated, Hybrid)
- 9,707 Real Picking Waves
- Staged Progression:
    * Stage 1: 10 waves x 100 iterations (Fast sanity benchmark)
    * Stage 2: 50 waves x 100 iterations (Intermediate benchmark)
    * Stage 3: 50 waves x 1,000 iterations (Paper-equivalent scale on real data)
    * Stage 4: Full-scale (All waves x 1,000 iterations)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.experiment import ExperimentResult, ExperimentRunner
from slap_gro.picking.wave import PickingWave, PickingWaveDataset
from slap_gro.storage import get_storage_policy
from slap_gro.storage.base import StoragePolicy
from slap_gro.warehouse.warehouse_layout import CSVWarehouse

# Predefined stage profiles matching Plan.md Phase 11
REAL_DATA_STAGES = {
    "stage_1": {"max_waves": 10, "iterations": 100, "desc": "10 waves x 100 iterations"},
    "10x100": {"max_waves": 10, "iterations": 100, "desc": "10 waves x 100 iterations"},
    "stage_2": {"max_waves": 50, "iterations": 100, "desc": "50 waves x 100 iterations"},
    "50x100": {"max_waves": 50, "iterations": 100, "desc": "50 waves x 100 iterations"},
    "stage_3": {"max_waves": 50, "iterations": 1000, "desc": "50 waves x 1,000 iterations"},
    "50x1000": {"max_waves": 50, "iterations": 1000, "desc": "50 waves x 1,000 iterations"},
    "full": {"max_waves": None, "iterations": 1000, "desc": "Full scale waves x 1,000 iterations"},
}


def build_real_storage_policies(
    file_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, StoragePolicy]:
    """
    Nạp 4 chính sách lưu trữ thực tế từ các tệp ma trận CSV quan sát được.
    
    Args:
        file_paths: Từ điển ánh xạ tên chính sách ('random', 'class_based', ...)
                   tới đường dẫn tệp CSV tùy chọn.
                   
    Returns:
        Dict chứa 4 đối tượng StoragePolicy tương ứng.
    """
    policies: Dict[str, StoragePolicy] = {}
    policy_names = ["random", "class_based", "dedicated", "hybrid"]

    for name in policy_names:
        custom_path = file_paths.get(name) if file_paths else None
        pol = get_storage_policy(
            policy=name,
            source="observed",
            file_path=custom_path,
        )
        policies[name] = pol

    return policies


def run_real_experiment(
    stage: Optional[str] = None,
    max_waves: Optional[int] = 10,
    waves_subset: Optional[Sequence[int]] = None,
    gro_config: Optional[GROConfig] = None,
    config_yaml: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    warehouse: Optional[CSVWarehouse] = None,
    wave_dataset: Optional[PickingWaveDataset] = None,
    storage_policies: Optional[Dict[str, StoragePolicy]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Thực thi luồng thực nghiệm Mode B trên tập dữ liệu thực tế (Real-data Mode).
    
    Hỗ trợ chạy theo các giai đoạn tăng dần (Staged progression) như quy định trong Plan.md:
        - stage="stage_1" hoặc "10x100": 10 waves x 100 iterations
        - stage="stage_2" hoặc "50x100": 50 waves x 100 iterations
        - stage="stage_3" hoặc "50x1000": 50 waves x 1,000 iterations
        - stage="full": Toàn bộ 9,707 waves x 1,000 iterations
        
    Args:
        stage: Tên giai đoạn định sẵn (nếu cung cấp sẽ ghi đè max_waves và iterations).
        max_waves: Giới hạn số wave thực hiện (mặc định 10 wave để an toàn).
        waves_subset: Danh sách wave_id cụ thể cần chạy (ví dụ [33168, 33169]).
        gro_config: Cấu hình GRO tùy biến (nếu None sẽ nạp từ configs/real_data.yaml).
        config_yaml: Đường dẫn tệp cấu hình YAML thay thế.
        output_dir: Thư mục lưu kết quả (mặc định 'outputs/real_data' hoặc 'outputs/real_data_{stage}').
        warehouse: Đối tượng CSVWarehouse có sẵn (nếu None sẽ tự nạp từ CSV).
        wave_dataset: Đối tượng PickingWaveDataset có sẵn (nếu None sẽ tự nạp từ CSV).
        storage_policies: Dict 4 StoragePolicy có sẵn (nếu None sẽ tự nạp từ CSV).
        seed: Seed tái lập ngẫu nhiên.
        verbose: In tiến trình thực hiện ra console.
        
    Returns:
        ExperimentResult chứa dữ liệu chi tiết và tóm tắt thống kê.
    """
    # Xử lý tham số theo stage nếu có
    staged_desc = "Tùy chỉnh"
    target_iterations = None
    if stage:
        stage_key = stage.lower().strip()
        if stage_key in REAL_DATA_STAGES:
            stage_info = REAL_DATA_STAGES[stage_key]
            max_waves = stage_info["max_waves"]
            target_iterations = stage_info["iterations"]
            staged_desc = stage_info["desc"]
        else:
            raise ValueError(
                f"Giai đoạn '{stage}' không hợp lệ. Các giai đoạn khả dụng: {list(REAL_DATA_STAGES.keys())}"
            )

    if verbose:
        print("=" * 70)
        print("KHỞI CHẠY THỰC NGHIỆM MODE B: REAL-DATA REPLICATION (PHASE 11)")
        print(f"Giai đoạn thực thi: {staged_desc} (max_waves={max_waves})")
        print("=" * 70)

    # 1. Khởi tạo kho thực tế CSVWarehouse (2,292 locations)
    if warehouse is None:
        warehouse = CSVWarehouse()
    if verbose:
        print(f"[1/5] Kho thực tế CSVWarehouse: {warehouse.total_locations:,} locations, "
              f"{len(warehouse.navigation_points)} navigation points.")

    # 2. Nạp 4 chính sách lưu trữ quan sát được
    if storage_policies is None:
        storage_policies = build_real_storage_policies()
    if verbose:
        print(f"[2/5] Nạp thành công 4 chính sách lưu trữ thực tế: {list(storage_policies.keys())}.")

    # 3. Nạp tập dữ liệu Picking Waves
    if wave_dataset is None:
        wave_dataset = PickingWaveDataset.from_csv()
    if verbose:
        print(f"[3/5] Nạp tập dữ liệu sóng nhặt hàng: Tổng cộng {len(wave_dataset):,} waves.")

    # 4. Lựa chọn danh sách waves để chạy
    selected_waves: List[PickingWave] = []
    if waves_subset is not None:
        for w_id in waves_subset:
            wave = wave_dataset.get(int(w_id))
            if wave is not None:
                selected_waves.append(wave)
            else:
                if verbose:
                    print(f"  [Cảnh báo] Wave ID {w_id} không tìm thấy trong tập dữ liệu.")
    else:
        all_waves = list(wave_dataset.waves.values())
        if max_waves is not None:
            selected_waves = all_waves[:max_waves]
        else:
            selected_waves = all_waves

    if verbose:
        print(f"[4/5] Đã chọn {len(selected_waves)} waves để thực thi.")

    # 5. Cấu hình GRO
    if gro_config is None:
        yaml_path = config_yaml or Path("configs/real_data.yaml")
        if Path(yaml_path).exists():
            gro_config = GROConfig.from_yaml(yaml_path)
        else:
            gro_config = GROConfig(
                population_size=1000,
                iterations=1000,
                tournament_size=3,
                crossover_rate=0.90,
                mutation_rate=0.08,
                elitism=True,
                random_seed=seed,
            )

    # Ghi đè số vòng lặp nếu stage chỉ định
    if target_iterations is not None:
        gro_config.iterations = target_iterations

    if verbose:
        print(f"[5/5] Cấu hình GRO: Pop={gro_config.population_size}, Iter={gro_config.iterations}, "
              f"T={gro_config.tournament_size}, Pc={gro_config.crossover_rate}, Pm={gro_config.mutation_rate}.")

    # Khởi chạy Runner
    runner = ExperimentRunner(
        warehouse=warehouse,
        storage_policies=storage_policies,
        gro_config=gro_config,
        depot_id="DEPOT",
        mode="real_data",
    )

    result = runner.run(
        waves=selected_waves,
        max_waves=len(selected_waves),
        verbose=verbose,
    )

    # Lưu kết quả
    default_subfolder = f"real_data_{stage}" if stage else "real_data"
    save_dir = Path(output_dir or Path("outputs") / default_subfolder)
    saved_files = result.save_results(save_dir)

    if verbose:
        print("-" * 70)
        print(f"Hoàn thành thực nghiệm Mode B! Kết quả đã lưu tại: {save_dir}")
        for k, p in saved_files.items():
            print(f"  * {k.upper():<8}: {p.name}")
        print("\nBảng tóm tắt kết quả (Summary Statistics):")
        print(result.summary_statistics().to_string(index=False))
        print("=" * 70)

    return result
