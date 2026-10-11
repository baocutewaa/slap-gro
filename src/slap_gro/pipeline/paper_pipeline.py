"""
Paper Replication Pipeline (Mode A) - Phase 11.

Replicates the paper's experimental setup:
- 847 Synthetic Warehouse Units (15,246 slots)
- 2,000 Synthetic Products (Pareto ABC turnover: 20% A / 30% B / 50% C)
- 4 Programmatic Storage Policies (Random, Class-Based, Dedicated, Hybrid)
- 50 Picking Waves (25 items each)
- GRO Optimizer (1,000 population, 1,000 iterations, T=3, Pc=0.90, Pm=0.08)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from slap_gro.data.product import ProductCatalog
from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.experiment import ExperimentResult, ExperimentRunner
from slap_gro.picking.wave import PickingWaveDataset
from slap_gro.storage import get_storage_policy
from slap_gro.storage.base import StoragePolicy
from slap_gro.warehouse.warehouse_layout import SyntheticWarehouse


def build_paper_storage_policies(
    warehouse: SyntheticWarehouse,
    catalog: ProductCatalog,
    seed: int = 42,
) -> Dict[str, StoragePolicy]:
    """
    Tạo và gán vị trí lưu trữ cho cả 4 chính sách theo phương pháp của bài báo.
    """
    policies: Dict[str, StoragePolicy] = {}
    policy_names = ["random", "class_based", "dedicated", "hybrid"]

    for name in policy_names:
        pol = get_storage_policy(
            policy=name,
            source="generated",
            warehouse=warehouse,
            catalog=catalog,
            seed=seed,
        )
        policies[name] = pol

    return policies


def run_paper_experiment(
    n_waves: int = 50,
    items_per_wave: int = 25,
    gro_config: Optional[GROConfig] = None,
    config_yaml: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Thực thi toàn bộ luồng thực nghiệm Mode A (Paper Replication).
    
    Args:
        n_waves: Số lượng wave cần chạy (Paper chuẩn = 50).
        items_per_wave: Số lượt nhặt trên mỗi wave (Paper chuẩn = 25).
        gro_config: Cấu hình GRO tùy biến (nếu None sẽ nạp từ paper_replication.yaml).
        config_yaml: Đường dẫn tệp cấu hình YAML thay thế.
        output_dir: Thư mục lưu kết quả (mặc định 'outputs/paper_replication').
        seed: Seed tái lập ngẫu nhiên.
        verbose: In tiến trình thực hiện ra màn hình console.
        
    Returns:
        Đối tượng ExperimentResult tổng hợp 200 lượt chạy (4 policies x 50 waves).
    """
    if verbose:
        print("=" * 70)
        print("KHỞI CHẠY THỰC NGHIỆM MODE A: PAPER REPLICATION (PHASE 11)")
        print("=" * 70)

    # 1. Khởi tạo kho mô phỏng 847 units
    warehouse = SyntheticWarehouse()
    if verbose:
        print(f"[1/5] Khởi tạo kho mô phỏng: {warehouse.total_locations} locations, {warehouse.total_capacity:,} slots.")

    # 2. Sinh 2,000 SKU tuân theo phân bố Pareto
    catalog = ProductCatalog.generate_synthetic(n_products=2000, seed=seed)
    if verbose:
        print(f"[2/5] Sinh danh mục sản phẩm: {len(catalog)} SKU với phân bố ABC: {catalog.abc_distribution}.")

    # 3. Phân bổ lưu trữ cho 4 chính sách (Random, Class-Based, Dedicated, Hybrid)
    storage_policies = build_paper_storage_policies(warehouse=warehouse, catalog=catalog, seed=seed)
    if verbose:
        print(f"[3/5] Khởi tạo thành công 4 chính sách lưu trữ: {list(storage_policies.keys())}.")

    # 4. Sinh n_waves picking waves
    # Dùng chính sách random làm gốc cho wave items (chứa SKU để sau đó map sang từng policy)
    wave_dataset = PickingWaveDataset.generate_synthetic(
        catalog=catalog,
        storage_policy=storage_policies["random"],
        n_waves=n_waves,
        items_per_wave=items_per_wave,
        seed=seed,
    )
    if verbose:
        print(f"[4/5] Sinh thành công {len(wave_dataset)} picking waves (mỗi wave {items_per_wave} SKU).")

    # 5. Cấu hình GRO Optimizer
    if gro_config is None:
        yaml_path = config_yaml or Path("configs/paper_replication.yaml")
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

    if verbose:
        print(f"[5/5] Cấu hình GRO: Pop={gro_config.population_size}, Iter={gro_config.iterations}, "
              f"T={gro_config.tournament_size}, Pc={gro_config.crossover_rate}, Pm={gro_config.mutation_rate}.")

    # Khởi chạy Runner
    runner = ExperimentRunner(
        warehouse=warehouse,
        storage_policies=storage_policies,
        gro_config=gro_config,
        depot_id="DEPOT",
        mode="paper_replication",
    )

    result = runner.run(
        waves=list(wave_dataset.waves.values()),
        max_waves=n_waves,
        verbose=verbose,
    )

    # Lưu kết quả nếu được cấu hình
    save_dir = Path(output_dir or "outputs/paper_replication")
    saved_files = result.save_results(save_dir)
    if verbose:
        print("-" * 70)
        print(f"Hoàn thành thực nghiệm Mode A! Kết quả đã lưu tại: {save_dir}")
        for k, p in saved_files.items():
            print(f"  * {k.upper():<8}: {p.name}")
        print("\nBảng tóm tắt kết quả (Summary Statistics):")
        print(result.summary_statistics().to_string(index=False))
        print("=" * 70)

    return result
