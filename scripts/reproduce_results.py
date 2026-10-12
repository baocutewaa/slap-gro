#!/usr/bin/env python3
"""
CLI Script to reproduce full experimental benchmark results (Milestone 8 Benchmark).

Generates the 5-way comparative benchmark reported in literature:
1. Original (Baseline: actual company route in Mode B, canonical FIFO in Mode A)
2. Random Storage Policy (with GRO)
3. Class-Based Storage Policy (with GRO)
4. Dedicated Storage Policy (with GRO)
5. Hybrid Storage Policy (with GRO)

Usage:
    python scripts/reproduce_results.py --mode paper --waves 50 --fast
    python scripts/reproduce_results.py --mode real --stage 10x100
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.paper_pipeline import run_paper_experiment
from slap_gro.pipeline.real_pipeline import REAL_DATA_STAGES, run_real_experiment


def main():
    parser = argparse.ArgumentParser(
        description="SLAP-GRO: Reproduce Full Benchmark Experiments (Original + 4 Policies)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "real", "both"],
        help="Experiment mode: 'paper' (Mode A), 'real' (Mode B), or 'both'",
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=50,
        help="Number of picking waves to evaluate (Paper default = 50)",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=25,
        help="Number of items per picking wave in Mode A",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of GRO iterations per wave",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=1000,
        help="GRO population size",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=list(REAL_DATA_STAGES.keys()),
        help="Predefined stage profile for Mode B (e.g., 10x100, 50x100, 50x1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast smoke test mode: 2 waves, pop=50, iter=20",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reproduced_results",
        help="Output directory to save summary statistics and comparison tables",
    )

    args = parser.parse_args()

    n_waves = args.waves
    pop_size = args.pop
    n_iterations = args.iterations
    items_per_wave = args.items

    if args.fast:
        print(">>> FAST MODE ENABLED: Running fast verification (2 waves, pop=50, iter=20)...")
        n_waves = 2
        pop_size = 50
        n_iterations = 20
        items_per_wave = 10

    gro_cfg = GROConfig(
        population_size=pop_size,
        iterations=n_iterations,
        tournament_size=3,
        crossover_rate=0.90,
        mutation_rate=0.08,
        elitism=True,
        random_seed=args.seed,
    )

    out_base = Path(args.output_dir)

    if args.mode in ("paper", "both"):
        print("\n" + "=" * 75)
        print("BẮT ĐẦU TÁI LẬP KẾT QUẢ MODE A: PAPER REPLICATION (ORIGINAL + 4 POLICIES)")
        print("=" * 75)
        paper_out = out_base / "paper"
        res_paper = run_paper_experiment(
            n_waves=n_waves,
            items_per_wave=items_per_wave,
            gro_config=gro_cfg,
            output_dir=paper_out,
            include_baseline=True,
            seed=args.seed,
            verbose=True,
        )
        print("\n[MODE A] BẢNG SO SÁNH 5 CHIẾN LƯỢC (ORIGINAL VS 4 CHÍNH SÁCH GRO):")
        print(res_paper.summary_statistics().to_string(index=False))
        # Xuất kiểm định thống kê ra outputs/tables
        tables_dir = Path("outputs/tables")
        res_paper.run_statistical_analysis(output_dir=tables_dir)

    if args.mode in ("real", "both"):
        print("\n" + "=" * 75)
        print("BẮT ĐẦU TÁI LẬP KẾT QUẢ MODE B: REAL DATA REPLICATION (ORIGINAL + 4 POLICIES)")
        print("=" * 75)
        real_out = out_base / "real"
        res_real = run_real_experiment(
            stage=args.stage,
            max_waves=n_waves,
            gro_config=gro_cfg,
            output_dir=real_out,
            include_baseline=True,
            seed=args.seed,
            verbose=True,
        )
        print("\n[MODE B] BẢNG SO SÁNH 5 CHIẾN LƯỢC (ORIGINAL VS 4 CHÍNH SÁCH GRO):")
        print(res_real.summary_statistics().to_string(index=False))
        # Xuất kiểm định thống kê ra outputs/tables nếu chưa chạy mode paper
        if args.mode == "real":
            tables_dir = Path("outputs/tables")
            res_real.run_statistical_analysis(output_dir=tables_dir)

    print("\nHoàn tất tái lập kết quả! Dữ liệu đã lưu tại:", out_base)
    print("Các bảng thống kê (ANOVA, Tukey, Summary) đã lưu tại: outputs/tables/")


if __name__ == "__main__":
    main()
