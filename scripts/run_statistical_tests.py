#!/usr/bin/env python3
"""
CLI Script to execute Comprehensive Statistical Testing Pipeline.

Executes the formal statistical analysis pipeline:
    Raw Picking Distances
             ↓
    Shapiro-Wilk Normality Test
             ↓
    One-way ANOVA (with η² and ω² effect sizes)
             ↓
    Post-hoc Tukey HSD (marked as Extension)
             ↓
    Pairwise Effect Sizes (Cohen's d, Hedges' g - marked as Extension)
             ↓
    Automated Export of Publication Tables:
        - outputs/tables/summary_statistics.csv
        - outputs/tables/anova.csv
        - outputs/tables/tukey.csv
        - outputs/tables/normality.csv
        - outputs/tables/effect_sizes.csv
        - outputs/tables/statistical_report.md

Usage:
    python scripts/run_statistical_tests.py --input outputs/paper_replication/raw_experiment_results.csv
    python scripts/run_statistical_tests.py --auto
    python scripts/run_statistical_tests.py --fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from slap_gro.evaluation.statistics import run_statistical_pipeline
from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.paper_pipeline import run_paper_experiment


def main():
    parser = argparse.ArgumentParser(
        description="SLAP-GRO: Statistical Testing & Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to raw experiment results CSV (e.g. outputs/paper_replication/raw_experiment_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/tables",
        help="Directory to export ANOVA, Tukey, summary statistics, and markdown reports",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance level alpha (typically 0.05)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically locate existing raw results or run a smoke experiment if none found",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a quick 2-wave experiment and evaluate statistical testing immediately",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Lỗi: Không tìm thấy file dữ liệu đầu vào: {input_path}")
            sys.exit(1)
    elif args.auto:
        # Search candidate paths in priority order
        candidates = [
            Path("outputs/reproduced_results/paper/raw_experiment_results.csv"),
            Path("outputs/paper_replication/raw_experiment_results.csv"),
            Path("outputs/reproduced_results/real/raw_experiment_results.csv"),
            Path("outputs/real_data/raw_experiment_results.csv"),
        ]
        for cand in candidates:
            if cand.exists():
                input_path = cand
                print(f">>> Tự động tìm thấy file dữ liệu thực nghiệm: {cand}")
                break

    # If fast mode or no candidate found under auto, run a quick simulation
    if args.fast or (args.auto and input_path is None):
        print(">>> Chạy thực nghiệm nhanh để tạo dữ liệu đối soát thống kê (2 waves, 4 policies)...")
        fast_gro = GROConfig(
            population_size=50,
            iterations=20,
            tournament_size=3,
            crossover_rate=0.90,
            mutation_rate=0.08,
            elitism=True,
            random_seed=42,
        )
        sim_res = run_paper_experiment(
            n_waves=2,
            items_per_wave=10,
            gro_config=fast_gro,
            include_baseline=True,
            verbose=False,
        )
        df_data = sim_res.to_dataframe()
    elif input_path is not None:
        print(f">>> Đang đọc dữ liệu từ: {input_path}")
        df_data = pd.read_csv(input_path)
    else:
        print("Vui lòng chỉ định file đầu vào bằng --input hoặc dùng cờ --auto / --fast.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("KHỞI CHẠY QUY TRÌNH KIỂM ĐỊNH THỐNG KÊ KHOA HỌC: SLAP-GRO")
    print("=" * 80)

    # Execute end-to-end pipeline
    result = run_statistical_pipeline(
        data=df_data,
        policy_col="policy_name",
        value_col="optimized_distance",
        baseline_policy="original",
        alpha=args.alpha,
        output_dir=out_dir,
    )

    # 1. Summary Statistics
    print("\n[1/5] BẢNG TỔNG HỢP CHỈ SỐ MÔ TẢ CỰ LY & 95% CI:")
    print(result.summary_statistics.to_string(index=False))

    # 2. Shapiro-Wilk Normality
    print("\n[2/5] BẢNG KIỂM ĐỊNH PHÂN PHỐI CHUẨN SHAPIRO-WILK (PAPER METHOD):")
    print(result.normality_table.to_string(index=False))

    # 3. Complete ANOVA Table
    print("\n[3/5] BẢNG PHÂN TÍCH PHƯƠNG SAI ONE-WAY ANOVA HOÀN CHỈNH (PAPER METHOD):")
    print(result.anova_table.to_string(index=False))
    print(f"  * F-statistic     : {result.f_statistic}")
    print(f"  * p-value         : {result.p_value:.6e}")
    print(f"  * Có ý nghĩa (α={args.alpha}): {'CÓ' if result.is_significant else 'KHÔNG'}")
    print(f"  * Eta-squared (η²): {result.eta_squared} ({result.eta_squared * 100:.2f}% phương sai)")
    print(f"  * Omega-squared(ω²): {result.omega_squared}")

    # 4. Tukey HSD Post-hoc
    print("\n[4/5] BẢNG KIỂM ĐỊNH HẬU NGHIỆM TUKEY HSD [EXTENSION]:")
    print(result.tukey_table.to_string(index=False))

    # 5. Pairwise Effect Sizes
    print("\n[5/5] MA TRẬN KÍCH THƯỚC HIỆU ỨNG COHEN'S d & HEDGES' g [EXTENSION]:")
    print(result.effect_sizes_table.to_string(index=False))

    # Saved files summary
    saved = result.save_tables(out_dir)
    print("\n" + "-" * 80)
    print(f"Đã xuất đầy đủ các bảng dữ liệu thống kê ra thư mục: {out_dir}")
    for name, path in saved.items():
        print(f"  * {name:<20}: {path.name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
