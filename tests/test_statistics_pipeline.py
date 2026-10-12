"""
Unit Tests cho Statistical Testing Pipeline
Kiểm thử toàn bộ các thành phần:
- compute_anova_table (SS, df, MS, F, p, η², ω²)
- compute_cohens_d & compute_hedges_g
- compute_effect_sizes
- evaluate_normality_all_groups (Shapiro-Wilk + ANOVA Residuals)
- run_tukey_hsd (với CI và flag is_extension)
- StatisticalPipelineResult & save_tables
- run_statistical_pipeline (cho DataFrame và Dict)
- ExperimentResult.run_statistical_analysis()
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slap_gro.evaluation.statistics import (
    StatisticalPipelineResult,
    compute_anova_table,
    compute_cohens_d,
    compute_effect_sizes,
    compute_hedges_g,
    evaluate_normality_all_groups,
    run_one_way_anova,
    run_statistical_pipeline,
    run_tukey_hsd,
    shapiro_normality_test,
)
from slap_gro.pipeline.experiment import SingleRunResult, ExperimentResult
from slap_gro.gro.optimizer import GROConfig


def test_compute_anova_table_full():
    """Kiểm tra bảng phân tích phương sai hoàn chỉnh: SS, df, MS, F, p, η², ω²."""
    np.random.seed(42)
    g1 = np.random.normal(100.0, 5.0, size=20)
    g2 = np.random.normal(150.0, 5.0, size=20)
    g3 = np.random.normal(200.0, 5.0, size=20)

    df_anova = compute_anova_table(g1, g2, g3)
    assert len(df_anova) == 3
    assert set(df_anova["source"]) == {"Between Groups", "Within Groups (Residual)", "Total"}

    row_between = df_anova[df_anova["source"] == "Between Groups"].iloc[0]
    row_within = df_anova[df_anova["source"] == "Within Groups (Residual)"].iloc[0]
    row_total = df_anova[df_anova["source"] == "Total"].iloc[0]

    # df: k - 1 = 2; N - k = 57; N - 1 = 59
    assert row_between["df"] == 2
    assert row_within["df"] == 57
    assert row_total["df"] == 59

    # SS_total = SS_between + SS_within
    assert pytest.approx(row_total["sum_sq"], abs=1.0) == row_between["sum_sq"] + row_within["sum_sq"]

    # F > 0, p < 0.001, eta_sq > 0.8
    assert row_between["f_stat"] > 100.0
    assert row_between["p_value"] < 0.001
    assert row_between["significant"] is True
    assert 0.8 < row_between["eta_squared"] <= 1.0
    assert 0.8 < row_between["omega_squared"] <= 1.0


def test_cohens_d_and_hedges_g():
    """Kiểm tra tính toán kích thước hiệu ứng Cohen's d và Hedges' g."""
    g1 = [10.0, 12.0, 11.0, 13.0, 12.0]
    g2 = [20.0, 22.0, 21.0, 23.0, 22.0]

    d = compute_cohens_d(g1, g2)
    g = compute_hedges_g(g1, g2)

    # g1 mean = 11.6, g2 mean = 21.6 -> d < 0 (hoặc ngược lại)
    assert d < -5.0
    assert abs(g) <= abs(d)  # Hedges' g có hiệu chỉnh nhỏ hơn Cohen's d

    # Hai nhóm đồng nhất -> d = 0.0
    assert compute_cohens_d(g1, g1) == 0.0


def test_compute_effect_sizes_table():
    """Kiểm tra bảng tổng hợp kích thước hiệu ứng đa chính sách."""
    groups = {
        "Random": [100.0, 105.0, 95.0, 102.0],
        "Dedicated": [80.0, 85.0, 78.0, 82.0],
        "Hybrid": [50.0, 52.0, 48.0, 51.0],
    }

    df_eff = compute_effect_sizes(groups)
    # C(3, 2) = 3 cặp
    assert len(df_eff) == 3
    assert "cohens_d" in df_eff.columns
    assert "hedges_g" in df_eff.columns
    assert "magnitude" in df_eff.columns
    assert all(df_eff["is_extension"] == True)
    assert df_eff.iloc[0]["magnitude"] in ["Negligible", "Small", "Medium", "Large"]


def test_evaluate_normality_all_groups():
    """Kiểm tra kiểm định phân phối chuẩn Shapiro-Wilk cho từng nhóm và phần dư gộp."""
    np.random.seed(42)
    norm_sample = np.random.normal(100.0, 10.0, size=40)
    groups = {
        "PolA": norm_sample,
        "PolB": norm_sample + 20.0,
    }

    df_norm = evaluate_normality_all_groups(groups)
    assert len(df_norm) == 3  # PolA, PolB, and ANOVA Residuals
    assert any("Residuals" in g for g in df_norm["group"])

    for _, row in df_norm.iterrows():
        assert 0.0 <= row["shapiro_w"] <= 1.0
        assert 0.0 <= row["p_value"] <= 1.0
        assert isinstance(row["is_normal"], bool)


def test_run_tukey_hsd_with_ci():
    """Kiểm tra kiểm định hậu nghiệm Tukey HSD có CI và cờ is_extension."""
    np.random.seed(42)
    groups = {
        "G1": np.random.normal(100.0, 5.0, size=20),
        "G2": np.random.normal(120.0, 5.0, size=20),
        "G3": np.random.normal(150.0, 5.0, size=20),
    }

    df_tukey = run_tukey_hsd(groups)
    assert len(df_tukey) == 3
    assert "group1" in df_tukey.columns
    assert "group2" in df_tukey.columns
    assert "mean_diff" in df_tukey.columns
    assert "p_adj" in df_tukey.columns
    assert "significant" in df_tukey.columns
    assert "is_extension" in df_tukey.columns
    assert all(df_tukey["is_extension"] == True)


def test_run_statistical_pipeline_dataframe(tmp_path: Path):
    """Kiểm tra pipeline kiểm định thống kê từ DataFrame và xuất file bảng số liệu."""
    df = pd.DataFrame([
        {"policy_name": "original", "optimized_distance": 500.0},
        {"policy_name": "original", "optimized_distance": 520.0},
        {"policy_name": "original", "optimized_distance": 510.0},
        {"policy_name": "random", "optimized_distance": 300.0},
        {"policy_name": "random", "optimized_distance": 310.0},
        {"policy_name": "random", "optimized_distance": 290.0},
        {"policy_name": "hybrid", "optimized_distance": 150.0},
        {"policy_name": "hybrid", "optimized_distance": 160.0},
        {"policy_name": "hybrid", "optimized_distance": 140.0},
    ])

    out_dir = tmp_path / "tables"
    result = run_statistical_pipeline(df, output_dir=out_dir)

    assert isinstance(result, StatisticalPipelineResult)
    assert len(result.summary_statistics) == 3
    assert len(result.anova_table) == 3
    assert len(result.tukey_table) == 3
    assert len(result.effect_sizes_table) == 3
    assert result.f_statistic > 50.0
    assert result.p_value < 0.001
    assert result.is_significant is True
    assert result.eta_squared > 0.9

    # Kiểm tra các file đã được ghi thành công
    expected_files = [
        "summary_statistics.csv",
        "anova.csv",
        "tukey.csv",
        "normality.csv",
        "effect_sizes.csv",
        "statistical_report.md",
    ]
    for fn in expected_files:
        assert (out_dir / fn).exists()

    # Nội dung file báo cáo Markdown
    report_content = (out_dir / "statistical_report.md").read_text(encoding="utf-8")
    assert "BÁO CÁO PHÂN TÍCH THỐNG KÊ KHOA HỌC" in report_content
    assert "ONE-WAY ANOVA" in report_content
    assert "TUKEY HSD" in report_content


def test_run_statistical_pipeline_dict():
    """Kiểm tra pipeline kiểm định thống kê với dữ liệu đầu vào là dictionary."""
    data_dict = {
        "A": [100.0, 102.0, 98.0, 101.0],
        "B": [120.0, 122.0, 119.0, 121.0],
    }
    result = run_statistical_pipeline(data_dict)
    assert result.is_significant is True
    assert len(result.summary_statistics) == 2


def test_experiment_result_run_statistical_analysis(tmp_path: Path):
    """Kiểm tra tích hợp ExperimentResult.run_statistical_analysis và auto save_results."""
    runs = [
        SingleRunResult(wave_id=1, policy_name="random", initial_distance=300.0, optimized_distance=200.0),
        SingleRunResult(wave_id=2, policy_name="random", initial_distance=320.0, optimized_distance=210.0),
        SingleRunResult(wave_id=1, policy_name="hybrid", initial_distance=300.0, optimized_distance=120.0),
        SingleRunResult(wave_id=2, policy_name="hybrid", initial_distance=320.0, optimized_distance=130.0),
    ]

    exp_res = ExperimentResult(
        mode="paper_replication",
        gro_config=GROConfig(),
        runs=runs,
        total_execution_time_seconds=0.5,
    )

    stats_res = exp_res.run_statistical_analysis()
    assert isinstance(stats_res, StatisticalPipelineResult)
    assert stats_res.is_significant is True

    # Kiểm tra save_results tự động sinh anova.csv và tukey.csv
    saved = exp_res.save_results(tmp_path)
    assert "anova" in saved
    assert "tukey" in saved
    assert (tmp_path / "anova.csv").exists()
    assert (tmp_path / "tukey.csv").exists()


def test_cli_run_statistical_tests(tmp_path: Path):
    """Kiểm tra chạy kịch bản CLI scripts/run_statistical_tests.py với cờ --fast."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_statistical_tests.py"
    out_dir = tmp_path / "cli_tables"

    cmd = [
        sys.executable,
        str(script_path),
        "--fast",
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "QUY TRÌNH KIỂM ĐỊNH THỐNG KÊ KHOA HỌC" in proc.stdout
    assert (out_dir / "anova.csv").exists()
    assert (out_dir / "tukey.csv").exists()
    assert (out_dir / "summary_statistics.csv").exists()
    assert (out_dir / "statistical_report.md").exists()
