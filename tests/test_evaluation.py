"""
Unit tests cho Module slap_gro.evaluation
(metrics, statistics, visualization)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from slap_gro.evaluation import (
    DistanceMetrics,
    DistributionSummary,
    compare_policies_pairwise,
    compute_confidence_interval,
    compute_distance_metrics,
    compute_distribution_summary,
    compute_improvement,
    compute_policy_metrics,
    compute_relative_improvements,
    evaluate_experiment_run,
    export_metrics_table,
    plot_convergence_curve,
    plot_distance_boxplot,
    plot_distance_scatter,
    plot_picking_route_2d,
    plot_policy_comparison_bar,
    run_one_way_anova,
    run_tukey_hsd,
    shapiro_normality_test,
)
from slap_gro.pipeline.experiment import SingleRunResult


def test_compute_distance_metrics_standard():
    """Kiểm tra tính toán chuẩn xác các chỉ số cự ly (mean, median, std, min, max, Q1, Q3, IQR)."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    metrics = compute_distance_metrics(data)

    assert metrics.n_samples == 5
    assert metrics.total_distance == 150.0
    assert metrics.mean == 30.0
    assert metrics.median == 30.0
    assert metrics.min == 10.0
    assert metrics.max == 50.0
    assert metrics.q1 == 20.0
    assert metrics.q3 == 40.0
    assert metrics.iqr == 20.0
    # Sample std of [10, 20, 30, 40, 50] with ddof=1 is sqrt(250) ~ 15.81
    assert pytest.approx(metrics.std, rel=1e-2) == 15.81
    assert metrics.vs_baseline_pct is None


def test_compute_distance_metrics_with_baseline():
    """Kiểm tra chỉ số cải thiện so với baseline."""
    data = [20.0, 30.0, 40.0]  # mean = 30.0
    metrics = compute_distance_metrics(data, baseline_distance=60.0)

    assert metrics.mean == 30.0
    # (60 - 30) / 60 * 100 = 50.0%
    assert metrics.vs_baseline_pct == 50.0


def test_compute_distance_metrics_edge_cases():
    """Kiểm tra xử lý ngoại lệ cho danh sách rỗng và trường hợp 1 phần tử."""
    with pytest.raises(ValueError, match="không được để rỗng"):
        compute_distance_metrics([])

    single = compute_distance_metrics([42.5])
    assert single.n_samples == 1
    assert single.total_distance == 42.5
    assert single.mean == 42.5
    assert single.median == 42.5
    assert single.std == 0.0
    assert single.min == 42.5
    assert single.max == 42.5
    assert single.q1 == 42.5
    assert single.q3 == 42.5
    assert single.iqr == 0.0


def test_compute_improvement():
    """Kiểm tra hàm compute_improvement cho số vô hướng và mảng numpy."""
    # Scalar
    imp = compute_improvement(100.0, 75.0)
    assert imp == 25.0

    # Negative improvement (worse)
    imp_worse = compute_improvement(100.0, 120.0)
    assert imp_worse == -20.0

    # Zero initial handling
    imp_zero = compute_improvement(0.0, 10.0)
    assert imp_zero == 0.0

    # Array
    arr_init = np.array([100.0, 200.0])
    arr_opt = np.array([80.0, 150.0])
    res = compute_improvement(arr_init, arr_opt)
    np.testing.assert_array_equal(res, np.array([20.0, 25.0]))


def test_compute_policy_metrics_dataframe():
    """Kiểm tra bảng tổng hợp đa chính sách compute_policy_metrics."""
    df = pd.DataFrame([
        {"wave_id": "W1", "policy_name": "original", "optimized_distance": 200.0},
        {"wave_id": "W2", "policy_name": "original", "optimized_distance": 220.0},
        {"wave_id": "W1", "policy_name": "random", "optimized_distance": 180.0},
        {"wave_id": "W2", "policy_name": "random", "optimized_distance": 190.0},
        {"wave_id": "W1", "policy_name": "hybrid", "optimized_distance": 120.0},
        {"wave_id": "W2", "policy_name": "hybrid", "optimized_distance": 130.0},
    ])

    summary = compute_policy_metrics(df, baseline_policy="original")

    assert len(summary) == 3
    # Sắp xếp theo mean tăng dần -> hybrid trước, sau đó random, sau đó original
    assert summary.iloc[0]["policy"] == "hybrid"
    assert summary.iloc[0]["mean"] == 125.0
    assert summary.iloc[0]["vs_baseline_pct"] > 0

    assert summary.iloc[-1]["policy"] == "original"
    assert summary.iloc[-1]["mean"] == 210.0

    # Kiểm tra đầy đủ các cột chuẩn
    expected_cols = ["policy", "n_waves", "total_distance", "mean", "median", "std", "min", "q1", "q3", "max", "iqr"]
    for col in expected_cols:
        assert col in summary.columns


def test_evaluate_experiment_run():
    """Kiểm tra trích xuất chỉ số từ SingleRunResult."""
    run = SingleRunResult(
        wave_id="wave_001",
        policy_name="hybrid",
        initial_distance=250.0,
        optimized_distance=150.0,
        improvement_pct=40.0,
        execution_time_seconds=1.25,
        iterations_run=50,
        n_picks=20,
    )

    metrics_dict = evaluate_experiment_run(run)
    assert metrics_dict["wave_id"] == "wave_001"
    assert metrics_dict["policy"] == "hybrid"
    assert metrics_dict["initial_distance"] == 250.0
    assert metrics_dict["optimized_distance"] == 150.0
    assert metrics_dict["improvement_pct"] == 40.0
    assert metrics_dict["execution_time_seconds"] == 1.25


def test_distribution_summary_and_confidence_interval():
    """Kiểm tra thống kê phân phối và khoảng tin cậy 95%."""
    np.random.seed(42)
    sample = np.random.normal(loc=100.0, scale=10.0, size=50)

    dist = compute_distribution_summary(sample)
    assert isinstance(dist, DistributionSummary)
    assert dist.n_samples == 50
    assert pytest.approx(dist.mean, abs=5.0) == 100.0
    assert dist.iqr > 0
    assert -2.0 < dist.skewness < 2.0

    # Confidence interval
    ci_low, ci_high = compute_confidence_interval(sample, confidence=0.95)
    assert ci_low < dist.mean < ci_high


def test_compare_policies_and_relative_improvements():
    """Kiểm tra hàm so sánh cặp và cải thiện tương đối."""
    df = pd.DataFrame([
        {"policy_name": "original", "optimized_distance": 200.0},
        {"policy_name": "random", "optimized_distance": 180.0},
        {"policy_name": "hybrid", "optimized_distance": 120.0},
    ])

    rel_df = compute_relative_improvements(df, baseline="original")
    assert len(rel_df) == 3
    hybrid_row = rel_df[rel_df["policy"] == "hybrid"].iloc[0]
    assert hybrid_row["reduction_meters"] == 80.0
    assert hybrid_row["reduction_pct"] == 40.0

    pair_df = compare_policies_pairwise(df)
    # Với 3 chính sách -> C(3, 2) = 3 cặp so sánh
    assert len(pair_df) == 3
    assert "diff_meters" in pair_df.columns


def test_statistical_tests_shapiro_anova_tukey():
    """Kiểm tra kiểm định Shapiro-Wilk, ANOVA và Tukey HSD."""
    np.random.seed(42)
    group1 = np.random.normal(100.0, 10.0, size=30)
    group2 = np.random.normal(150.0, 10.0, size=30)
    group3 = np.random.normal(200.0, 10.0, size=30)

    # Shapiro-Wilk
    stat, p_val, is_normal = shapiro_normality_test(group1)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= p_val <= 1.0

    # ANOVA
    f_stat, p_val, is_sig = run_one_way_anova(group1, group2, group3)
    assert is_sig is True
    assert p_val < 0.001

    # Tukey HSD
    tukey_res = run_tukey_hsd({"G1": group1, "G2": group2, "G3": group3})
    assert len(tukey_res) == 3
    assert "p_adj" in tukey_res.columns


def test_visualization_plots(tmp_path: Path):
    """Kiểm tra các hàm trực quan hóa biểu đồ và lưu file hình ảnh."""
    df = pd.DataFrame([
        {"policy_name": "random", "initial_distance": 210.0, "optimized_distance": 180.0},
        {"policy_name": "random", "initial_distance": 230.0, "optimized_distance": 190.0},
        {"policy_name": "hybrid", "initial_distance": 200.0, "optimized_distance": 120.0},
        {"policy_name": "hybrid", "initial_distance": 220.0, "optimized_distance": 130.0},
    ])

    box_path = tmp_path / "boxplot.png"
    ax_box = plot_distance_boxplot(df, save_path=box_path)
    assert box_path.exists()
    assert ax_box is not None

    scatter_path = tmp_path / "scatter.png"
    ax_scat = plot_distance_scatter(df, save_path=scatter_path)
    assert scatter_path.exists()
    assert ax_scat is not None

    bar_path = tmp_path / "bar.png"
    metrics_df = compute_policy_metrics(df)
    ax_bar = plot_policy_comparison_bar(metrics_df, save_path=bar_path)
    assert bar_path.exists()
    assert ax_bar is not None

    conv_path = tmp_path / "conv.png"
    ax_conv = plot_convergence_curve([300.0, 250.0, 200.0, 180.0, 150.0], save_path=conv_path)
    assert conv_path.exists()
    assert ax_conv is not None

    route_path = tmp_path / "route.png"
    locs = [(10.0, 15.0), (20.0, 25.0), (30.0, 10.0)]
    ax_route = plot_picking_route_2d(locs, route_order=[0, 1, 2], save_path=route_path)
    assert route_path.exists()
    assert ax_route is not None


def test_export_metrics_table(tmp_path: Path):
    """Kiểm tra xuất bảng số liệu ra CSV và Markdown."""
    df = pd.DataFrame([
        {"policy": "hybrid", "mean": 125.0, "median": 125.0, "std": 7.07},
        {"policy": "random", "mean": 185.0, "median": 185.0, "std": 7.07},
    ])

    csv_out = tmp_path / "summary.csv"
    export_metrics_table(df, csv_out, format="csv")
    assert csv_out.exists()
    loaded_csv = pd.read_csv(csv_out)
    assert len(loaded_csv) == 2

    md_out = tmp_path / "summary.md"
    export_metrics_table(df, md_out, format="markdown")
    assert md_out.exists()
    content = md_out.read_text(encoding="utf-8")
    assert "hybrid" in content
