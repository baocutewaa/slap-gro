"""
Package slap_gro.evaluation
===========================

Cung cấp đầy đủ các công cụ đánh giá số liệu, phân tích thống kê và trực quan hóa:
- metrics: Tính toán cự ly, mean, median, std, min, max, quartiles Q1/Q3, IQR, % cải thiện.
- statistics: Phân tích phân phối, khoảng tin cậy, kiểm định Shapiro-Wilk, ANOVA, Tukey HSD.
- visualization: Vẽ Boxplot, Scatter so sánh trước/sau GRO, Bar chart kèm sai số, đường cong hội tụ, bản đồ lộ trình 2D.
"""

from slap_gro.evaluation.metrics import (
    DistanceMetrics,
    compute_distance_metrics,
    compute_improvement,
    compute_policy_metrics,
    evaluate_experiment_run,
    export_metrics_table,
)
from slap_gro.evaluation.statistics import (
    DistributionSummary,
    StatisticalPipelineResult,
    compare_policies_pairwise,
    compute_anova_table,
    compute_cohens_d,
    compute_confidence_interval,
    compute_distribution_summary,
    compute_effect_sizes,
    compute_hedges_g,
    compute_relative_improvements,
    evaluate_normality_all_groups,
    run_one_way_anova,
    run_statistical_pipeline,
    run_tukey_hsd,
    shapiro_normality_test,
    test_normality_shapiro,
)
from slap_gro.evaluation.visualization import (
    plot_convergence_curve,
    plot_distance_boxplot,
    plot_distance_scatter,
    plot_effect_sizes_bar,
    plot_picking_route_2d,
    plot_policy_comparison_bar,
    plot_residuals_diagnostics,
    plot_route_wave,
    plot_storage_policy_layout,
    plot_tukey_hsd_forest,
    plot_warehouse_3d,
    generate_all_phase15_figures,
)

__all__ = [
    # Metrics
    "DistanceMetrics",
    "compute_distance_metrics",
    "compute_improvement",
    "compute_policy_metrics",
    "evaluate_experiment_run",
    "export_metrics_table",
    # Statistical Inference
    "DistributionSummary",
    "StatisticalPipelineResult",
    "compute_distribution_summary",
    "compute_confidence_interval",
    "compute_relative_improvements",
    "compare_policies_pairwise",
    "shapiro_normality_test",
    "test_normality_shapiro",
    "evaluate_normality_all_groups",
    "run_one_way_anova",
    "compute_anova_table",
    "run_tukey_hsd",
    "compute_cohens_d",
    "compute_hedges_g",
    "compute_effect_sizes",
    "run_statistical_pipeline",
    # Visualization
    "plot_distance_boxplot",
    "plot_distance_scatter",
    "plot_policy_comparison_bar",
    "plot_convergence_curve",
    "plot_picking_route_2d",
    "plot_tukey_hsd_forest",
    "plot_effect_sizes_bar",
    "plot_residuals_diagnostics",
    "plot_warehouse_3d",
    "plot_storage_policy_layout",
    "plot_route_wave",
    "generate_all_phase15_figures",
]
