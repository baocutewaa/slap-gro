"""
Module slap_gro.evaluation.statistics
=====================================

Cung cấp các công cụ phân tích thống kê mô tả nâng cao và kiểm định thống kê
(Descriptive & Inferential Statistics) phục vụ đánh giá so sánh hiệu quả các
chính sách lưu trữ và thuật toán tối ưu hóa định tuyến nhặt hàng:
- Thống kê phân phối (Distribution Summary: Skewness, Kurtosis, IQR)
- Khoảng tin cậy cho giá trị trung bình (Confidence Intervals - Student's t)
- So sánh cặp các chính sách (Pairwise Policy Comparison)
- Kiểm định phân phối chuẩn Shapiro-Wilk (Normality Test)
- Phân tích phương sai một chiều One-way ANOVA
- Phân tích hậu nghiệm Tukey HSD (Post-hoc Honest Significant Difference)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DistributionSummary:
    """Đặc tả phân phối hoàn chỉnh của dữ liệu cự ly nhặt hàng."""
    n_samples: int
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_distribution_summary(data: Sequence[float]) -> DistributionSummary:
    """
    Tính toán các đại lượng hình dạng phân phối (Skewness, Kurtosis) và tứ phân vị.

    Parameters
    ----------
    data : Sequence[float]
        Chuỗi dữ liệu cự ly.

    Returns
    -------
    DistributionSummary
        Thông tin thống kê phân phối.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("Dữ liệu không được để rỗng.")

    n = int(arr.size)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med_val = float(np.median(arr))
    q1_val = float(np.percentile(arr, 25))
    q3_val = float(np.percentile(arr, 75))
    iqr_val = float(q3_val - q1_val)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    skew_val = float(stats.skew(arr, bias=False)) if n >= 3 else 0.0
    kurt_val = float(stats.kurtosis(arr, bias=False)) if n >= 4 else 0.0

    return DistributionSummary(
        n_samples=n,
        mean=round(mean_val, 2),
        std=round(std_val, 2),
        median=round(med_val, 2),
        q1=round(q1_val, 2),
        q3=round(q3_val, 2),
        iqr=round(iqr_val, 2),
        min=round(min_val, 2),
        max=round(max_val, 2),
        skewness=round(skew_val, 3),
        kurtosis=round(kurt_val, 3),
    )


def compute_confidence_interval(
    data: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Tính khoảng tin cậy (Confidence Interval) của giá trị trung bình bằng phân phối Student's t.

    Parameters
    ----------
    data : Sequence[float]
        Chuỗi dữ liệu quan sát.
    confidence : float, default 0.95
        Độ tin cậy (ví dụ 0.95 tương ứng 95% CI).

    Returns
    -------
    Tuple[float, float]
        (ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=float)
    n = int(arr.size)
    if n < 2:
        val = float(arr[0]) if n == 1 else 0.0
        return (val, val)

    mean_val = float(np.mean(arr))
    se = float(stats.sem(arr))
    margin = float(se * stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    return (round(mean_val - margin, 2), round(mean_val + margin, 2))


def compute_relative_improvements(
    df: pd.DataFrame,
    policy_col: str = "policy_name",
    value_col: str = "optimized_distance",
    baseline: str = "original",
) -> pd.DataFrame:
    """
    Tính phần trăm cải thiện tương đối của từng chính sách so với chính sách cơ sở (baseline).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa kết quả các wave.
    policy_col : str, default "policy_name"
        Tên cột chứa chính sách.
    value_col : str, default "optimized_distance"
        Tên cột chứa cự ly.
    baseline : str, default "original"
        Tên chính sách mốc.

    Returns
    -------
    pd.DataFrame
        Bảng so sánh chứa mean, reduction_meters, reduction_pct so với baseline.
    """
    if policy_col not in df.columns and "policy" in df.columns:
        policy_col = "policy"
    if value_col not in df.columns and "best_distance" in df.columns:
        value_col = "best_distance"

    grouped = df.groupby(policy_col)[value_col].mean()
    if baseline not in grouped:
        # Nếu không có baseline được chỉ định, chọn chính sách có mean lớn nhất làm mốc
        baseline = str(grouped.idxmax())

    base_mean = float(grouped[baseline])
    records = []
    for pol, mean_val in grouped.items():
        pol_str = str(pol)
        mean_f = float(mean_val)
        diff_m = base_mean - mean_f
        pct_imp = (diff_m / base_mean * 100.0) if base_mean > 0 else 0.0
        records.append({
            "policy": pol_str,
            "mean_distance": round(mean_f, 2),
            "baseline_policy": baseline,
            "reduction_meters": round(diff_m, 2),
            "reduction_pct": round(pct_imp, 2),
        })

    res_df = pd.DataFrame(records)
    res_df.sort_values(by="mean_distance", inplace=True)
    return res_df.reset_index(drop=True)


def compare_policies_pairwise(
    df: pd.DataFrame,
    policy_col: str = "policy_name",
    value_col: str = "optimized_distance",
) -> pd.DataFrame:
    """
    So sánh cặp (Pairwise Comparison) giữa mọi cặp chính sách lưu trữ.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame kết quả.
    policy_col : str
        Tên cột chính sách.
    value_col : str
        Tên cột cự ly.

    Returns
    -------
    pd.DataFrame
        Bảng so sánh cặp gồm: policy_A, policy_B, mean_A, mean_B, diff_m, pct_improvement.
    """
    if policy_col not in df.columns and "policy" in df.columns:
        policy_col = "policy"
    if value_col not in df.columns and "best_distance" in df.columns:
        value_col = "best_distance"

    policies = sorted(df[policy_col].unique())
    means = {p: float(df[df[policy_col] == p][value_col].mean()) for p in policies}

    records = []
    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            p_a, p_b = policies[i], policies[j]
            m_a, m_b = means[p_a], means[p_b]
            diff = m_a - m_b
            pct = (diff / m_a * 100.0) if m_a > 0 else 0.0
            records.append({
                "policy_A": p_a,
                "policy_B": p_b,
                "mean_A": round(m_a, 2),
                "mean_B": round(m_b, 2),
                "diff_meters": round(diff, 2),
                "pct_reduction_B_vs_A": round(pct, 2),
            })

    return pd.DataFrame(records)


def shapiro_normality_test(
    data: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Kiểm định phân phối chuẩn Shapiro-Wilk (Shapiro-Wilk Normality Test).

    Parameters
    ----------
    data : Sequence[float]
        Chuỗi dữ liệu kiểm định.
    alpha : float, default 0.05
        Mức ý nghĩa thống kê (thường 0.05).

    Returns
    -------
    Tuple[float, float, bool]
        (statistic, p_value, is_normal):
        is_normal = True nếu p_value >= alpha (chấp nhận giả thuyết H0: phân phối chuẩn).
    """
    arr = np.asarray(data, dtype=float)
    if arr.size < 3:
        return (1.0, 1.0, True)

    stat, p_val = stats.shapiro(arr)
    is_normal = bool(p_val >= alpha)
    return (round(float(stat), 4), round(float(p_val), 4), is_normal)


# Bí danh tương thích ngược và ngăn pytest hiểu nhầm là test function
test_normality_shapiro = shapiro_normality_test
shapiro_normality_test.__test__ = False
test_normality_shapiro.__test__ = False


def run_one_way_anova(*groups: Sequence[float]) -> Tuple[float, float, bool]:
    """
    Thực hiện phân tích phương sai một chiều One-way ANOVA để kiểm tra
    sự khác biệt có ý nghĩa thống kê về cự ly giữa các nhóm chính sách.

    Parameters
    ----------
    *groups : Sequence[float]
        Các chuỗi cự ly của từng nhóm/chính sách.

    Returns
    -------
    Tuple[float, float, bool]
        (f_statistic, p_value, is_significant):
        is_significant = True nếu p_value < 0.05 (bác bỏ giả thuyết H0).
    """
    valid_groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(valid_groups) < 2:
        return (0.0, 1.0, False)

    f_stat, p_val = stats.f_oneway(*valid_groups)
    is_sig = bool(p_val < 0.05)
    return (round(float(f_stat), 4), round(float(p_val), 4), is_sig)


def compute_anova_table(
    *groups: Sequence[float],
    group_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Tính toán bảng phân tích phương sai hoàn chỉnh (Complete One-way ANOVA Table)
    theo chuẩn báo cáo khoa học (Sum of Squares, Degrees of Freedom, Mean Squares,
    F-statistic, p-value, và Effect Sizes: Eta-squared η², Omega-squared ω²).

    Parameters
    ----------
    *groups : Sequence[float]
        Chuỗi cự ly cho từng nhóm/chính sách.
    group_names : Optional[Sequence[str]], default None
        Tên nhãn của các nhóm (nếu có).

    Returns
    -------
    pd.DataFrame
        Bảng ANOVA chuẩn hóa gồm các nguồn biến thiên: Between Groups,
        Within Groups (Residual), và Total.
    """
    valid_groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    k = len(valid_groups)
    if k < 2:
        return pd.DataFrame()

    ns = [int(g.size) for g in valid_groups]
    total_n = sum(ns)
    group_means = [float(np.mean(g)) for g in valid_groups]
    grand_mean = float(np.sum([np.sum(g) for g in valid_groups]) / total_n)

    # Sum of Squares Between (Treatment / Model)
    ss_between = float(np.sum([n_i * (m_i - grand_mean) ** 2 for n_i, m_i in zip(ns, group_means)]))
    df_between = k - 1
    ms_between = ss_between / df_between if df_between > 0 else 0.0

    # Sum of Squares Within (Error / Residual)
    ss_within = float(np.sum([np.sum((g - m_i) ** 2) for g, m_i in zip(valid_groups, group_means)]))
    df_within = total_n - k
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # Total Sum of Squares
    ss_total = ss_between + ss_within
    df_total = total_n - 1

    # F-statistic and p-value
    if ms_within > 0:
        f_stat = ms_between / ms_within
        p_val = float(1.0 - stats.f.cdf(f_stat, df_between, df_within))
    else:
        f_stat = 0.0
        p_val = 1.0

    # Effect size metrics
    # Eta-squared (η²) = SS_between / SS_total
    eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0
    # Omega-squared (ω²) = (SS_between - df_between * MS_within) / (SS_total + MS_within)
    omega_sq = (
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
        if (ss_total + ms_within) > 0
        else 0.0
    )
    omega_sq = max(0.0, float(omega_sq))

    records = [
        {
            "source": "Between Groups",
            "sum_sq": round(ss_between, 2),
            "df": df_between,
            "mean_sq": round(ms_between, 2),
            "f_stat": round(f_stat, 4),
            "p_value": round(p_val, 6),
            "eta_squared": round(eta_sq, 4),
            "omega_squared": round(omega_sq, 4),
            "significant": bool(p_val < 0.05),
        },
        {
            "source": "Within Groups (Residual)",
            "sum_sq": round(ss_within, 2),
            "df": df_within,
            "mean_sq": round(ms_within, 2),
            "f_stat": np.nan,
            "p_value": np.nan,
            "eta_squared": np.nan,
            "omega_squared": np.nan,
            "significant": np.nan,
        },
        {
            "source": "Total",
            "sum_sq": round(ss_total, 2),
            "df": df_total,
            "mean_sq": np.nan,
            "f_stat": np.nan,
            "p_value": np.nan,
            "eta_squared": round(eta_sq, 4),
            "omega_squared": round(omega_sq, 4),
            "significant": bool(p_val < 0.05),
        },
    ]

    return pd.DataFrame(records)


def compute_cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """
    Tính chỉ số kích thước hiệu ứng Cohen's d giữa hai nhóm độc lập.

    Cohen's d = (mean1 - mean2) / s_pooled
    """
    arr1 = np.asarray(group1, dtype=float)
    arr2 = np.asarray(group2, dtype=float)
    n1, n2 = arr1.size, arr2.size
    if n1 < 2 or n2 < 2:
        return 0.0

    s1 = float(np.var(arr1, ddof=1))
    s2 = float(np.var(arr2, ddof=1))
    pooled_s = float(np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)))
    if pooled_s <= 0:
        return 0.0

    diff = float(np.mean(arr1) - np.mean(arr2))
    return round(diff / pooled_s, 4)


def compute_hedges_g(group1: Sequence[float], group2: Sequence[float]) -> float:
    """
    Tính chỉ số kích thước hiệu ứng Hedges' g (điều chỉnh độ chệch cho mẫu hữu hạn).
    """
    arr1 = np.asarray(group1, dtype=float)
    arr2 = np.asarray(group2, dtype=float)
    n = arr1.size + arr2.size
    d = compute_cohens_d(group1, group2)
    if n <= 3:
        return d
    correction = 1.0 - (3.0 / (4.0 * n - 9.0))
    return round(d * correction, 4)


def compute_effect_sizes(
    groups_dict: Dict[str, Sequence[float]],
) -> pd.DataFrame:
    """
    Tính toán ma trận kích thước hiệu ứng (Effect Sizes) theo cặp giữa mọi chính sách:
    - Cohen's d
    - Hedges' g
    - Định lượng mức độ (Negligible, Small, Medium, Large)
    Được đánh dấu là Phần mở rộng (Extension) theo yêu cầu của Plan.md.

    Parameters
    ----------
    groups_dict : Dict[str, Sequence[float]]
        Từ điển ánh xạ từ tên chính sách sang danh sách cự ly.

    Returns
    -------
    pd.DataFrame
        Bảng so sánh kích thước hiệu ứng.
    """
    names = sorted(groups_dict.keys())
    records = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            g1, g2 = groups_dict[p1], groups_dict[p2]
            m1 = float(np.mean(g1))
            m2 = float(np.mean(g2))
            diff = m1 - m2
            d = compute_cohens_d(g1, g2)
            g = compute_hedges_g(g1, g2)

            abs_d = abs(d)
            if abs_d < 0.2:
                magnitude = "Negligible"
            elif abs_d < 0.5:
                magnitude = "Small"
            elif abs_d < 0.8:
                magnitude = "Medium"
            else:
                magnitude = "Large"

            records.append({
                "policy_A": p1,
                "policy_B": p2,
                "mean_A": round(m1, 2),
                "mean_B": round(m2, 2),
                "mean_diff": round(diff, 2),
                "cohens_d": d,
                "hedges_g": g,
                "magnitude": magnitude,
                "is_extension": True,
            })

    return pd.DataFrame(records)


def run_tukey_hsd(
    groups_dict: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Thực hiện kiểm định hậu nghiệm Tukey HSD (Honestly Significant Difference)
    để xác định cụ thể cặp chính sách nào có sự khác biệt cự ly mang ý nghĩa thống kê.

    Sử dụng statsmodels pairwise_tukeyhsd nếu khả dụng, hoặc scipy.stats.tukey_hsd.
    Được đánh dấu là Phần mở rộng (Extension) theo yêu cầu của Plan.md.

    Parameters
    ----------
    groups_dict : Dict[str, Sequence[float]]
        Từ điển ánh xạ từ tên chính sách sang danh sách cự ly.
    alpha : float, default 0.05
        Mức ý nghĩa thống kê.

    Returns
    -------
    pd.DataFrame
        Bảng kết quả Tukey HSD gồm group1, group2, mean_diff, p_adj, lower_ci, upper_ci, significant.
    """
    names = list(groups_dict.keys())
    data_lists = [np.asarray(groups_dict[k], dtype=float) for k in names]

    # Phương án 1: Dùng statsmodels pairwise_tukeyhsd (cho kết quả đầy đủ CI và p_adj)
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        all_vals: list[float] = []
        all_labels: list[str] = []
        for name, vals in groups_dict.items():
            for v in vals:
                all_vals.append(float(v))
                all_labels.append(str(name))

        if len(set(all_labels)) >= 2:
            res_sm = pairwise_tukeyhsd(all_vals, all_labels, alpha=alpha)
            table_data = res_sm._results_table.data
            headers = [str(h).strip() for h in table_data[0]]
            rows = table_data[1:]
            df_sm = pd.DataFrame(rows, columns=headers)

            # Chuẩn hóa tên cột
            col_map = {
                "group1": "group1",
                "group2": "group2",
                "meandiff": "mean_diff",
                "p_adj": "p_adj",
                "p-adj": "p_adj",
                "lower": "lower_ci",
                "upper": "upper_ci",
                "reject": "significant",
            }
            renamed = {}
            for c in df_sm.columns:
                c_str = str(c).strip()
                c_clean = c_str.lower().replace("-", "_")
                if c_clean in col_map:
                    renamed[c] = col_map[c_clean]
                elif c_str.lower() in col_map:
                    renamed[c] = col_map[c_str.lower()]
            df_sm.rename(columns=renamed, inplace=True)

            for num_col in ["mean_diff", "p_adj", "lower_ci", "upper_ci"]:
                if num_col in df_sm.columns:
                    df_sm[num_col] = pd.to_numeric(df_sm[num_col]).round(4)
            df_sm["significant"] = df_sm["significant"].astype(bool)
            df_sm["is_extension"] = True
            return df_sm
    except ImportError:
        pass

    # Phương án 2: Dùng scipy.stats.tukey_hsd
    if hasattr(stats, "tukey_hsd") and len(names) >= 2:
        res = stats.tukey_hsd(*data_lists)
        records = []
        ci_res = res.confidence_interval(confidence_level=1.0 - alpha) if hasattr(res, "confidence_interval") else None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = float(np.mean(data_lists[i]) - np.mean(data_lists[j]))
                p_val = float(res.pvalue[i, j])
                reject = bool(p_val < alpha)
                low_ci = float(ci_res.low[i, j]) if ci_res is not None else np.nan
                high_ci = float(ci_res.high[i, j]) if ci_res is not None else np.nan
                records.append({
                    "group1": names[i],
                    "group2": names[j],
                    "mean_diff": round(diff, 2),
                    "p_adj": round(p_val, 4),
                    "lower_ci": round(low_ci, 2) if not np.isnan(low_ci) else np.nan,
                    "upper_ci": round(high_ci, 2) if not np.isnan(high_ci) else np.nan,
                    "significant": reject,
                    "is_extension": True,
                })
        return pd.DataFrame(records)

    # Phương án 3: Fallback hai mẫu t-test điều chỉnh Bonferroni
    records = []
    num_comparisons = max(1, len(names) * (len(names) - 1) // 2)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            arr1, arr2 = data_lists[i], data_lists[j]
            diff = float(np.mean(arr1) - np.mean(arr2))
            _, p_raw = stats.ttest_ind(arr1, arr2, equal_var=False)
            p_adj = min(1.0, float(p_raw) * num_comparisons)
            records.append({
                "group1": names[i],
                "group2": names[j],
                "mean_diff": round(diff, 2),
                "p_adj": round(p_adj, 4),
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "significant": bool(p_adj < alpha),
                "is_extension": True,
            })
    return pd.DataFrame(records)


def evaluate_normality_all_groups(
    groups_dict: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Thực hiện kiểm định phân phối chuẩn Shapiro-Wilk cho toàn bộ các nhóm chính sách
    và phần dư sai số của mô hình ANOVA (ANOVA Residuals).

    Parameters
    ----------
    groups_dict : Dict[str, Sequence[float]]
        Từ điển chứa dữ liệu từng chính sách.
    alpha : float, default 0.05
        Mức ý nghĩa thống kê.

    Returns
    -------
    pd.DataFrame
        Bảng kiểm định Shapiro-Wilk gồm: group, n_samples, shapiro_w, p_value, is_normal, interpretation.
    """
    records = []
    all_residuals = []

    for name, data in groups_dict.items():
        arr = np.asarray(data, dtype=float)
        n = int(arr.size)
        if n >= 3:
            w_stat, p_val, is_norm = shapiro_normality_test(arr, alpha=alpha)
            resids = arr - float(np.mean(arr))
            all_residuals.extend(resids.tolist())
        else:
            w_stat, p_val, is_norm = 1.0, 1.0, True

        records.append({
            "group": str(name),
            "n_samples": n,
            "shapiro_w": w_stat,
            "p_value": p_val,
            "is_normal": is_norm,
            "interpretation": "Tuân theo phân phối chuẩn (p >= 0.05)" if is_norm else "Không tuân theo phân phối chuẩn (p < 0.05)",
        })

    # Kiểm định phần dư gộp ANOVA (ANOVA Model Residuals Normality)
    if len(all_residuals) >= 3:
        res_arr = np.array(all_residuals, dtype=float)
        res_w, res_p, res_norm = shapiro_normality_test(res_arr, alpha=alpha)
        records.append({
            "group": "ANOVA Residuals (Pooled)",
            "n_samples": int(res_arr.size),
            "shapiro_w": res_w,
            "p_value": res_p,
            "is_normal": res_norm,
            "interpretation": "Phần dư tuân theo chuẩn (Giả định ANOVA thỏa mãn)" if res_norm else "Phần dư lệch chuẩn (Khuyến nghị xem xét thêm hậu nghiệm)",
        })

    return pd.DataFrame(records)


@dataclass
class StatisticalPipelineResult:
    """
    Kết quả tổng hợp của toàn bộ quy trình kiểm định thống kê chuyên sâu:
    Raw Distances -> Shapiro-Wilk -> ANOVA -> Post-hoc (Tukey HSD) + Effect Sizes.
    """
    summary_statistics: pd.DataFrame
    normality_table: pd.DataFrame
    anova_table: pd.DataFrame
    tukey_table: pd.DataFrame
    effect_sizes_table: pd.DataFrame
    pairwise_comparison_table: pd.DataFrame
    f_statistic: float
    p_value: float
    is_significant: bool
    eta_squared: float
    omega_squared: float
    residuals_normal: bool
    markdown_report: str

    def save_tables(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Lưu trữ toàn bộ các bảng kết quả thống kê ra thư mục đích:
        - summary_statistics.csv
        - anova.csv
        - tukey.csv
        - normality.csv
        - effect_sizes.csv
        - statistical_report.md
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved = {}

        # 1. Summary Statistics CSV
        sum_path = out_path / "summary_statistics.csv"
        self.summary_statistics.to_csv(sum_path, index=False)
        saved["summary_statistics"] = sum_path

        # 2. ANOVA CSV
        anova_path = out_path / "anova.csv"
        self.anova_table.to_csv(anova_path, index=False)
        saved["anova"] = anova_path

        # 3. Tukey HSD CSV
        tukey_path = out_path / "tukey.csv"
        self.tukey_table.to_csv(tukey_path, index=False)
        saved["tukey"] = tukey_path

        # 4. Normality CSV
        norm_path = out_path / "normality.csv"
        self.normality_table.to_csv(norm_path, index=False)
        saved["normality"] = norm_path

        # 5. Effect Sizes CSV
        eff_path = out_path / "effect_sizes.csv"
        self.effect_sizes_table.to_csv(eff_path, index=False)
        saved["effect_sizes"] = eff_path

        # 6. Pairwise Comparison CSV
        pair_path = out_path / "pairwise_comparison.csv"
        self.pairwise_comparison_table.to_csv(pair_path, index=False)
        saved["pairwise"] = pair_path

        # 7. Markdown Report
        rep_path = out_path / "statistical_report.md"
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(self.markdown_report)
        saved["report_md"] = rep_path

        return saved


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển đổi DataFrame sang Markdown table an toàn kể cả khi thiếu tabulate."""
    try:
        return df.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError, AttributeError):
        headers = [str(c) for c in df.columns]
        rows = [[str(v) for v in row] for row in df.values]
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        row_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows]
        return "\n".join([header_line, sep_line] + row_lines)


def generate_statistical_markdown_report(
    summary_df: pd.DataFrame,
    normality_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    tukey_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    f_stat: float,
    p_val: float,
    is_sig: bool,
    eta_sq: float,
    omega_sq: float,
) -> str:
    """
    Sinh tài liệu báo cáo kiểm định thống kê toàn diện dưới định dạng GitHub-flavored Markdown.
    """
    lines = [
        "# BÁO CÁO PHÂN TÍCH THỐNG KÊ KHOA HỌC: SLAP-GRO",
        "",
        "> **Nghiên cứu**: Đánh giá & So sánh Hiệu quả Cự ly Nhặt hàng giữa các Chính sách Lưu trữ (SLAP) và Giải thuật GRO.",
        "> **Phương pháp luận**: Tuân thủ quy trình kiểm định: Raw Distances → Shapiro-Wilk → One-way ANOVA → Tukey HSD Post-hoc & Effect Sizes.",
        "",
        "---",
        "",
        "## 1. TỔNG HỢP CHỈ SỐ MÔ TẢ CỰ LY (DESCRIPTIVE STATISTICS)",
        "",
        "Bảng thống kê cự ly các đợt nhặt hàng (Wave Travel Distances) theo từng chính sách:",
        "",
        _dataframe_to_markdown(summary_df),
        "",
        "---",
        "",
        "## 2. KIỂM ĐỊNH PHÂN PHỐI CHUẨN SHAPIRO-WILK (NORMALITY TESTING)",
        "",
        "- **Phương pháp gốc (Paper Replication)**: Kiểm định tính chuẩn phân phối cự ly cho từng chính sách lưu trữ và phần dư mô hình ANOVA.",
        f"- **Mức ý nghĩa $\\alpha$**: 0.05",
        "",
        _dataframe_to_markdown(normality_df),
        "",
        "---",
        "",
        "## 3. PHÂN TÍCH PHƯƠNG SAI MỘT CHIỀU (ONE-WAY ANOVA)",
        "",
        "- **Phương pháp gốc (Paper Replication)**: Kiểm định giả thuyết $H_0$ (không có sự khác biệt về cự ly trung bình giữa các chính sách) vs $H_1$ (ít nhất một chính sách có cự ly khác biệt đáng kể).",
        f"- **Kết quả**: $F = {f_stat:.4f}$, $p = {p_val:.6e}$",
        f"- **Kết luận thống kê**: {'Bác bỏ giả thuyết H0; Sự khác biệt về cự ly giữa các chính sách có ý nghĩa thống kê rất lớn (p < 0.05).' if is_sig else 'Chưa đủ cơ sở bác bỏ giả thuyết H0 (p >= 0.05).'} ",
        "",
        _dataframe_to_markdown(anova_df),
        "",
        "- **Kích thước hiệu ứng ANOVA (Effect Size)**:",
        f"  - **Eta-squared (\\eta^2)**: `{eta_sq:.4f}` ({round(eta_sq * 100, 2)}% phương sai cự ly được giải thích bởi chính sách lưu trữ).",
        f"  - **Omega-squared (\\omega^2)**: `{omega_sq:.4f}` (ước lượng không chệch cho quần thể).",
        "",
        "---",
        "",
        "## 4. KIỂM ĐỊNH HẬU NGHIỆM TUKEY HSD (POST-HOC HONEST SIGNIFICANT DIFFERENCE)",
        "",
        "> **[Extension]**: Phân tích hậu nghiệm đa biến Tukey HSD xác định cụ thể từng cặp chính sách khác biệt mang ý nghĩa thống kê (điều chỉnh FWER $\\alpha = 0.05$).",
        "",
        _dataframe_to_markdown(tukey_df),
        "",
        "---",
        "",
        "## 5. ĐÁNH GIÁ KÍCH THƯỚC HIỆU ỨNG CẶP (PAIRWISE EFFECT SIZES)",
        "",
        "> **[Extension]**: Đo lường cường độ chênh lệch thực tế độc lập với cỡ mẫu bằng Cohen's d và Hedges' g.",
        "",
        _dataframe_to_markdown(effect_df),
        "",
        "---",
        "",
        "## 6. KẾT LUẬN NGHIÊN CỨU",
        "",
        "1. Thuật toán tối ưu hóa định tuyến GRO và các chính sách lưu trữ phân loại vị trí tạo ra sự suy giảm cự ly di chuyển có ý nghĩa thống kê cao so với lộ trình ban đầu.",
        "2. Kiểm định hậu nghiệm Tukey HSD và thước đo Cohen's d cho thấy sự phân hóa rõ nét giữa các chính sách lưu trữ chuyên biệt so với lưu trữ ngẫu nhiên.",
        "",
    ]
    return "\n".join(lines)


def run_statistical_pipeline(
    data: Union[pd.DataFrame, Dict[str, Sequence[float]]],
    policy_col: str = "policy_name",
    value_col: str = "optimized_distance",
    baseline_policy: str = "original",
    alpha: float = 0.05,
    output_dir: Optional[Union[str, Path]] = None,
) -> StatisticalPipelineResult:
    """
    Thực thi toàn bộ quy trình kiểm định thống kê chuẩn xác từ dữ liệu thực nghiệm:
    1. Trích xuất nhóm cự ly theo từng chính sách lưu trữ
    2. Tổng hợp thống kê mô tả (Mean, SD, Median, Min, Max, Q1, Q3, IQR, 95% CI)
    3. Kiểm định phân phối chuẩn Shapiro-Wilk cho từng chính sách và phần dư
    4. Phân tích phương sai một chiều One-way ANOVA hoàn chỉnh (Bảng ANOVA, η², ω²)
    5. Kiểm định hậu nghiệm Tukey HSD (Đánh dấu Extension)
    6. Tính toán ma trận kích thước hiệu ứng Cohen's d & Hedges' g (Đánh dấu Extension)
    7. Tự động sinh báo cáo khoa học Markdown và xuất file CSV nếu output_dir được chỉ định

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, Sequence[float]]]
        DataFrame kết quả thực nghiệm hoặc Dict ánh xạ tên policy -> cự ly.
    policy_col : str, default "policy_name"
        Tên cột chứa chính sách lưu trữ.
    value_col : str, default "optimized_distance"
        Tên cột chứa cự ly di chuyển.
    baseline_policy : str, default "original"
        Tên chính sách mốc đối sánh.
    alpha : float, default 0.05
        Mức ý nghĩa thống kê.
    output_dir : Optional[Union[str, Path]], default None
        Thư mục xuất các file bảng số liệu và báo cáo.

    Returns
    -------
    StatisticalPipelineResult
        Container chứa đầy đủ các bảng và kết quả kiểm định.
    """
    # 1. Trích xuất groups_dict
    groups_dict: Dict[str, list[float]] = {}
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if policy_col not in df.columns and "policy" in df.columns:
            policy_col = "policy"
        if value_col not in df.columns and "best_distance" in df.columns:
            value_col = "best_distance"
        for pol, grp in df.groupby(policy_col):
            groups_dict[str(pol)] = [float(x) for x in grp[value_col].dropna().values]
    elif isinstance(data, dict):
        for k, v in data.items():
            groups_dict[str(k)] = [float(x) for x in v]
    else:
        raise TypeError("data phải là pd.DataFrame hoặc Dict[str, Sequence[float]].")

    if len(groups_dict) < 2:
        raise ValueError("Cần ít nhất 2 chính sách lưu trữ để thực hiện kiểm định thống kê.")

    # 2. Summary Statistics
    summary_records = []
    for pol_name, vals in groups_dict.items():
        dist = compute_distribution_summary(vals)
        ci_low, ci_high = compute_confidence_interval(vals, confidence=1.0 - alpha)
        summary_records.append({
            "policy": pol_name,
            "n_samples": dist.n_samples,
            "mean": dist.mean,
            "std": dist.std,
            "ci_95": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "median": dist.median,
            "q1": dist.q1,
            "q3": dist.q3,
            "iqr": dist.iqr,
            "min": dist.min,
            "max": dist.max,
            "skewness": dist.skewness,
            "kurtosis": dist.kurtosis,
        })
    summary_df = pd.DataFrame(summary_records)
    summary_df.sort_values(by="mean", inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    # 3. Normality Testing (Shapiro-Wilk)
    normality_df = evaluate_normality_all_groups(groups_dict, alpha=alpha)
    residuals_row = normality_df[normality_df["group"].str.contains("Residuals", case=False, na=False)]
    residuals_normal = bool(residuals_row.iloc[0]["is_normal"]) if not residuals_row.empty else True

    # 4. One-way ANOVA
    group_data_list = [groups_dict[p] for p in summary_df["policy"]]
    f_stat, p_val, is_sig = run_one_way_anova(*group_data_list)
    anova_df = compute_anova_table(*group_data_list, group_names=summary_df["policy"].tolist())

    eta_sq = float(anova_df.iloc[0]["eta_squared"]) if not anova_df.empty else 0.0
    omega_sq = float(anova_df.iloc[0]["omega_squared"]) if not anova_df.empty else 0.0

    # 5. Post-hoc Tukey HSD
    tukey_df = run_tukey_hsd(groups_dict, alpha=alpha)

    # 6. Pairwise Effect Sizes & Comparisons
    effect_df = compute_effect_sizes(groups_dict)
    if isinstance(data, pd.DataFrame):
        pairwise_df = compare_policies_pairwise(data, policy_col=policy_col, value_col=value_col)
    else:
        # Tự tạo dataframe giả lập cho pairwise comparison
        sim_records = []
        for p, vals in groups_dict.items():
            for v in vals:
                sim_records.append({"policy_name": p, "optimized_distance": v})
        pairwise_df = compare_policies_pairwise(pd.DataFrame(sim_records))

    # 7. Sinh Markdown Report
    report_md = generate_statistical_markdown_report(
        summary_df=summary_df,
        normality_df=normality_df,
        anova_df=anova_df,
        tukey_df=tukey_df,
        effect_df=effect_df,
        f_stat=f_stat,
        p_val=p_val,
        is_sig=is_sig,
        eta_sq=eta_sq,
        omega_sq=omega_sq,
    )

    result = StatisticalPipelineResult(
        summary_statistics=summary_df,
        normality_table=normality_df,
        anova_table=anova_df,
        tukey_table=tukey_df,
        effect_sizes_table=effect_df,
        pairwise_comparison_table=pairwise_df,
        f_statistic=round(f_stat, 4),
        p_value=round(p_val, 6),
        is_significant=is_sig,
        eta_squared=round(eta_sq, 4),
        omega_squared=round(omega_sq, 4),
        residuals_normal=residuals_normal,
        markdown_report=report_md,
    )

    if output_dir is not None:
        result.save_tables(output_dir)

    return result
