"""
Module slap_gro.evaluation.metrics
==================================

Cung cấp các hàm và cấu trúc dữ liệu tính toán toàn bộ chỉ số đánh giá cự ly
nhặt hàng (Picking Travel Distance Metrics) theo đúng đặc tả của bài báo khoa học:
- Total Distance (Tổng quãng đường)
- Mean (Trung bình cự ly)
- Median (Trung vị)
- Standard Deviation (Độ lệch chuẩn mẫu, ddof=1)
- Minimum & Maximum (Cực tiểu, Cực đại)
- Quartiles: Q1 (25th percentile), Q3 (75th percentile)
- Interquartile Range (IQR = Q3 - Q1)
- Percentage Improvement vs Baseline (% cải thiện cự ly)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class DistanceMetrics:
    """
    Tập hợp đầy đủ các chỉ số thống kê mô tả cự ly nhặt hàng theo chuẩn bài báo.
    """
    total_distance: float
    n_samples: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    vs_baseline_pct: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành từ điển dữ liệu thuần."""
        res = asdict(self)
        return res

    def to_series(self, name: Optional[str] = None) -> pd.Series:
        """Chuyển đổi thành pd.Series để ghép bảng."""
        return pd.Series(self.to_dict(), name=name)


def compute_distance_metrics(
    distances: Sequence[float],
    baseline_distance: Optional[float] = None,
) -> DistanceMetrics:
    """
    Tính toán toàn diện các chỉ số thống kê mô tả cho chuỗi cự ly nhặt hàng.

    Parameters
    ----------
    distances : Sequence[float]
        Danh sách hoặc mảng chứa cự ly nhặt hàng của các wave (mét).
    baseline_distance : Optional[float], optional
        Cự ly tham chiếu của baseline (ví dụ Original route hoặc Random storage)
        để tính tỷ lệ % cải thiện cự ly.

    Returns
    -------
    DistanceMetrics
        Đối tượng chứa đầy đủ: total, mean, median, std, min, max, q1, q3, iqr, vs_baseline_pct.

    Raises
    ------
    ValueError
        Nếu danh sách distances rỗng.
    """
    arr = np.asarray(distances, dtype=float)
    if arr.size == 0:
        raise ValueError("Danh sách cự ly (distances) không được để rỗng.")

    n_samples = int(arr.size)
    total_dist = float(np.sum(arr))
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n_samples > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    q1_val = float(np.percentile(arr, 25))
    q3_val = float(np.percentile(arr, 75))
    iqr_val = float(q3_val - q1_val)

    vs_baseline: Optional[float] = None
    if baseline_distance is not None and baseline_distance > 0:
        vs_baseline = round(float((baseline_distance - mean_val) / baseline_distance * 100.0), 2)

    return DistanceMetrics(
        total_distance=round(total_dist, 2),
        n_samples=n_samples,
        mean=round(mean_val, 2),
        median=round(median_val, 2),
        std=round(std_val, 2),
        min=round(min_val, 2),
        max=round(max_val, 2),
        q1=round(q1_val, 2),
        q3=round(q3_val, 2),
        iqr=round(iqr_val, 2),
        vs_baseline_pct=vs_baseline,
    )


def compute_improvement(
    initial: Union[float, np.ndarray, Sequence[float]],
    optimized: Union[float, np.ndarray, Sequence[float]],
) -> Union[float, np.ndarray]:
    """
    Tính phần trăm cải thiện (giảm cự ly) giữa cự ly ban đầu và cự ly tối ưu.
    Formula: (initial - optimized) / initial * 100.0

    Parameters
    ----------
    initial : Union[float, np.ndarray, Sequence[float]]
        Cự ly ban đầu trước khi tối ưu.
    optimized : Union[float, np.ndarray, Sequence[float]]
        Cự ly sau khi tối ưu.

    Returns
    -------
    Union[float, np.ndarray]
        Tỷ lệ % cải thiện (dương nghĩa là giảm được cự ly).
    """
    init_arr = np.asarray(initial, dtype=float)
    opt_arr = np.asarray(optimized, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        imp = np.where(init_arr > 0, (init_arr - opt_arr) / init_arr * 100.0, 0.0)

    if np.isscalar(initial) and np.isscalar(optimized):
        return round(float(imp.item()), 2)
    return np.round(imp, 2)


def compute_policy_metrics(
    data: Union[pd.DataFrame, Any],
    baseline_policy: str = "original",
    policy_col: str = "policy_name",
    value_col: str = "optimized_distance",
) -> pd.DataFrame:
    """
    Tính bảng tổng hợp số liệu thống kê đầy đủ (Summary Statistics Table)
    cho tất cả các chính sách lưu trữ (Storage Policies) từ DataFrame hoặc ExperimentResult.

    Bảng kết quả bao gồm các cột chuẩn:
    - policy: Tên chính sách lưu trữ
    - n_waves: Số lượng wave được đánh giá
    - total_distance: Tổng cự ly nhặt hàng (m)
    - mean: Cự ly trung bình (m)
    - median: Trung vị (m)
    - std: Độ lệch chuẩn (m)
    - min: Cự ly nhỏ nhất (m)
    - q1: Phân vị thứ nhất 25% (m)
    - q3: Phân vị thứ ba 75% (m)
    - max: Cự ly lớn nhất (m)
    - iqr: Khoảng tứ phân vị IQR (m)
    - vs_baseline_pct: Tỷ lệ % cải thiện so với baseline_policy (nếu có)

    Parameters
    ----------
    data : Union[pd.DataFrame, Any]
        DataFrame chứa kết quả hoặc đối tượng ExperimentResult.
    baseline_policy : str, default "original"
        Tên chính sách dùng làm mốc so sánh (thường là 'original' hoặc 'random').
    policy_col : str, default "policy_name"
        Tên cột chứa tên chính sách.
    value_col : str, default "optimized_distance"
        Tên cột chứa giá trị cự ly tối ưu.

    Returns
    -------
    pd.DataFrame
        Bảng thống kê mô tả chuẩn hóa, sắp xếp theo mean tăng dần.
    """
    if hasattr(data, "to_dataframe"):
        df = data.to_dataframe()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Dữ liệu đầu vào phải là pandas.DataFrame hoặc ExperimentResult.")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "policy",
                "n_waves",
                "total_distance",
                "mean",
                "median",
                "std",
                "min",
                "q1",
                "q3",
                "max",
                "iqr",
                "vs_baseline_pct",
            ]
        )

    # Chuẩn hóa tên cột nếu DataFrame dùng 'policy' thay vì 'policy_name'
    if policy_col not in df.columns and "policy" in df.columns:
        policy_col = "policy"
    if value_col not in df.columns and "best_distance" in df.columns:
        value_col = "best_distance"

    # Lấy baseline mean nếu baseline_policy tồn tại
    baseline_mean: Optional[float] = None
    baseline_rows = df[df[policy_col] == baseline_policy]
    if not baseline_rows.empty:
        baseline_vals = baseline_rows[value_col].dropna().values
        if len(baseline_vals) > 0:
            baseline_mean = float(np.mean(baseline_vals))

    records: list[dict[str, Any]] = []
    for policy_val, group in df.groupby(policy_col):
        vals = group[value_col].dropna().values
        if len(vals) == 0:
            continue
        m = compute_distance_metrics(vals, baseline_distance=baseline_mean if policy_val != baseline_policy else None)
        row_dict = m.to_dict()
        row_dict["policy"] = str(policy_val)
        row_dict["n_waves"] = row_dict.pop("n_samples")
        records.append(row_dict)

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df

    # Sắp xếp lại thứ tự cột cho tường minh
    cols_order = [
        "policy",
        "n_waves",
        "total_distance",
        "mean",
        "median",
        "std",
        "min",
        "q1",
        "q3",
        "max",
        "iqr",
        "vs_baseline_pct",
    ]
    existing_cols = [c for c in cols_order if c in result_df.columns]
    result_df = result_df[existing_cols]

    # Sắp xếp các chính sách theo mean cự ly tăng dần
    result_df.sort_values(by="mean", inplace=True)
    return result_df.reset_index(drop=True)


def evaluate_experiment_run(run: Any) -> dict[str, Any]:
    """
    Trích xuất chỉ số chi tiết cho một lần chạy đơn lẻ (SingleRunResult).

    Parameters
    ----------
    run : Any
        Đối tượng SingleRunResult từ pipeline thực nghiệm.

    Returns
    -------
    dict[str, Any]
        Từ điển chứa wave_id, policy, cự ly ban đầu, cự ly tối ưu, % cải thiện.
    """
    initial_d = getattr(run, "initial_distance", 0.0)
    opt_d = getattr(run, "optimized_distance", getattr(run, "best_distance", 0.0))
    imp_pct = getattr(run, "improvement_pct", compute_improvement(initial_d, opt_d))

    return {
        "wave_id": getattr(run, "wave_id", "unknown"),
        "policy": getattr(run, "policy_name", getattr(run, "policy", "unknown")),
        "n_picks": getattr(run, "n_picks", getattr(run, "num_picks", 0)),
        "initial_distance": round(float(initial_d), 2),
        "optimized_distance": round(float(opt_d), 2),
        "improvement_pct": round(float(imp_pct), 2),
        "execution_time_seconds": round(float(getattr(run, "execution_time_seconds", 0.0)), 3),
    }


def export_metrics_table(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "csv",
) -> Path:
    """
    Xuất bảng chỉ số đánh giá ra tệp định dạng CSV, LaTeX hoặc Markdown.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Bảng chỉ số đánh giá cự ly đã tính toán.
    output_path : Union[str, Path]
        Đường dẫn file đích.
    format : str, default "csv"
        Định dạng xuất ('csv', 'latex', 'markdown').

    Returns
    -------
    Path
        Đường dẫn tệp đã lưu.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt == "csv":
        metrics_df.to_csv(p, index=False)
    elif fmt == "latex":
        with open(p, "w", encoding="utf-8") as f:
            f.write(metrics_df.to_latex(index=False))
    elif fmt in ("markdown", "md"):
        try:
            md_str = metrics_df.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            headers = [str(c) for c in metrics_df.columns]
            rows = [[str(v) for v in row] for row in metrics_df.values]
            col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
            header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
            sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
            row_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows]
            md_str = "\n".join([header_line, sep_line] + row_lines) + "\n"
        with open(p, "w", encoding="utf-8") as f:
            f.write(md_str)
    else:
        raise ValueError(f"Định dạng không được hỗ trợ: {format}. Chọn 'csv', 'latex', hoặc 'markdown'.")

    return p
