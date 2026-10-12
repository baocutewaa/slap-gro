"""
Module slap_gro.evaluation.visualization
========================================

Cung cấp các hàm trực quan hóa dữ liệu và biểu đồ khoa học (Publication-Quality Plots)
phục vụ đánh giá hiệu năng của thuật toán SLAP-GRO theo chuẩn bài báo:
- Boxplot so sánh phân phối cự ly giữa các chính sách (Distance Boxplot)
- Scatter plot so sánh cự ly trước và sau tối ưu hóa (Initial vs Optimized Scatter)
- Bar plot so sánh cự ly trung bình kèm thanh sai số (Mean Distance Bar Chart)
- Đường cong hội tụ thuật toán di truyền GRO (GRO Convergence Curve)
- Bản đồ trực quan hóa tuyến đường nhặt hàng 2D (2D Picking Route Visualization)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # Chế độ headless cho môi trường kiểm thử và server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Thiết lập phong cách hiển thị biểu đồ hiện đại, chuẩn bài báo
STYLE_COLORS = {
    "original": "#8c564b",
    "random": "#d62728",
    "dedicated": "#ff7f0e",
    "class_based": "#1f77b4",
    "hybrid": "#2ca02c",
}


def _get_policy_color(policy_name: str) -> str:
    norm = str(policy_name).lower().replace("-", "_")
    return STYLE_COLORS.get(norm, "#7f7f7f")


def plot_distance_boxplot(
    data: Union[pd.DataFrame, Dict[str, Sequence[float]]],
    policy_col: str = "policy_name",
    value_col: str = "optimized_distance",
    title: str = "Picking Travel Distance by Storage Policy",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ biểu đồ Boxplot so sánh phân bố cự ly nhặt hàng giữa các chính sách lưu trữ.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, Sequence[float]]]
        DataFrame hoặc Dictionary chứa danh sách cự ly cho mỗi chính sách.
    policy_col : str, default "policy_name"
        Tên cột chính sách (nếu data là DataFrame).
    value_col : str, default "optimized_distance"
        Tên cột cự ly (nếu data là DataFrame).
    title : str
        Tiêu đề biểu đồ.
    save_path : Optional[Union[str, Path]]
        Đường dẫn lưu file ảnh (nếu cần).
    ax : Optional[plt.Axes]
        Axes matplotlib có sẵn (nếu muốn ghép subplot).

    Returns
    -------
    plt.Axes
        Đối tượng Axes đã vẽ.
    """
    if isinstance(data, dict):
        policies = list(data.keys())
        distances = [list(data[k]) for k in policies]
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if policy_col not in df.columns and "policy" in df.columns:
            policy_col = "policy"
        if value_col not in df.columns and "best_distance" in df.columns:
            value_col = "best_distance"
        policies = sorted(df[policy_col].unique())
        distances = [df[df[policy_col] == p][value_col].dropna().tolist() for p in policies]
    else:
        raise TypeError("Dữ liệu đầu vào phải là pandas.DataFrame hoặc dict.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        created_fig = True

    box = ax.boxplot(
        distances,
        patch_artist=True,
        tick_labels=[p.replace("_", "-").title() for p in policies],
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "color": "gray", "alpha": 0.5},
    )

    for patch, policy in zip(box["boxes"], policies):
        color = _get_policy_color(policy)
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Travel Distance (meters)", fontsize=11)
    ax.set_xlabel("Storage Policy", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_distance_scatter(
    df: pd.DataFrame,
    initial_col: str = "initial_distance",
    optimized_col: str = "optimized_distance",
    policy_col: str = "policy_name",
    title: str = "Picking Distance: Before vs After GRO Optimization",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ Scatter Plot so sánh cự ly trước và sau khi tối ưu hóa bằng GRO.
    Các điểm nằm dưới đường phân giác y=x thể hiện cự ly đã được cải thiện (giảm).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa kết quả các wave.
    initial_col : str
        Cột cự ly ban đầu.
    optimized_col : str
        Cột cự ly tối ưu sau GRO.
    policy_col : str
        Cột tên chính sách lưu trữ.
    title : str
        Tiêu đề biểu đồ.
    save_path : Optional[Union[str, Path]]
        Đường dẫn lưu file ảnh.
    ax : Optional[plt.Axes]
        Axes matplotlib.

    Returns
    -------
    plt.Axes
    """
    data = df.copy()
    if policy_col not in data.columns and "policy" in data.columns:
        policy_col = "policy"
    if optimized_col not in data.columns and "best_distance" in data.columns:
        optimized_col = "best_distance"

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        created_fig = True

    policies = sorted(data[policy_col].unique())
    all_init = data[initial_col].dropna().values
    all_opt = data[optimized_col].dropna().values

    min_val = min(float(np.min(all_init)), float(np.min(all_opt))) * 0.9 if len(all_init) > 0 else 0
    max_val = max(float(np.max(all_init)), float(np.max(all_opt))) * 1.05 if len(all_init) > 0 else 100

    # Vẽ đường đối giác y = x làm mốc
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6, label="y = x (No improvement)")

    for pol in policies:
        sub = data[data[policy_col] == pol]
        ax.scatter(
            sub[initial_col],
            sub[optimized_col],
            label=pol.replace("_", "-").title(),
            color=_get_policy_color(pol),
            alpha=0.7,
            edgecolors="w",
            s=50,
        )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Initial Distance (meters)", fontsize=11)
    ax.set_ylabel("Optimized Distance (meters)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_policy_comparison_bar(
    metrics_df: pd.DataFrame,
    value_col: str = "mean",
    err_col: str = "std",
    title: str = "Mean Picking Distance by Storage Policy (with SD)",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ biểu đồ cột (Bar Chart) so sánh cự ly trung bình giữa các chính sách,
    kèm thanh sai số độ lệch chuẩn (Standard Deviation Error Bars).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Bảng chỉ số đánh giá (output từ compute_policy_metrics).
    value_col : str, default "mean"
        Cột giá trị trung bình.
    err_col : str, default "std"
        Cột sai số độ lệch chuẩn.
    title : str
        Tiêu đề biểu đồ.
    save_path : Optional[Union[str, Path]]
        Đường dẫn file ảnh.
    ax : Optional[plt.Axes]
        Axes matplotlib.

    Returns
    -------
    plt.Axes
    """
    df = metrics_df.copy()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        created_fig = True

    policies = df["policy"].tolist()
    means = df[value_col].tolist()
    stds = df[err_col].tolist() if err_col in df.columns else [0.0] * len(means)
    colors = [_get_policy_color(p) for p in policies]

    bars = ax.bar(
        [p.replace("_", "-").title() for p in policies],
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    # Hiển thị số liệu trên đỉnh mỗi cột
    for bar, m in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2.0,
            f"{m:.1f} m",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Travel Distance (meters)", fontsize=11)
    ax.set_xlabel("Storage Policy", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_convergence_curve(
    history_best_fitness: Sequence[float],
    title: str = "GRO Convergence Curve",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ đường cong hội tụ của thuật toán di truyền GRO (Fitness vs Generations).

    Parameters
    ----------
    history_best_fitness : Sequence[float]
        Lịch sử cự ly tối ưu qua từng thế hệ.
    title : str
        Tiêu đề biểu đồ.
    save_path : Optional[Union[str, Path]]
        Đường dẫn file ảnh.
    ax : Optional[plt.Axes]
        Axes matplotlib.

    Returns
    -------
    plt.Axes
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        created_fig = True

    generations = list(range(len(history_best_fitness)))
    ax.plot(generations, history_best_fitness, color="#1f77b4", linewidth=2, label="Best Tour Distance")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Generation / Iteration", fontsize=11)
    ax.set_ylabel("Best Tour Distance (meters)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_picking_route_2d(
    locations: Sequence[Any],
    route_order: Optional[Sequence[int]] = None,
    depot_coord: tuple[float, float] = (0.0, 0.0),
    title: str = "2D Picking Route Traversal",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ sơ đồ lộ trình nhặt hàng 2D đi qua các điểm lấy hàng và quay về Depot.

    Parameters
    ----------
    locations : Sequence[Any]
        Danh sách tọa độ các điểm nhặt hàng (tuples (x, y) hoặc đối tượng có thuộc tính x, y).
    route_order : Optional[Sequence[int]]
        Thứ tự ghé thăm các điểm nhặt.
    depot_coord : tuple[float, float]
        Tọa độ vị trí xuất phát và kết thúc (Depot).
    title : str
        Tiêu đề bản đồ lộ trình.
    save_path : Optional[Union[str, Path]]
        Đường dẫn file ảnh.
    ax : Optional[plt.Axes]
        Axes matplotlib.

    Returns
    -------
    plt.Axes
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        created_fig = True

    # Trích xuất tọa độ
    coords = []
    for loc in locations:
        if isinstance(loc, (tuple, list)):
            coords.append((float(loc[0]), float(loc[1])))
        elif hasattr(loc, "x") and hasattr(loc, "y"):
            coords.append((float(loc.x), float(loc.y)))
        else:
            coords.append((0.0, 0.0))

    if route_order is not None and len(route_order) > 0:
        ordered_coords = [coords[idx] for idx in route_order if idx < len(coords)]
    else:
        ordered_coords = coords

    # Chuỗi hành trình từ Depot qua các điểm và về lại Depot
    path = [depot_coord] + ordered_coords + [depot_coord]
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # Vẽ đường đi nối các điểm
    ax.plot(xs, ys, "b-", linewidth=1.5, alpha=0.7, label="Picking Route")
    # Vẽ các điểm nhặt
    pick_xs = [p[0] for p in ordered_coords]
    pick_ys = [p[1] for p in ordered_coords]
    ax.scatter(pick_xs, pick_ys, color="red", s=60, zorder=5, label="Pick Items")

    # Đánh số thứ tự các điểm ghé thăm
    for i, (x, y) in enumerate(ordered_coords, 1):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    # Điểm Depot
    ax.scatter([depot_coord[0]], [depot_coord[1]], color="green", marker="s", s=100, zorder=6, label="Depot")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("X coordinate (meters)", fontsize=11)
    ax.set_ylabel("Y coordinate (meters)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_tukey_hsd_forest(
    tukey_df: pd.DataFrame,
    title: str = "Tukey HSD Post-hoc 95% Confidence Intervals",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ biểu đồ Forest Plot thể hiện khoảng tin cậy 95% đồng thời (Simultaneous 95% CI)
    giữa các cặp so sánh chính sách từ kết quả kiểm định Tukey HSD.

    Các cặp có khoảng tin cậy không cắt qua đường 0 (zero line) là khác biệt có ý nghĩa thống kê (p < 0.05).
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(tukey_df) * 0.45)), dpi=150)
        created_fig = True

    df = tukey_df.copy()
    # Tạo nhãn so sánh cặp
    labels = [f"{r['group1']} vs {r['group2']}" for _, r in df.iterrows()]
    y_pos = np.arange(len(labels))

    # Điểm ước lượng trung bình
    mean_diffs = df["mean_diff"].values
    lowers = df["lower_ci"].values if "lower_ci" in df.columns else mean_diffs
    uppers = df["upper_ci"].values if "upper_ci" in df.columns else mean_diffs
    sigs = df["significant"].values if "significant" in df.columns else [False] * len(df)

    # Vẽ đường trung tính x=0
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)

    for i in range(len(df)):
        low = lowers[i]
        upp = uppers[i]
        diff = mean_diffs[i]
        is_sig = bool(sigs[i])

        color = "#27ae60" if is_sig else "#7f8c8d"  # Xanh lá nếu có ý nghĩa, xám nếu không
        marker = "o" if is_sig else "s"

        if not np.isnan(low) and not np.isnan(upp):
            ax.plot([low, upp], [i, i], color=color, linewidth=2.0, zorder=2)
            ax.scatter([diff], [i], color=color, s=50, marker=marker, zorder=3, edgecolors="black", linewidths=0.5)
        else:
            ax.scatter([diff], [i], color=color, s=50, marker=marker, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Đưa cặp đầu tiên lên trên
    ax.set_xlabel("Mean Distance Difference (meters)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Chú thích
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#27ae60", marker="o", lw=2, label="Significant (p < 0.05)"),
        Line2D([0], [0], color="#7f8c8d", marker="s", lw=2, label="Not Significant (p >= 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="best", frameon=True, fontsize=9)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_effect_sizes_bar(
    effect_df: pd.DataFrame,
    title: str = "Pairwise Effect Sizes (Cohen's d)",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ biểu đồ cột ngang thể hiện kích thước hiệu ứng chuẩn hóa (Cohen's d)
    giữa các cặp chính sách kèm các ngưỡng định lượng (Small=0.2, Medium=0.5, Large=0.8).
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(effect_df) * 0.45)), dpi=150)
        created_fig = True

    df = effect_df.copy()
    labels = [f"{r['policy_A']} vs {r['policy_B']}" for _, r in df.iterrows()]
    d_vals = df["cohens_d"].values
    y_pos = np.arange(len(labels))

    # Màu sắc dựa trên độ lớn |d|
    colors = []
    for d in d_vals:
        ad = abs(d)
        if ad >= 0.8:
            colors.append("#c0392b")  # Đỏ đậm - Large
        elif ad >= 0.5:
            colors.append("#e67e22")  # Cam - Medium
        elif ad >= 0.2:
            colors.append("#f1c40f")  # Vàng - Small
        else:
            colors.append("#95a5a6")  # Xám - Negligible

    bars = ax.barh(y_pos, d_vals, color=colors, edgecolor="black", linewidth=0.5, height=0.6, zorder=2)
    ax.axvline(0, color="black", linewidth=1.0)
    # Đường ngưỡng
    for th in [0.2, 0.5, 0.8, -0.2, -0.5, -0.8]:
        ax.axvline(th, color="#bdc3c7", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (Standardized Mean Difference)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.4, axis="x")

    # Hiển thị giá trị d trên từng cột
    for bar, d in zip(bars, d_vals):
        width = bar.get_width()
        x_pos = width + (0.05 if width >= 0 else -0.15)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{d:.2f}",
                va="center", ha="left" if width >= 0 else "right", fontsize=9, fontweight="bold")

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    if created_fig and save_path is None:
        plt.close(fig)

    return ax


def plot_residuals_diagnostics(
    residuals: Sequence[float],
    title: str = "ANOVA Model Residuals Normality Diagnostics",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Vẽ 2 biểu đồ chẩn đoán tính chuẩn của phần dư sai số mô hình ANOVA:
    1. Histogram & Đường cong phân phối chuẩn lý thuyết
    2. Normal Q-Q Plot (Quantile-Quantile Plot)
    """
    from scipy import stats

    arr = np.asarray(residuals, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # 1. Histogram & Normal curve
    n, bins, patches = ax1.hist(arr, bins=12, density=True, color="#3498db", edgecolor="black", alpha=0.7)
    mu, std = float(np.mean(arr)), float(np.std(arr, ddof=1))
    x_grid = np.linspace(min(arr), max(arr), 100)
    p_norm = stats.norm.pdf(x_grid, mu, std)
    ax1.plot(x_grid, p_norm, "r-", linewidth=2.0, label=f"Normal Fit (μ={mu:.1f}, σ={std:.1f})")
    ax1.set_title("Residuals Histogram & Normal Fit", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Residual (meters)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist="norm")
    ax2.plot(osm, osr, "bo", markersize=6, alpha=0.8)
    ax2.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=2.0, label=f"Fit line (R²={r**2:.3f})")
    ax2.set_title("Normal Q-Q Plot", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax2.set_ylabel("Sample Quantiles", fontsize=10)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)

    return fig
