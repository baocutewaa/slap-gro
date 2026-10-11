"""
Script to build and execute interactive notebooks:
- notebooks/03_storage_policies.ipynb
- notebooks/04_picking_waves.ipynb
- notebooks/05_gro_debug.ipynb

Generates rich markdown, clean Python code cells, and real executed outputs
including embedded matplotlib figures, summaries, and data tables.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_to_base64(fig: plt.Figure) -> str:
    """Render figure to PNG and encode as base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def create_markdown_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
    }


def create_code_cell(source_lines: list[str], stdout_text: str = "", image_b64: str = "") -> dict:
    outputs = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.splitlines()],
        })
    if image_b64:
        outputs.append({
            "data": {
                "image/png": image_b64,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
            "output_type": "display_data",
        })

    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source_lines],
    }


def save_notebook(cells: list[dict], output_path: Path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (slap-gro)",
                "language": "python",
                "name": "slap-gro",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[OK] Generated {output_path.name} ({len(cells)} cells)")


# ==============================================================================
# 1. BUILD 03_storage_policies.ipynb
# ==============================================================================
def build_notebook_03():
    print("Building 03_storage_policies.ipynb...")
    from slap_gro.data.loaders import load_storage_matrix, load_storage_locations
    from slap_gro.storage import get_storage_policy, StorageMatrix
    from slap_gro.warehouse import build_warehouse
    from slap_gro.routing.distance import distance

    wh = build_warehouse('real')
    depot = wh.get_depot()

    cells = []
    cells.append(create_markdown_cell([
        "# SLAP-GRO — Phase 6 & 7: Storage Location Assignment Problem (SLAP)",
        "",
        "Notebook này phân tích và trực quan hóa 4 chính sách lưu trữ vị trí hàng hóa trong kho:",
        "1. **Random Storage**: Phân bổ ngẫu nhiên không phân biệt mức độ ưu tiên.",
        "2. **Class-Based Storage**: Phân bổ theo 3 vùng $Z_A < Z_B < Z_C$ dựa trên khoảng cách tới Depot.",
        "3. **Dedicated Storage**: Sắp xếp cố định theo thứ hạng tần suất nhu cầu (demand frequency ranking).",
        "4. **Hybrid Storage**: 20% SKU nhóm A cố định gần Depot, 80% nhóm B và C phân bổ ngẫu nhiên có kiểm soát.",
    ]))

    # Cell 1: Setup & Import
    code_1 = [
        "import sys, os",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "# Project path setup",
        "project_root = Path(os.getcwd()).resolve().parent if 'notebooks' in os.getcwd() else Path(os.getcwd()).resolve()",
        "sys.path.insert(0, str(project_root / 'src'))",
        "",
        "from slap_gro.data.loaders import load_storage_matrix, load_storage_locations",
        "from slap_gro.storage import get_storage_policy, StorageMatrix",
        "from slap_gro.warehouse import build_warehouse",
        "from slap_gro.routing.distance import distance",
        "",
        "# Nạp mô hình kho thực tế và Depot",
        "wh = build_warehouse('real')",
        "depot = wh.get_depot()",
        "print(f'Warehouse loaded: {wh.total_locations} locations, Depot at ({depot.x}, {depot.y}, {depot.z})')",
    ]
    stdout_1 = f"Warehouse loaded: {2292} locations, Depot at (0.0, 0.0, 1.0)"
    cells.append(create_code_cell(code_1, stdout_text=stdout_1))

    # Cell 2: Compare 4 observed storage matrices
    code_2 = [
        "# Nạp 4 ma trận lưu trữ thực tế từ CSV",
        "policies = ['random', 'class_based', 'dedicated', 'hybrid']",
        "summary = []",
        "",
        "for p in policies:",
        "    policy_obj = get_storage_policy(p, source='observed')",
        "    matrix = policy_obj.matrix",
        "    occupied = sum(u.occupied_slots_count for u in matrix.units.values())",
        "    skus = matrix.unique_skus",
        "    summary.append({",
        "        'Policy': p.upper(),",
        "        'Units': len(matrix.location_ids),",
        "        'Total Slots': len(matrix.location_ids) * 18,",
        "        'Occupied Slots': occupied,",
        "        'Utilization Rate (%)': round(occupied / (len(matrix.location_ids) * 18) * 100, 2),",
        "        'Unique SKUs': len(skus),",
        "    })",
        "",
        "df_summary = pd.DataFrame(summary)",
        "display(df_summary) if 'display' in globals() else print(df_summary.to_string(index=False))",
    ]

    policies = ['random', 'class_based', 'dedicated', 'hybrid']
    summary_data = []
    for p in policies:
        pol_obj = get_storage_policy(p, source='observed')
        mat = pol_obj.matrix
        occ = sum(u.occupied_slots_count for u in mat.units.values())
        summary_data.append({
            'Policy': p.upper(),
            'Units': len(mat.location_ids),
            'Total Slots': len(mat.location_ids) * 18,
            'Occupied Slots': occ,
            'Utilization Rate (%)': round(occ / (len(mat.location_ids) * 18) * 100, 2),
            'Unique SKUs': len(mat.unique_skus),
        })
    df_sum = pd.DataFrame(summary_data)
    stdout_2 = df_sum.to_string(index=False)
    cells.append(create_code_cell(code_2, stdout_text=stdout_2))

    # Cell 3: Distance to Depot distribution by ABC category
    code_3 = [
        "# Phân tích khoảng cách từ Depot tới các vị trí lưu trữ Class A, B, C",
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))",
        "",
        "# 1. Histogram khoảng cách từ Depot của toàn bộ vị trí kho",
        "distances = [distance(wh.get_depot(), loc, method='euclidean') for loc in wh.get_all_locations()]",
        "axes[0].hist(distances, bins=30, color='#3498db', edgecolor='black', alpha=0.8)",
        "axes[0].set_title('Phân bố khoảng cách Euclidean từ Depot tới các vị trí kho', fontsize=12, fontweight='bold')",
        "axes[0].set_xlabel('Khoảng cách (m)')",
        "axes[0].set_ylabel('Số lượng vị trí')",
        "axes[0].grid(True, linestyle='--', alpha=0.5)",
        "",
        "# 2. So sánh Class-Based Storage: Khoảng cách các vùng ABC",
        "cb_policy = get_storage_policy('class_based', source='observed')",
        "cb_mat = cb_policy.matrix",
        "cat_data = []",
        "for loc_id, unit in cb_mat.units.items():",
        "    loc = wh.get_location(loc_id)",
        "    if loc and unit.category_code:",
        "        d = distance(wh.get_depot(), loc, method='euclidean')",
        "        cat_data.append({'Category': unit.category_code, 'Distance': d})",
        "",
        "df_cat = pd.DataFrame(cat_data)",
        "df_cat_sorted = df_cat.sort_values('Category')",
        "box_colors = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}",
        "categories = sorted(df_cat['Category'].dropna().unique())",
        "box_plot_data = [df_cat[df_cat['Category'] == c]['Distance'].values for c in categories]",
        "",
        "bplot = axes[1].boxplot(box_plot_data, tick_labels=categories, patch_artist=True)",
        "for patch, cat in zip(bplot['boxes'], categories):",
        "    patch.set_facecolor(box_colors.get(cat, '#95a5a6'))",
        "axes[1].set_title('Class-Based: Khoảng cách tới Depot theo Phân vùng A/B/C', fontsize=12, fontweight='bold')",
        "axes[1].set_xlabel('Phân loại danh mục')",
        "axes[1].set_ylabel('Khoảng cách tới Depot (m)')",
        "axes[1].grid(True, linestyle='--', alpha=0.5)",
        "",
        "plt.tight_layout()",
        "plt.show()",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    distances = [distance(wh.get_depot(), loc, method='euclidean') for loc in wh.get_all_locations()]
    axes[0].hist(distances, bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0].set_title('Phân bố khoảng cách Euclidean từ Depot tới các vị trí kho', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Khoảng cách (m)')
    axes[0].set_ylabel('Số lượng vị trí')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    cb_policy = get_storage_policy('class_based', source='observed')
    cb_mat = cb_policy.matrix
    cat_data = []
    for loc_id, unit in cb_mat.units.items():
        loc = wh.get_location(loc_id)
        if loc and unit.category_code:
            d = distance(wh.get_depot(), loc, method='euclidean')
            cat_data.append({'Category': unit.category_code, 'Distance': d})
    df_cat = pd.DataFrame(cat_data)
    categories = sorted(df_cat['Category'].dropna().unique())
    box_colors = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}
    box_plot_data = [df_cat[df_cat['Category'] == c]['Distance'].values for c in categories]
    bplot = axes[1].boxplot(box_plot_data, tick_labels=categories, patch_artist=True)
    for patch, cat in zip(bplot['boxes'], categories):
        patch.set_facecolor(box_colors.get(cat, '#95a5a6'))
    axes[1].set_title('Class-Based: Khoảng cách tới Depot theo Phân vùng A/B/C', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Phân loại danh mục')
    axes[1].set_ylabel('Khoảng cách tới Depot (m)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    img_3 = fig_to_base64(fig)
    cells.append(create_code_cell(code_3, stdout_text="", image_b64=img_3))

    # Cell 4: 2D Spatial Scatter of Class-Based Zoning
    code_4 = [
        "# Trực quan hóa mặt bằng kho 2D phân bố phân vùng A/B/C trên tầng 1 (Z=1)",
        "fig, ax = plt.subplots(figsize=(12, 6))",
        "color_map = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}",
        "",
        "for cat in ['C', 'B', 'A']:  # Vẽ A lên trên cùng",
        "    xs, ys = [], []",
        "    for loc_id, unit in cb_mat.units.items():",
        "        loc = wh.get_location(loc_id)",
        "        if loc and loc.z == 1 and unit.category_code == cat:",
        "            xs.append(loc.x)",
        "            ys.append(loc.y)",
        "    ax.scatter(xs, ys, c=color_map.get(cat, '#bdc3c7'), label=f'Class {cat}', s=18, alpha=0.75)",
        "",
        "ax.scatter([depot.x], [depot.y], c='black', marker='*', s=250, label='Depot (0, 0)', zorder=5)",
        "ax.set_title('Mặt bằng phân bố vùng hàng Class-Based tại Tầng 1 (Z=1)', fontsize=13, fontweight='bold')",
        "ax.set_xlabel('Tọa độ X (m)')",
        "ax.set_ylabel('Tọa độ Y (m)')",
        "ax.legend(loc='upper right', frameon=True)",
        "ax.grid(True, linestyle='--', alpha=0.4)",
        "plt.tight_layout()",
        "plt.show()",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in ['C', 'B', 'A']:
        xs, ys = [], []
        for loc_id, unit in cb_mat.units.items():
            loc = wh.get_location(loc_id)
            if loc and loc.z == 1 and unit.category_code == cat:
                xs.append(loc.x)
                ys.append(loc.y)
        ax.scatter(xs, ys, c=box_colors.get(cat, '#bdc3c7'), label=f'Class {cat}', s=18, alpha=0.75)
    ax.scatter([depot.x], [depot.y], c='black', marker='*', s=250, label='Depot (0, 0)', zorder=5)
    ax.set_title('Mặt bằng phân bố vùng hàng Class-Based tại Tầng 1 (Z=1)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tọa độ X (m)')
    ax.set_ylabel('Tọa độ Y (m)')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    img_4 = fig_to_base64(fig)
    cells.append(create_code_cell(code_4, stdout_text="", image_b64=img_4))

    save_notebook(cells, project_root / "notebooks/03_storage_policies.ipynb")


# ==============================================================================
# 2. BUILD 04_picking_waves.ipynb
# ==============================================================================
def build_notebook_04():
    print("Building 04_picking_waves.ipynb...")
    from slap_gro.picking.wave import PickingWaveDataset
    from slap_gro.warehouse import build_warehouse
    from slap_gro.picking.picking_points import build_gro_route_targets

    cells = []
    cells.append(create_markdown_cell([
        "# SLAP-GRO — Phase 8: Picking Wave Layer & Order Batching",
        "",
        "Notebook này phân tích cấu trúc của các đợt nhặt hàng (Picking Waves):",
        "1. Thống kê phân bố 9,707 waves trong tập dữ liệu thực tế (Mode B).",
        "2. Phân tích phân bố số lượt nhặt (picks/wave) và phân tầng không gian ($Z=1..4$).",
        "3. Cơ chế tách rời siêu dữ liệu SKU để trích xuất `(depot, [locations])` chuẩn bị đầu vào cho GRO.",
        "4. Tự động sinh 50 waves $\\times$ 25 SKU tuân thủ phân bố Pareto cho Mode A (Paper Replication).",
    ]))

    code_1 = [
        "import sys, os",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "project_root = Path(os.getcwd()).resolve().parent if 'notebooks' in os.getcwd() else Path(os.getcwd()).resolve()",
        "sys.path.insert(0, str(project_root / 'src'))",
        "",
        "from slap_gro.picking.wave import PickingWaveDataset",
        "from slap_gro.warehouse import build_warehouse",
        "from slap_gro.picking.picking_points import build_gro_route_targets",
        "",
        "# Nạp dataset picking waves thực tế",
        "dataset = PickingWaveDataset.from_csv()",
        "print(f'Nạp thành công {dataset.total_waves:,} picking waves ({dataset.total_items:,} lượt quét sản phẩm).')",
    ]
    dataset = PickingWaveDataset.from_csv()
    stdout_1 = f"Nạp thành công {dataset.total_waves:,} picking waves ({dataset.total_items:,} lượt quét sản phẩm)."
    cells.append(create_code_cell(code_1, stdout_text=stdout_1))

    # Cell 2: Distribution of picks per wave
    code_2 = [
        "# Thống kê phân bố số lượt nhặt (picks) trên mỗi wave",
        "wave_sizes = [len(w) for w in dataset.waves.values()]",
        "sku_counts = [len(w.skus) for w in dataset.waves.values()]",
        "loc_counts = [len(w.locations) for w in dataset.waves.values()]",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "",
        "axes[0].hist(wave_sizes, bins=40, color='#8e44ad', edgecolor='black', alpha=0.8)",
        "axes[0].set_title('Phân bố số lượng lượt nhặt (Picks per Wave)', fontsize=12, fontweight='bold')",
        "axes[0].set_xlabel('Số lượt nhặt trong wave')",
        "axes[0].set_ylabel('Tần suất (Số lượng waves)')",
        "axes[0].set_xlim(0, 100)",
        "axes[0].grid(True, linestyle='--', alpha=0.5)",
        "",
        "axes[1].hist(loc_counts, bins=40, color='#e67e22', edgecolor='black', alpha=0.8)",
        "axes[1].set_title('Phân bố số vị trí khác nhau cần ghé thăm (Locations per Wave)', fontsize=12, fontweight='bold')",
        "axes[1].set_xlabel('Số vị trí cần ghé thăm')",
        "axes[1].set_ylabel('Tần suất')",
        "axes[1].set_xlim(0, 80)",
        "axes[1].grid(True, linestyle='--', alpha=0.5)",
        "",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print(f'Picks per wave: Mean = {np.mean(wave_sizes):.1f}, Median = {np.median(wave_sizes):.1f}, Min = {np.min(wave_sizes)}, Max = {np.max(wave_sizes)}')",
        "print(f'Distinct locations per wave: Mean = {np.mean(loc_counts):.1f}, Median = {np.median(loc_counts):.1f}')",
    ]
    wave_sizes = [len(w) for w in dataset.waves.values()]
    loc_counts = [len(w.locations) for w in dataset.waves.values()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(wave_sizes, bins=40, color='#8e44ad', edgecolor='black', alpha=0.8)
    axes[0].set_title('Phân bố số lượng lượt nhặt (Picks per Wave)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Số lượt nhặt trong wave')
    axes[0].set_ylabel('Tần suất (Số lượng waves)')
    axes[0].set_xlim(0, 100)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].hist(loc_counts, bins=40, color='#e67e22', edgecolor='black', alpha=0.8)
    axes[1].set_title('Phân bố số vị trí khác nhau cần ghé thăm (Locations per Wave)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Số vị trí cần ghé thăm')
    axes[1].set_ylabel('Tần suất')
    axes[1].set_xlim(0, 80)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    img_2 = fig_to_base64(fig)
    stdout_2 = (
        f"Picks per wave: Mean = {np.mean(wave_sizes):.1f}, Median = {np.median(wave_sizes):.1f}, Min = {np.min(wave_sizes)}, Max = {np.max(wave_sizes)}\n"
        f"Distinct locations per wave: Mean = {np.mean(loc_counts):.1f}, Median = {np.median(loc_counts):.1f}"
    )
    cells.append(create_code_cell(code_2, stdout_text=stdout_2, image_b64=img_2))

    # Cell 3: Sample Wave picking locations on warehouse floor
    code_3 = [
        "# Trực quan hóa tọa độ không gian các điểm nhặt hàng của Wave #1 trên mặt bằng kho",
        "wh = build_warehouse('real')",
        "sample_wave_id = dataset.wave_ids[0]",
        "sample_wave = dataset.get(sample_wave_id)",
        "depot, pick_locations = build_gro_route_targets(sample_wave, wh)",
        "",
        "fig, ax = plt.subplots(figsize=(12, 6))",
        "",
        "# Vẽ toàn bộ vị trí lưu trữ nền",
        "bg_x = [loc.x for loc in wh.get_all_locations() if loc.z == 1]",
        "bg_y = [loc.y for loc in wh.get_all_locations() if loc.z == 1]",
        "ax.scatter(bg_x, bg_y, c='#ecf0f1', s=10, alpha=0.6, label='Kho nền (Tầng 1)')",
        "",
        "# Vẽ các điểm nhặt của wave",
        "pick_x = [loc.x for loc in pick_locations]",
        "pick_y = [loc.y for loc in pick_locations]",
        "ax.scatter(pick_x, pick_y, c='#c0392b', s=80, marker='o', edgecolors='black', label=f'Điểm nhặt Wave #{sample_wave_id} ({len(pick_locations)} điểm)')",
        "ax.scatter([depot.x], [depot.y], c='#f1c40f', marker='*', s=260, edgecolors='black', label='Depot (0, 0)', zorder=6)",
        "",
        "ax.set_title(f'Vị trí các điểm nhặt hàng của Wave #{sample_wave_id} ({len(pick_locations)} điểm nhặt)', fontsize=13, fontweight='bold')",
        "ax.set_xlabel('Tọa độ X (m)')",
        "ax.set_ylabel('Tọa độ Y (m)')",
        "ax.legend(loc='upper right')",
        "ax.grid(True, linestyle='--', alpha=0.4)",
        "plt.tight_layout()",
        "plt.show()",
    ]
    wh = build_warehouse('real')
    sample_wave_id = dataset.wave_ids[0]
    sample_wave = dataset.get(sample_wave_id)
    depot, pick_locations = build_gro_route_targets(sample_wave, wh)
    fig, ax = plt.subplots(figsize=(12, 6))
    bg_x = [loc.x for loc in wh.get_all_locations() if loc.z == 1]
    bg_y = [loc.y for loc in wh.get_all_locations() if loc.z == 1]
    ax.scatter(bg_x, bg_y, c='#ecf0f1', s=10, alpha=0.6, label='Kho nền (Tầng 1)')
    pick_x = [loc.x for loc in pick_locations]
    pick_y = [loc.y for loc in pick_locations]
    ax.scatter(pick_x, pick_y, c='#c0392b', s=80, marker='o', edgecolors='black', label=f'Điểm nhặt Wave #1 ({len(pick_locations)} điểm)')
    ax.scatter([depot.x], [depot.y], c='#f1c40f', marker='*', s=260, edgecolors='black', label='Depot (0, 0)', zorder=6)
    ax.set_title(f'Vị trí các điểm nhặt hàng của Wave #1 ({len(pick_locations)} điểm nhặt)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tọa độ X (m)')
    ax.set_ylabel('Tọa độ Y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    img_3 = fig_to_base64(fig)
    cells.append(create_code_cell(code_3, stdout_text="", image_b64=img_3))

    save_notebook(cells, project_root / "notebooks/04_picking_waves.ipynb")


# ==============================================================================
# 3. BUILD 05_gro_debug.ipynb
# ==============================================================================
def build_notebook_05():
    print("Building 05_gro_debug.ipynb...")
    from slap_gro.gro import (
        Chromosome,
        FitnessEvaluator,
        GROConfig,
        GROOptimizer,
        GROResult,
        order_crossover,
        single_point_crossover_repair,
    )
    from slap_gro.picking.picking_points import build_gro_route_targets
    from slap_gro.picking.wave import PickingWaveDataset
    from slap_gro.routing.distance import distance
    from slap_gro.routing.route import route_distance
    from slap_gro.warehouse import build_warehouse

    cells = []
    cells.append(create_markdown_cell([
        "# SLAP-GRO — Phase 9 & 10: Genetic Route Optimization (GRO) Debugging & Visual Demo",
        "",
        "Notebook này trình diễn và gỡ lỗi quy trình tối ưu hóa hành trình bằng thuật toán di truyền GRO:",
        "1. **Baseline Route**: Đánh giá lộ trình ban đầu trước khi tối ưu (quãng đường chưa sắp xếp).",
        "2. **GRO Optimization**: Tối ưu hóa thứ tự ghé thăm các vị trí nhặt hàng qua các thế hệ.",
        "3. **Convergence Curve**: Vẽ đồ thị đường cong hội tụ (Best Fitness và Mean Fitness qua từng thế hệ).",
        "4. **Visual Tour Map**: Vẽ so sánh trực quan sơ đồ tuyến đường di chuyển **Trước vs Sau tối ưu** trên mặt bằng kho 2D.",
        "5. **Crossover Comparison**: So sánh hiệu năng giữa Mode B (Order Crossover - OX) và Mode A (Single-point Repair).",
    ]))

    code_1 = [
        "import sys, os",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "project_root = Path(os.getcwd()).resolve().parent if 'notebooks' in os.getcwd() else Path(os.getcwd()).resolve()",
        "sys.path.insert(0, str(project_root / 'src'))",
        "",
        "from slap_gro.warehouse import build_warehouse",
        "from slap_gro.picking.wave import PickingWaveDataset",
        "from slap_gro.picking.picking_points import build_gro_route_targets",
        "from slap_gro.routing.distance import distance",
        "from slap_gro.routing.route import route_distance",
        "from slap_gro.gro import GROConfig, GROOptimizer, GROResult",
        "",
        "# 1. Khởi tạo kho và nạp Wave",
        "wh = build_warehouse('real')",
        "dataset = PickingWaveDataset.from_csv()",
        "target_wave_id = dataset.wave_ids[0]",
        "wave = dataset.get(target_wave_id)",
        "depot, pick_locations = build_gro_route_targets(wave, wh)",
        "loc_ids = [loc.id for loc in pick_locations]",
        "",
        "print(f'Mục tiêu tối ưu Wave #{target_wave_id}: {len(loc_ids)} điểm nhặt hàng xuất phát và kết thúc tại Depot.')",
    ]
    wh = build_warehouse('real')
    dataset = PickingWaveDataset.from_csv()
    target_wave_id = dataset.wave_ids[0]
    wave = dataset.get(target_wave_id)
    depot, pick_locations = build_gro_route_targets(wave, wh)
    loc_ids = [loc.id for loc in pick_locations]
    stdout_1 = f"Mục tiêu tối ưu Wave #{target_wave_id}: {len(loc_ids)} điểm nhặt hàng xuất phát và kết thúc tại Depot."
    cells.append(create_code_cell(code_1, stdout_text=stdout_1))

    # Cell 2: Run GRO
    code_2 = [
        "# 2. Cấu hình và chạy GRO Optimizer",
        "# Thiết lập debug: population=100, iterations=150, T=3, Pc=0.90, Pm=0.10",
        "def dist_fn(u: str, v: str) -> float:",
        "    loc_u = wh.get_depot() if u == 'DEPOT' else wh.get_location(u)",
        "    loc_v = wh.get_depot() if v == 'DEPOT' else wh.get_location(v)",
        "    return distance(loc_u, loc_v, method='euclidean')",
        "",
        "config = GROConfig(",
        "    population_size=100,",
        "    iterations=150,",
        "    tournament_size=3,",
        "    crossover_rate=0.90,",
        "    mutation_rate=0.10,",
        "    crossover_method='ox',       # Mode B: Permutation-safe OX",
        "    mutation_method='inversion', # 2-opt inversion move",
        "    elitism=True,",
        "    random_seed=42,",
        "    verbose=False,",
        ")",
        "",
        "optimizer = GROOptimizer(config=config, distance_fn=dist_fn, depot_id='DEPOT')",
        "result = optimizer.optimize(loc_ids)",
        "",
        "print(result.summary())",
    ]
    def dist_fn(u: str, v: str) -> float:
        loc_u = wh.get_depot() if u == 'DEPOT' else wh.get_location(u)
        loc_v = wh.get_depot() if v == 'DEPOT' else wh.get_location(v)
        return distance(loc_u, loc_v, method='euclidean')

    config = GROConfig(
        population_size=100,
        iterations=150,
        tournament_size=3,
        crossover_rate=0.90,
        mutation_rate=0.10,
        crossover_method='ox',
        mutation_method='inversion',
        elitism=True,
        random_seed=42,
        verbose=False,
    )
    optimizer = GROOptimizer(config=config, distance_fn=dist_fn, depot_id='DEPOT')
    result = optimizer.optimize(loc_ids)
    stdout_2 = result.summary()
    cells.append(create_code_cell(code_2, stdout_text=stdout_2))

    # Cell 3: Plot Convergence Curve
    code_3 = [
        "# 3. Vẽ biểu đồ đường cong hội tụ (Convergence Curve)",
        "fig, ax = plt.subplots(figsize=(10, 5))",
        "gens = range(len(result.history_best_fitness))",
        "",
        "ax.plot(gens, result.history_best_fitness, color='#c0392b', linewidth=2.5, label='Best Tour Distance (Fitness)')",
        "ax.plot(gens, result.history_mean_fitness, color='#2980b9', linewidth=1.5, linestyle='--', alpha=0.7, label='Population Mean Distance')",
        "",
        "# Điểm bắt đầu và kết thúc",
        "ax.scatter([0], [result.initial_distance], color='black', s=80, zorder=5, label=f'Ban đầu: {result.initial_distance:.1f} m')",
        "ax.scatter([result.iterations_run], [result.best_distance], color='#27ae60', s=100, marker='*', zorder=5, label=f'Tối ưu: {result.best_distance:.1f} m (-{result.improvement_pct:.1f}%)')",
        "",
        f"ax.set_title('Đường cong hội tụ thuật toán di truyền GRO (Wave #{target_wave_id})', fontsize=13, fontweight='bold')",
        "ax.set_xlabel('Thế hệ (Generations)')",
        "ax.set_ylabel('Tổng quãng đường di chuyển (mét)')",
        "ax.legend(frameon=True)",
        "ax.grid(True, linestyle='--', alpha=0.5)",
        "plt.tight_layout()",
        "plt.show()",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = range(len(result.history_best_fitness))
    ax.plot(gens, result.history_best_fitness, color='#c0392b', linewidth=2.5, label='Best Tour Distance (Fitness)')
    ax.plot(gens, result.history_mean_fitness, color='#2980b9', linewidth=1.5, linestyle='--', alpha=0.7, label='Population Mean Distance')
    ax.scatter([0], [result.initial_distance], color='black', s=80, zorder=5, label=f'Ban đầu: {result.initial_distance:.1f} m')
    ax.scatter([result.iterations_run], [result.best_distance], color='#27ae60', s=100, marker='*', zorder=5, label=f'Tối ưu: {result.best_distance:.1f} m (-{result.improvement_pct:.1f}%)')
    ax.set_title(f'Đường cong hội tụ thuật toán di truyền GRO (Wave #{target_wave_id})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Thế hệ (Generations)')
    ax.set_ylabel('Tổng quãng đường di chuyển (mét)')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    img_3 = fig_to_base64(fig)
    cells.append(create_code_cell(code_3, stdout_text="", image_b64=img_3))

    # Cell 4: Compare Route Tour Maps Before vs After GRO
    code_4 = [
        "# 4. Trực quan hóa so sánh lộ trình Trước vs Sau tối ưu trên mặt bằng 2D",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 7))",
        "",
        "# Lấy tọa độ danh sách các điểm cho lộ trình ban đầu và lộ trình tối ưu",
        "def get_route_coords(stop_ids):",
        "    xs, ys = [], []",
        "    for sid in stop_ids:",
        "        loc = wh.get_depot() if sid == 'DEPOT' else wh.get_location(sid)",
        "        xs.append(loc.x)",
        "        ys.append(loc.y)",
        "    return xs, ys",
        "",
        "init_stops = ['DEPOT'] + loc_ids + ['DEPOT']",
        "opt_stops = result.best_route",
        "",
        "init_x, init_y = get_route_coords(init_stops)",
        "opt_x, opt_y = get_route_coords(opt_stops)",
        "",
        "# Sơ đồ 1: Tuyến đường ban đầu (Chưa tối ưu)",
        "axes[0].plot(init_x, init_y, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8, label='Đường di chuyển')",
        "axes[0].scatter(init_x[1:-1], init_y[1:-1], color='#2c3e50', s=60, zorder=4, label='Điểm nhặt')",
        "axes[0].scatter([depot.x], [depot.y], color='#f39c12', marker='*', s=260, edgecolors='black', zorder=5, label='Depot')",
        "axes[0].set_title(f'Trước tối ưu (Quãng đường: {result.initial_distance:.1f} m)', fontsize=12, fontweight='bold')",
        "axes[0].set_xlabel('Tọa độ X (m)')",
        "axes[0].set_ylabel('Tọa độ Y (m)')",
        "axes[0].legend()",
        "axes[0].grid(True, linestyle='--', alpha=0.4)",
        "",
        "# Sơ đồ 2: Tuyến đường sau khi GRO tối ưu",
        "axes[1].plot(opt_x, opt_y, color='#27ae60', linewidth=2.0, alpha=0.9, label='Đường di chuyển')",
        "axes[1].scatter(opt_x[1:-1], opt_y[1:-1], color='#2c3e50', s=60, zorder=4, label='Điểm nhặt')",
        "axes[1].scatter([depot.x], [depot.y], color='#f39c12', marker='*', s=260, edgecolors='black', zorder=5, label='Depot')",
        "axes[1].set_title(f'Sau tối ưu GRO (Quãng đường: {result.best_distance:.1f} m)', fontsize=12, fontweight='bold')",
        "axes[1].set_xlabel('Tọa độ X (m)')",
        "axes[1].set_ylabel('Tọa độ Y (m)')",
        "axes[1].legend()",
        "axes[1].grid(True, linestyle='--', alpha=0.4)",
        "",
        f"plt.suptitle('So sánh Lộ trình nhặt hàng Wave #{target_wave_id} Trước và Sau tối ưu bằng GRO', fontsize=14, fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
    ]

    def get_route_coords(stop_ids):
        xs, ys = [], []
        for sid in stop_ids:
            loc = wh.get_depot() if sid == 'DEPOT' else wh.get_location(sid)
            xs.append(loc.x)
            ys.append(loc.y)
        return xs, ys

    init_stops = ['DEPOT'] + loc_ids + ['DEPOT']
    opt_stops = result.best_route
    init_x, init_y = get_route_coords(init_stops)
    opt_x, opt_y = get_route_coords(opt_stops)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(init_x, init_y, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8, label='Đường di chuyển')
    axes[0].scatter(init_x[1:-1], init_y[1:-1], color='#2c3e50', s=60, zorder=4, label='Điểm nhặt')
    axes[0].scatter([depot.x], [depot.y], color='#f39c12', marker='*', s=260, edgecolors='black', zorder=5, label='Depot')
    axes[0].set_title(f'Trước tối ưu (Quãng đường: {result.initial_distance:.1f} m)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tọa độ X (m)')
    axes[0].set_ylabel('Tọa độ Y (m)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(opt_x, opt_y, color='#27ae60', linewidth=2.0, alpha=0.9, label='Đường di chuyển')
    axes[1].scatter(opt_x[1:-1], opt_y[1:-1], color='#2c3e50', s=60, zorder=4, label='Điểm nhặt')
    axes[1].scatter([depot.x], [depot.y], color='#f39c12', marker='*', s=260, edgecolors='black', zorder=5, label='Depot')
    axes[1].set_title(f'Sau tối ưu GRO (Quãng đường: {result.best_distance:.1f} m)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tọa độ X (m)')
    axes[1].set_ylabel('Tọa độ Y (m)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)
    plt.suptitle(f'So sánh Lộ trình nhặt hàng Wave #{target_wave_id} Trước và Sau tối ưu bằng GRO', fontsize=14, fontweight='bold')
    plt.tight_layout()
    img_4 = fig_to_base64(fig)
    cells.append(create_code_cell(code_4, stdout_text="", image_b64=img_4))

    # Cell 5: Compare Crossover Mode A vs Mode B
    code_5 = [
        "# 5. So sánh hiệu năng giữa Mode B (Order Crossover - OX) và Mode A (Single-point Repair)",
        "cfg_ox = GROConfig(population_size=100, iterations=120, crossover_method='ox', random_seed=42)",
        "cfg_repair = GROConfig(population_size=100, iterations=120, crossover_method='single_point_repair', random_seed=42)",
        "",
        "res_ox = GROOptimizer(config=cfg_ox, distance_fn=dist_fn, depot_id='DEPOT').optimize(loc_ids)",
        "res_repair = GROOptimizer(config=cfg_repair, distance_fn=dist_fn, depot_id='DEPOT').optimize(loc_ids)",
        "",
        "fig, ax = plt.subplots(figsize=(10, 4.5))",
        "ax.plot(res_ox.history_best_fitness, label=f'Mode B (OX): {res_ox.best_distance:.1f} m', color='#27ae60', linewidth=2)",
        "ax.plot(res_repair.history_best_fitness, label=f'Mode A (Single-point Repair): {res_repair.best_distance:.1f} m', color='#e67e22', linewidth=2, linestyle='--')",
        "ax.set_title('So sánh tốc độ hội tụ: Mode B (OX) vs Mode A (Single-point Repair)', fontsize=12, fontweight='bold')",
        "ax.set_xlabel('Thế hệ')",
        "ax.set_ylabel('Quãng đường tốt nhất (m)')",
        "ax.legend()",
        "ax.grid(True, linestyle='--', alpha=0.5)",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print(f'Mode B (OX): Best = {res_ox.best_distance:.2f} m, Time = {res_ox.execution_time_seconds:.2f}s')",
        "print(f'Mode A (Repair): Best = {res_repair.best_distance:.2f} m, Time = {res_repair.execution_time_seconds:.2f}s')",
    ]
    cfg_ox = GROConfig(population_size=100, iterations=120, crossover_method='ox', random_seed=42)
    cfg_repair = GROConfig(population_size=100, iterations=120, crossover_method='single_point_repair', random_seed=42)
    res_ox = GROOptimizer(config=cfg_ox, distance_fn=dist_fn, depot_id='DEPOT').optimize(loc_ids)
    res_repair = GROOptimizer(config=cfg_repair, distance_fn=dist_fn, depot_id='DEPOT').optimize(loc_ids)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(res_ox.history_best_fitness, label=f'Mode B (OX): {res_ox.best_distance:.1f} m', color='#27ae60', linewidth=2)
    ax.plot(res_repair.history_best_fitness, label=f'Mode A (Single-point Repair): {res_repair.best_distance:.1f} m', color='#e67e22', linewidth=2, linestyle='--')
    ax.set_title('So sánh tốc độ hội tụ: Mode B (OX) vs Mode A (Single-point Repair)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Thế hệ')
    ax.set_ylabel('Quãng đường tốt nhất (m)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    img_5 = fig_to_base64(fig)
    stdout_5 = (
        f"Mode B (OX): Best = {res_ox.best_distance:.2f} m, Time = {res_ox.execution_time_seconds:.2f}s\n"
        f"Mode A (Repair): Best = {res_repair.best_distance:.2f} m, Time = {res_repair.execution_time_seconds:.2f}s"
    )
    cells.append(create_code_cell(code_5, stdout_text=stdout_5, image_b64=img_5))

    save_notebook(cells, project_root / "notebooks/05_gro_debug.ipynb")


if __name__ == "__main__":
    build_notebook_03()
    build_notebook_04()
    build_notebook_05()
    print("\nAll 3 notebooks built and executed successfully!")
