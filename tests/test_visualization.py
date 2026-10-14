"""
Unit tests for Phase 15 - Warehouse & Scientific Visualizations.
Tests all 8 core publication figures and helper plotting routines:
- warehouse_3d.png
- random_storage.png
- class_based_storage.png
- dedicated_storage.png
- hybrid_storage.png
- route_wave_x.png
- distance_boxplot.png
- distance_scatter.png
- generate_all_phase15_figures
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from slap_gro.evaluation.visualization import (
    generate_all_phase15_figures,
    plot_convergence_curve,
    plot_distance_boxplot,
    plot_distance_scatter,
    plot_policy_comparison_bar,
    plot_route_wave,
    plot_storage_policy_layout,
    plot_warehouse_3d,
)
from slap_gro.picking.wave import PickingItem, PickingWave
from slap_gro.warehouse import build_warehouse


@pytest.fixture
def mock_warehouse():
    """Fixture providing a lightweight Warehouse for quick tests."""
    return build_warehouse("paper")


@pytest.fixture
def mock_picking_wave(mock_warehouse) -> PickingWave:
    """Fixture providing a small PickingWave for route plotting tests."""
    loc_keys = [loc.id for loc in mock_warehouse.get_all_locations()][:10]
    items = [
        PickingItem(reference=f"SKU_{i:04d}", quantity=1, location=loc)
        for i, loc in enumerate(loc_keys, 1)
    ]
    return PickingWave(wave_id=101, items=items)


@pytest.fixture
def mock_experiment_df() -> pd.DataFrame:
    """Fixture providing mock experiment benchmark results."""
    records = []
    policies = ["original", "random", "class_based", "dedicated", "hybrid"]
    base_dists = [4100.0, 1100.0, 950.0, 880.0, 820.0]
    for pol, mean_d in zip(policies, base_dists):
        for w_id in range(1, 6):
            records.append(
                {
                    "wave_id": w_id,
                    "policy_name": pol,
                    "initial_distance": mean_d * 1.3 + w_id * 10,
                    "optimized_distance": mean_d + w_id * 8,
                }
            )
    return pd.DataFrame(records)


def test_plot_warehouse_3d(mock_warehouse: SyntheticWarehouse, tmp_path: Path):
    """Test 3D warehouse layout plotting with levels 1..4."""
    out_file = tmp_path / "test_warehouse_3d.png"
    ax = plot_warehouse_3d(mock_warehouse, title="Test 3D Warehouse", save_path=out_file, dpi=80)
    assert ax is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
    plt.close("all")


def test_plot_storage_policy_layout_all_types(mock_warehouse: SyntheticWarehouse, tmp_path: Path):
    """Test 2D storage policy layout maps for random, class_based, dedicated, hybrid."""
    for pol_name in ["random", "class_based", "dedicated", "hybrid"]:
        out_file = tmp_path / f"test_{pol_name}.png"
        ax = plot_storage_policy_layout(
            pol_name,
            mock_warehouse,
            policy_name=pol_name,
            save_path=out_file,
            dpi=80,
        )
        assert ax is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 1000
        plt.close("all")


def test_plot_route_wave(mock_warehouse: SyntheticWarehouse, mock_picking_wave: PickingWave, tmp_path: Path):
    """Test picking route traversal visualization with stops and arrows."""
    out_file = tmp_path / "test_route.png"
    ax = plot_route_wave(mock_picking_wave, mock_warehouse, save_path=out_file, dpi=80)
    assert ax is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
    plt.close("all")


def test_plot_distance_boxplot_and_scatter(mock_experiment_df: pd.DataFrame, tmp_path: Path):
    """Test Boxplot and Scatter plots comparing travel distances."""
    box_file = tmp_path / "test_boxplot.png"
    ax_box = plot_distance_boxplot(mock_experiment_df, save_path=box_file)
    assert ax_box is not None
    assert box_file.exists()
    assert box_file.stat().st_size > 1000
    plt.close("all")

    scat_file = tmp_path / "test_scatter.png"
    ax_scat = plot_distance_scatter(mock_experiment_df, save_path=scat_file)
    assert ax_scat is not None
    assert scat_file.exists()
    assert scat_file.stat().st_size > 1000
    plt.close("all")


def test_generate_all_phase15_figures(
    mock_warehouse: SyntheticWarehouse,
    mock_picking_wave: PickingWave,
    mock_experiment_df: pd.DataFrame,
    tmp_path: Path,
):
    """
    Test generate_all_phase15_figures master generator.
    Verifies that all 8 core figures required by Plan.md Phase 15 exist.
    """
    out_dir = tmp_path / "figures"
    generated = generate_all_phase15_figures(
        warehouse=mock_warehouse,
        sample_wave=mock_picking_wave,
        experiment_df=mock_experiment_df,
        output_dir=out_dir,
        dpi=80,
    )

    core_figures = [
        "warehouse_3d",
        "random_storage",
        "class_based_storage",
        "dedicated_storage",
        "hybrid_storage",
        "route_wave_x",
        "distance_boxplot",
        "distance_scatter",
    ]

    for fig_key in core_figures:
        assert fig_key in generated, f"Missing figure key: {fig_key}"
        fig_path = generated[fig_key]
        assert fig_path.exists(), f"Figure file does not exist: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure file too small: {fig_path}"

    plt.close("all")
