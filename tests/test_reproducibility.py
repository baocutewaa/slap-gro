"""
Unit and integration tests for Phase 17 - Reproducibility & Output Artifacts.
=============================================================================

Validates:
1. Determinism:
   - Running pipeline components with identical seed (seed=42) produces 100% identical outputs:
     * ProductCatalog.generate_synthetic
     * SyntheticPickingWave generation
     * Storage assignment
     * GRO genetic optimizer tours and distances
2. Output Directory Hierarchy (Plan.md Section 22):
   outputs/
   ├── tables/
   │   ├── summary_statistics.csv
   │   ├── anova.csv
   │   └── tukey.csv
   ├── figures/
   │   ├── warehouse.png
   │   ├── boxplot.png
   │   └── scatter.png
   ├── routes/
   └── logs/
3. Structured Experiment Log Schema:
   - Each log file has:
     {
       "seed": 42,
       "population_size": 1000,
       "generations": 1000,
       "tournament_size": 3,
       "crossover_probability": 0.9,
       "mutation_probability": 0.08,
       "storage_policy": "hybrid"
     }
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from slap_gro.config.loader import load_paper_config, load_real_config
from slap_gro.data.product import ProductCatalog
from slap_gro.gro.optimizer import GROConfig, GROOptimizer
from slap_gro.picking.wave import PickingWaveDataset
from slap_gro.pipeline.paper_pipeline import build_paper_storage_policies
from slap_gro.storage import get_storage_policy
from slap_gro.warehouse.warehouse_layout import build_warehouse


def test_config_seed_42():
    """Verify that both mode configurations explicitly declare seed: 42."""
    paper_cfg = load_paper_config()
    real_cfg = load_real_config()

    assert paper_cfg.get("seed") == 42
    assert real_cfg.get("seed") == 42


def test_product_catalog_determinism():
    """Verify ProductCatalog generates identical SKUs and ABC rankings with seed=42."""
    cat1 = ProductCatalog.generate_synthetic(n_products=100, seed=42)
    cat2 = ProductCatalog.generate_synthetic(n_products=100, seed=42)

    assert cat1.references == cat2.references
    assert [cat1[r].abc_class for r in cat1.references] == [cat2[r].abc_class for r in cat2.references]
    assert [cat1[r].sector for r in cat1.references] == [cat2[r].sector for r in cat2.references]


def test_storage_policy_determinism():
    """Verify storage policies allocate identical slots with seed=42."""
    wh1 = build_warehouse("paper")
    wh2 = build_warehouse("paper")
    cat1 = ProductCatalog.generate_synthetic(n_products=50, seed=42)
    cat2 = ProductCatalog.generate_synthetic(n_products=50, seed=42)

    pol1 = get_storage_policy("random", source="generated", warehouse=wh1, catalog=cat1, seed=42)
    pol2 = get_storage_policy("random", source="generated", warehouse=wh2, catalog=cat2, seed=42)

    for r in cat1.references:
        locs1 = pol1.get_locations_for_sku(r)
        locs2 = pol2.get_locations_for_sku(r)
        assert locs1 == locs2


def test_picking_wave_generation_determinism():
    """Verify synthetic picking wave sampling produces identical waves with seed=42."""
    wh = build_warehouse("paper")
    catalog = ProductCatalog.generate_synthetic(n_products=50, seed=42)
    policy = get_storage_policy("random", source="generated", warehouse=wh, catalog=catalog, seed=42)

    ds1 = PickingWaveDataset.generate_synthetic(
        catalog=catalog,
        storage_policy=policy,
        n_waves=3,
        items_per_wave=8,
        seed=42,
    )
    ds2 = PickingWaveDataset.generate_synthetic(
        catalog=catalog,
        storage_policy=policy,
        n_waves=3,
        items_per_wave=8,
        seed=42,
    )

    for w_id in ds1.wave_ids:
        items1 = [(it.reference, it.location) for it in ds1[w_id].items]
        items2 = [(it.reference, it.location) for it in ds2[w_id].items]
        assert items1 == items2


def test_gro_optimizer_determinism():
    """Verify GRO optimizer produces bitwise-identical best tours and distances given seed=42."""
    locs = ["L1", "L2", "L3", "L4", "L5"]
    all_points = ["DEPOT"] + locs
    dist_mat = {
        (u, v): float(abs(i - j) * 12.0)
        for i, u in enumerate(all_points)
        for j, v in enumerate(all_points)
    }

    cfg1 = GROConfig(population_size=20, iterations=15, tournament_size=3, random_seed=42, verbose=False)
    opt1 = GROOptimizer(config=cfg1, depot_id="DEPOT", distance_matrix=dist_mat)
    res1 = opt1.optimize(picking_locations=locs)

    cfg2 = GROConfig(population_size=20, iterations=15, tournament_size=3, random_seed=42, verbose=False)
    opt2 = GROOptimizer(config=cfg2, depot_id="DEPOT", distance_matrix=dist_mat)
    res2 = opt2.optimize(picking_locations=locs)

    assert res1.best_distance == res2.best_distance
    assert res1.best_route == res2.best_route
    assert res1.history_best_fitness == res2.history_best_fitness


def test_phase17_output_directory_structure():
    """
    Verify outputs/ directory tree matches Plan.md Section 22:
    outputs/
    ├── tables/
    │   ├── summary_statistics.csv
    │   ├── anova.csv
    │   └── tukey.csv
    ├── figures/
    │   ├── warehouse.png
    │   ├── boxplot.png
    │   └── scatter.png
    ├── routes/
    └── logs/
    """
    base_out = Path("outputs")
    assert base_out.exists()

    # Tables
    tables_dir = base_out / "tables"
    assert tables_dir.exists()
    assert (tables_dir / "summary_statistics.csv").exists()
    assert (tables_dir / "anova.csv").exists()
    assert (tables_dir / "tukey.csv").exists()

    # Figures
    figures_dir = base_out / "figures"
    assert figures_dir.exists()
    assert (figures_dir / "warehouse.png").exists()
    assert (figures_dir / "boxplot.png").exists()
    assert (figures_dir / "scatter.png").exists()

    # Routes & Logs
    routes_dir = base_out / "routes"
    assert routes_dir.exists()
    route_files = list(routes_dir.glob("*.json"))
    assert len(route_files) > 0

    logs_dir = base_out / "logs"
    assert logs_dir.exists()
    log_files = list(logs_dir.glob("*.json"))
    assert len(log_files) > 0


def test_phase17_experiment_log_schema():
    """
    Verify experiment log JSON format strictly adheres to Plan.md Section 22:
    {
      "seed": 42,
      "population_size": 1000,
      "generations": 1000,
      "tournament_size": 3,
      "crossover_probability": 0.9,
      "mutation_probability": 0.08,
      "storage_policy": "hybrid"
    }
    """
    log_file = Path("outputs/logs/experiment_log_hybrid.json")
    assert log_file.exists(), f"Log file missing: {log_file}"

    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "seed",
        "population_size",
        "generations",
        "tournament_size",
        "crossover_probability",
        "mutation_probability",
        "storage_policy",
    ]

    for k in required_keys:
        assert k in data, f"Required key '{k}' missing from experiment log JSON"

    assert data["seed"] == 42
    assert isinstance(data["population_size"], int)
    assert isinstance(data["generations"], int)
    assert data["tournament_size"] in (2, 3)
    assert pytest.approx(data["crossover_probability"]) == 0.9
    assert pytest.approx(data["mutation_probability"]) == 0.08
    assert data["storage_policy"] == "hybrid"
