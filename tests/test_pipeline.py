"""
Unit tests for SLAP-GRO Pipeline & Experiment Runner.
"""

from pathlib import Path
import pytest
import pandas as pd

from slap_gro.data.product import ProductCatalog
from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.experiment import (
    ExperimentResult,
    ExperimentRunner,
    SingleRunResult,
)
from slap_gro.pipeline.paper_pipeline import (
    build_paper_storage_policies,
    run_paper_experiment,
)
from slap_gro.pipeline.real_pipeline import (
    REAL_DATA_STAGES,
    build_real_storage_policies,
    run_real_experiment,
)
from slap_gro.picking.wave import PickingWave, PickingWaveDataset
from slap_gro.warehouse.warehouse_layout import SyntheticWarehouse


@pytest.fixture
def mini_gro_config():
    """Ultra-fast GRO configuration for test assertions."""
    return GROConfig(
        population_size=10,
        iterations=5,
        tournament_size=2,
        crossover_rate=0.8,
        mutation_rate=0.1,
        random_seed=42,
    )


class TestSingleRunResult:
    """Tests for SingleRunResult data structure."""

    def test_single_run_result_creation_and_dict(self):
        result = SingleRunResult(
            mode="paper_replication",
            policy="random",
            wave_id=1,
            num_picks=10,
            unique_locations=8,
            initial_distance=250.0,
            best_distance=180.0,
            improvement_pct=28.0,
            iterations=5,
            computation_time_s=0.12,
            best_route=["DEPOT", "LOC_1", "LOC_2", "DEPOT"],
        )

        assert result.policy == "random"
        assert result.wave_id == 1
        assert result.initial_distance == 250.0
        assert result.best_distance == 180.0
        assert result.improvement_pct == 28.0

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["policy"] == "random"
        assert d["initial_distance"] == 250.0
        assert d["best_route"] == "DEPOT -> LOC_1 -> LOC_2 -> DEPOT"


class TestExperimentResult:
    """Tests for ExperimentResult collection, dataframe export, and statistics."""

    def test_experiment_result_aggregation_and_summary(self, tmp_path):
        exp = ExperimentResult(mode="test_mode")

        # Add 2 runs for random policy
        exp.add_run(
            SingleRunResult(
                mode="test_mode",
                policy="random",
                wave_id=1,
                num_picks=5,
                unique_locations=5,
                initial_distance=200.0,
                best_distance=150.0,
                improvement_pct=25.0,
                iterations=5,
                computation_time_s=0.1,
            )
        )
        exp.add_run(
            SingleRunResult(
                mode="test_mode",
                policy="random",
                wave_id=2,
                num_picks=5,
                unique_locations=5,
                initial_distance=180.0,
                best_distance=140.0,
                improvement_pct=22.2,
                iterations=5,
                computation_time_s=0.1,
            )
        )

        # Add 2 runs for hybrid policy
        exp.add_run(
            SingleRunResult(
                mode="test_mode",
                policy="hybrid",
                wave_id=1,
                num_picks=5,
                unique_locations=5,
                initial_distance=120.0,
                best_distance=90.0,
                improvement_pct=25.0,
                iterations=5,
                computation_time_s=0.1,
            )
        )
        exp.add_run(
            SingleRunResult(
                mode="test_mode",
                policy="hybrid",
                wave_id=2,
                num_picks=5,
                unique_locations=5,
                initial_distance=110.0,
                best_distance=80.0,
                improvement_pct=27.3,
                iterations=5,
                computation_time_s=0.1,
            )
        )

        assert len(exp) == 4
        assert exp.total_runs == 4

        # Test to_dataframe
        df = exp.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "policy" in df.columns
        assert "best_distance" in df.columns

        # Test summary_statistics
        summary = exp.summary_statistics()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # 2 policies: random, hybrid
        assert "mean" in summary.columns
        assert "median" in summary.columns
        assert "mean_improvement_pct" in summary.columns

        # Check save_results
        saved = exp.save_results(tmp_path / "test_out")
        assert Path(saved["detailed_csv"]).exists()
        assert Path(saved["summary_csv"]).exists()
        assert Path(saved["summary_json"]).exists()


class TestExperimentRunner:
    """Tests for ExperimentRunner engine."""

    def test_runner_validates_inputs(self, mini_gro_config):
        wh = SyntheticWarehouse()
        with pytest.raises(ValueError, match="At least one storage policy"):
            ExperimentRunner(warehouse=wh, storage_policies={}, gro_config=mini_gro_config)

    def test_runner_execution_on_synthetic_warehouse(self, mini_gro_config):
        wh = SyntheticWarehouse()
        catalog = ProductCatalog.generate_synthetic(n_products=100, seed=42)
        policies = {
            "random": build_paper_storage_policies(wh, catalog, seed=42)["random"],
            "hybrid": build_paper_storage_policies(wh, catalog, seed=42)["hybrid"],
        }

        wave_dataset = PickingWaveDataset.generate_synthetic(
            catalog=catalog,
            storage_policy=policies["random"],
            n_waves=2,
            items_per_wave=6,
            seed=42,
        )

        runner = ExperimentRunner(
            warehouse=wh,
            storage_policies=policies,
            gro_config=mini_gro_config,
            depot_id="DEPOT",
            mode="paper_replication",
        )

        result = runner.run(waves=list(wave_dataset.waves.values()), max_waves=2, verbose=False)
        assert len(result) == 4  # 2 policies x 2 waves
        assert result.total_runs == 4

        df = result.to_dataframe()
        assert set(df["policy"].unique()) == {"random", "hybrid"}
        assert set(df["wave_id"].unique()) == {1, 2}
        assert (df["best_distance"] <= df["initial_distance"]).all()


class TestPaperPipeline:
    """Tests for Mode A Paper Replication Pipeline."""

    def test_build_paper_storage_policies(self):
        wh = SyntheticWarehouse()
        catalog = ProductCatalog.generate_synthetic(n_products=50, seed=42)
        policies = build_paper_storage_policies(wh, catalog, seed=42)

        assert set(policies.keys()) == {"random", "class_based", "dedicated", "hybrid"}
        for pol in policies.values():
            assert pol.matrix is not None
            assert len(pol.matrix) > 0

    def test_run_paper_experiment_fast(self, mini_gro_config, tmp_path):
        res = run_paper_experiment(
            n_waves=2,
            items_per_wave=5,
            gro_config=mini_gro_config,
            output_dir=tmp_path / "paper_test",
            seed=42,
            verbose=False,
        )

        assert len(res) == 8  # 4 policies x 2 waves
        summary = res.summary_statistics()
        assert len(summary) == 4
        assert set(summary["policy"]) == {"random", "class_based", "dedicated", "hybrid"}


class TestRealPipeline:
    """Tests for Mode B Real Data Pipeline."""

    def test_build_real_storage_policies(self):
        policies = build_real_storage_policies()
        assert set(policies.keys()) == {"random", "class_based", "dedicated", "hybrid"}
        for pol in policies.values():
            assert pol.matrix is not None
            assert len(pol.matrix) >= 2292

    def test_real_pipeline_invalid_stage(self):
        with pytest.raises(ValueError, match="không hợp lệ"):
            run_real_experiment(stage="invalid_stage_99")

    def test_run_real_experiment_fast(self, mini_gro_config, tmp_path):
        res = run_real_experiment(
            max_waves=2,
            gro_config=mini_gro_config,
            output_dir=tmp_path / "real_test",
            seed=42,
            verbose=False,
        )

        assert len(res) == 8  # 4 policies x 2 waves
        summary = res.summary_statistics()
        assert len(summary) == 4
        assert set(summary["policy"]) == {"random", "class_based", "dedicated", "hybrid"}
