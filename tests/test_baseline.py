"""
Unit tests for Baseline Routing Engine.
"""

import pytest

from slap_gro.data.product import ProductCatalog
from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.experiment import ExperimentRunner
from slap_gro.pipeline.paper_pipeline import build_paper_storage_policies
from slap_gro.picking.wave import PickingItem, PickingWave, PickingWaveDataset
from slap_gro.routing.baseline import (
    AVAILABLE_BASELINES,
    BaselineResult,
    NearestNeighborBaseline,
    OriginalCompanyRouteBaseline,
    ReconstructedFIFOBaseline,
    SShapeBaseline,
    evaluate_all_baselines,
    evaluate_baseline,
    get_baseline_heuristic,
)
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.warehouse_layout import SyntheticWarehouse


@pytest.fixture
def sample_depot():
    return Location(id="DEPOT", x=0.0, y=0.0, z=1.0, capacity=0)


@pytest.fixture
def sample_locations():
    return [
        Location(id="LOC_1", x=5.0, y=10.0, z=1.0, aisle=1),
        Location(id="LOC_2", x=5.0, y=30.0, z=1.0, aisle=1),
        Location(id="LOC_3", x=15.0, y=25.0, z=1.0, aisle=2),
        Location(id="LOC_4", x=15.0, y=10.0, z=1.0, aisle=2),
    ]


@pytest.fixture
def sample_wave():
    return PickingWave(
        wave_id=999,
        items=[
            PickingItem(reference="SKU_A", quantity=1, location="SYN-A-01-001"),
            PickingItem(reference="SKU_B", quantity=1, location="SYN-A-01-002"),
            PickingItem(reference="SKU_C", quantity=1, location="SYN-B-08-001"),
        ],
    )


class TestBaselineHeuristics:
    """Test individual heuristic routing strategies."""

    def test_original_baseline_differentiation(self, sample_depot, sample_locations):
        # Real-data Mode
        real_baseline = OriginalCompanyRouteBaseline(is_real_data=True)
        assert real_baseline.baseline_type == "actual_company_route"
        assert "thực tế" in real_baseline.description

        # Paper Mode / Reconstructed
        sim_baseline = OriginalCompanyRouteBaseline(is_real_data=False)
        assert sim_baseline.baseline_type == "reconstructed_baseline"
        assert "tái dựng" in sim_baseline.description

        # Route order preservation
        tour = real_baseline.route(sample_depot, sample_locations)
        assert tour[0].id == "DEPOT"
        assert tour[-1].id == "DEPOT"
        assert [loc.id for loc in tour[1:-1]] == ["LOC_1", "LOC_2", "LOC_3", "LOC_4"]

    def test_reconstructed_fifo_baseline(self, sample_depot, sample_locations):
        fifo = ReconstructedFIFOBaseline()
        assert fifo.baseline_type == "reconstructed_baseline"
        assert fifo.name == "fifo"
        tour = fifo.route(sample_depot, sample_locations)
        assert len(tour) == 6
        assert tour[0].id == "DEPOT"
        assert tour[-1].id == "DEPOT"

    def test_nearest_neighbor_baseline(self, sample_depot, sample_locations):
        nn = NearestNeighborBaseline()
        assert nn.baseline_type == "reconstructed_baseline"
        assert nn.name == "nearest_neighbor"

        tour = nn.route(sample_depot, sample_locations)
        assert tour[0].id == "DEPOT"
        assert tour[-1].id == "DEPOT"
        # Visited all unique locations
        visited_ids = [loc.id for loc in tour[1:-1]]
        assert set(visited_ids) == {"LOC_1", "LOC_2", "LOC_3", "LOC_4"}
        assert len(visited_ids) == 4

        # From (0,0), closest is LOC_1 (5, 10), then LOC_2 (5, 30), etc.
        assert visited_ids[0] == "LOC_1"

    def test_sshape_baseline(self, sample_depot, sample_locations):
        sshape = SShapeBaseline()
        assert sshape.baseline_type == "reconstructed_baseline"
        assert sshape.name == "s_shape"

        tour = sshape.route(sample_depot, sample_locations)
        assert tour[0].id == "DEPOT"
        assert tour[-1].id == "DEPOT"
        visited_ids = [loc.id for loc in tour[1:-1]]
        assert set(visited_ids) == {"LOC_1", "LOC_2", "LOC_3", "LOC_4"}

        # Aisle 1 (chẵn / 0): Y tăng dần -> LOC_1 (y=10), LOC_2 (y=30)
        # Aisle 2 (lẻ / 1): Y giảm dần -> LOC_3 (y=25), LOC_4 (y=10)
        assert visited_ids == ["LOC_1", "LOC_2", "LOC_3", "LOC_4"]

    def test_empty_and_single_pick_edge_cases(self, sample_depot):
        for h_cls in [OriginalCompanyRouteBaseline, NearestNeighborBaseline, SShapeBaseline]:
            h = h_cls()
            # Empty picks
            empty_tour = h.route(sample_depot, [])
            assert [loc.id for loc in empty_tour] == ["DEPOT", "DEPOT"]

            # Single pick
            single_loc = Location(id="SINGLE", x=10.0, y=10.0, z=1.0)
            single_tour = h.route(sample_depot, [single_loc])
            assert [loc.id for loc in single_tour] == ["DEPOT", "SINGLE", "DEPOT"]


class TestBaselineEvaluation:
    """Test evaluation helpers and result serialization."""

    def test_get_baseline_heuristic_factory(self):
        h_orig = get_baseline_heuristic("original", is_real_data=True)
        assert isinstance(h_orig, OriginalCompanyRouteBaseline)

        h_nn = get_baseline_heuristic("nn")
        assert isinstance(h_nn, NearestNeighborBaseline)

        h_s = get_baseline_heuristic("serpentine")
        assert isinstance(h_s, SShapeBaseline)

        with pytest.raises(ValueError, match="Không tìm thấy heuristic"):
            get_baseline_heuristic("unknown_heuristic_xyz")

    def test_evaluate_baseline_on_synthetic_warehouse(self, sample_wave):
        wh = SyntheticWarehouse()
        res = evaluate_baseline(
            wave=sample_wave,
            warehouse=wh,
            method="original",
            is_real_data=False,
        )

        assert isinstance(res, BaselineResult)
        assert res.wave_id == 999
        assert res.baseline_name == "original"
        assert res.baseline_type == "reconstructed_baseline"
        assert res.total_distance > 0.0
        assert res.stops_count > 0
        assert res.stops[0] == "DEPOT"
        assert res.stops[-1] == "DEPOT"

        d = res.to_dict()
        assert "stops_str" in d
        assert "total_distance" in d

    def test_evaluate_all_baselines(self, sample_wave):
        wh = SyntheticWarehouse()
        all_res = evaluate_all_baselines(
            wave=sample_wave,
            warehouse=wh,
            methods=["original", "nearest_neighbor", "s_shape"],
            is_real_data=False,
        )

        assert set(all_res.keys()) == {"original", "nearest_neighbor", "s_shape"}
        for name, r in all_res.items():
            assert r.total_distance > 0.0
            assert len(r.stops) >= 2


class TestExperimentRunnerBaselineIntegration:
    """Test ExperimentRunner integration with Baseline Routing Heuristics."""

    def test_runner_with_include_baseline(self):
        wh = SyntheticWarehouse()
        catalog = ProductCatalog.generate_synthetic(n_products=50, seed=42)
        policies = {
            "random": build_paper_storage_policies(wh, catalog, seed=42)["random"],
        }
        wave_dataset = PickingWaveDataset.generate_synthetic(
            catalog=catalog,
            storage_policy=policies["random"],
            n_waves=2,
            items_per_wave=5,
            seed=42,
        )

        cfg = GROConfig(population_size=10, iterations=5, random_seed=42)
        runner = ExperimentRunner(
            warehouse=wh,
            storage_policies=policies,
            gro_config=cfg,
            depot_id="DEPOT",
            mode="paper_replication",
        )

        # Run with include_baseline=True
        result = runner.run(
            waves=list(wave_dataset.waves.values()),
            max_waves=2,
            include_baseline=True,
            verbose=False,
        )

        # 1 policy x 2 waves + 2 baseline runs = 4 runs
        assert len(result) == 4
        df = result.to_dataframe()
        assert "original" in df["policy"].values
        assert "random" in df["policy"].values

        summary = result.summary_statistics()
        assert set(summary["policy"]) == {"original", "random"}
        assert "vs_original_pct" in summary.columns
