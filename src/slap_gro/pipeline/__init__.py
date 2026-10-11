"""
Pipeline module for SLAP-GRO experiment execution and benchmark runners (Phase 11).
"""

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

__all__ = [
    "SingleRunResult",
    "ExperimentResult",
    "ExperimentRunner",
    "build_paper_storage_policies",
    "run_paper_experiment",
    "REAL_DATA_STAGES",
    "build_real_storage_policies",
    "run_real_experiment",
]
