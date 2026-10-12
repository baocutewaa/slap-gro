#!/usr/bin/env python3
"""
CLI Script to execute Real Data Experiment (Mode B).

Usage:
    # Staged progression:
    python scripts/run_real_data.py --stage 10x100
    python scripts/run_real_data.py --stage 50x100
    python scripts/run_real_data.py --stage 50x1000
    python scripts/run_real_data.py --stage full

    # Custom run or fast smoke test:
    python scripts/run_real_data.py --waves 5 --fast
"""

import argparse
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.real_pipeline import REAL_DATA_STAGES, run_real_experiment


def main():
    parser = argparse.ArgumentParser(
        description="SLAP-GRO Mode B: Real Data Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=list(REAL_DATA_STAGES.keys()),
        help="Predefined staged execution profile (e.g. 10x100, 50x100, 50x1000, full)",
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=10,
        help="Number of real waves to process if not using a predefined stage",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of GA iterations per wave",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=1000,
        help="Population size for GRO genetic algorithm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GRO genetic algorithm",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML configuration file (e.g. configs/real_data.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save CSV and JSON experimental results",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include Original baseline in evaluation",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 2 waves, 50 population, 20 iterations for quick verification",
    )

    args = parser.parse_args()

    max_waves = args.waves
    pop_size = args.pop
    n_iterations = args.iterations
    stage = args.stage

    if args.fast:
        print(">>> FAST MODE ENABLED: Overriding settings for quick verification.")
        max_waves = 2
        pop_size = 50
        n_iterations = 20
        stage = None

    gro_cfg = GROConfig(
        population_size=pop_size,
        iterations=n_iterations,
        tournament_size=3,
        crossover_rate=0.90,
        mutation_rate=0.08,
        elitism=True,
        random_seed=args.seed,
    )

    result = run_real_experiment(
        stage=stage,
        max_waves=max_waves,
        gro_config=gro_cfg,
        config_yaml=args.config,
        output_dir=args.output_dir,
        include_baseline=args.include_baseline,
        seed=args.seed,
        verbose=True,
    )

    print("\nSummary Results:")
    print(result.summary_statistics().to_string(index=False))


if __name__ == "__main__":
    main()
