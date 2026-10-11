#!/usr/bin/env python3
"""
CLI Script to execute Paper Replication Experiment (Mode A) - Phase 11.

Usage:
    python scripts/run_paper_replication.py --waves 50 --items 25 --iterations 1000 --pop 1000
    python scripts/run_paper_replication.py --fast  # Runs a 5-wave fast smoke test
"""

import argparse
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slap_gro.gro.optimizer import GROConfig
from slap_gro.pipeline.paper_pipeline import run_paper_experiment


def main():
    parser = argparse.ArgumentParser(
        description="SLAP-GRO Mode A: Paper Replication Experiment Runner (Phase 11)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=50,
        help="Number of picking waves to simulate and optimize (Paper default = 50)",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=25,
        help="Number of items per picking wave (Paper default = 25)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of GA iterations per wave optimization (Paper default = 1000)",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=1000,
        help="Population size for GRO genetic algorithm (Paper default = 1000)",
    )
    parser.add_argument(
        "--tournament",
        type=int,
        default=3,
        help="Tournament selection size (Paper default = 3)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.90,
        help="Crossover probability Pc (Paper default = 0.90)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.08,
        help="Mutation probability Pm (Paper default = 0.08)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic warehouse, catalog, waves and GRO",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML configuration file (e.g. configs/paper_replication.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/paper_replication",
        help="Output directory to save CSV and JSON experimental results",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 2 waves, 50 population, 20 iterations for quick verification",
    )

    args = parser.parse_args()

    n_waves = args.waves
    items_per_wave = args.items
    pop_size = args.pop
    n_iterations = args.iterations

    if args.fast:
        print(">>> FAST MODE ENABLED: Overriding settings for quick test run.")
        n_waves = 2
        items_per_wave = 10
        pop_size = 50
        n_iterations = 20

    gro_cfg = GROConfig(
        population_size=pop_size,
        iterations=n_iterations,
        tournament_size=args.tournament,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism=True,
        random_seed=args.seed,
    )

    result = run_paper_experiment(
        n_waves=n_waves,
        items_per_wave=items_per_wave,
        gro_config=gro_cfg,
        config_yaml=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=True,
    )

    print("\nSummary Results:")
    print(result.summary_statistics().to_string(index=False))


if __name__ == "__main__":
    main()
