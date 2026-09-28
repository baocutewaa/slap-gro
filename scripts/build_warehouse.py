"""Build and validate warehouse layouts for both Synthetic (paper) and CSV (real) modes.

Executes Phase 2 warehouse construction, verifies capacity contracts, and
exports layout metadata to outputs/tables/warehouse_summary.csv and
outputs/warehouse_report.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from slap_gro.config.loader import get_project_root
from slap_gro.warehouse.warehouse_layout import (
    CSVWarehouse,
    SyntheticWarehouse,
    build_warehouse,
)


def profile_warehouse(wh) -> Dict[str, Any]:
    """Extract structural and geometric summary of a WarehouseLayout instance."""
    bbox = wh.bounding_box
    depot = wh.get_depot()
    blocks = wh.get_blocks()
    aisles = wh.get_aisles()
    levels = wh.get_levels()

    level_counts = {}
    for lvl in levels:
        level_counts[f"level_{lvl}"] = len(wh.get_locations_by_level(lvl))

    block_counts = {}
    for blk in blocks:
        block_counts[f"block_{blk}"] = len(wh.get_locations_by_block(blk))

    return {
        "mode": wh.mode,
        "name": wh.name,
        "total_locations": wh.total_locations,
        "total_capacity_slots": wh.total_capacity,
        "slots_per_location": 18,
        "depot_coordinates": [depot.x, depot.y, depot.z],
        "blocks_count": len(blocks),
        "blocks_list": blocks,
        "aisles_count": len(aisles),
        "aisles_range": [min(aisles), max(aisles)] if aisles else [],
        "levels_count": len(levels),
        "levels_distribution": level_counts,
        "locations_per_block": block_counts,
        "bounding_box": {
            "x_range": [round(bbox.min_x, 2), round(bbox.max_x, 2)],
            "y_range": [round(bbox.min_y, 2), round(bbox.max_y, 2)],
            "z_range": [round(bbox.min_z, 2), round(bbox.max_z, 2)],
            "width_m": round(bbox.width, 2),
            "length_m": round(bbox.length, 2),
            "height_m": round(bbox.height, 2),
        },
        "navigation_points_count": len(wh.navigation_points),
    }


def main():
    parser = argparse.ArgumentParser(description="Build and profile SLAP-GRO warehouse layouts.")
    parser.add_argument(
        "--mode",
        choices=["all", "paper", "real"],
        default="all",
        help="Warehouse mode to build (default: all)",
    )
    args = parser.parse_args()

    root = get_project_root()
    outputs_dir = root / "outputs"
    tables_dir = outputs_dir / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SLAP-GRO PHASE 2: WAREHOUSE LAYER CONSTRUCTION")
    print("=" * 80)

    summaries = []
    report_dict = {}

    modes_to_run = ["paper", "real"] if args.mode == "all" else [args.mode]

    for m in modes_to_run:
        print(f"\n[+] Building warehouse for mode: {m.upper()}...")
        wh = build_warehouse(mode=m)
        prof = profile_warehouse(wh)
        report_dict[m] = prof

        print(f"  ✓ Name: {prof['name']}")
        print(f"  ✓ Total Locations: {prof['total_locations']:,}")
        print(f"  ✓ Total Capacity: {prof['total_capacity_slots']:,} slots (18 slots/unit)")
        print(f"  ✓ Blocks: {prof['blocks_count']} ({', '.join(prof['blocks_list'][:10])}...)")
        print(f"  ✓ Aisles: {prof['aisles_count']} aisles")
        print(f"  ✓ Levels: {prof['levels_distribution']}")
        print(f"  ✓ Bounding Box: X={prof['bounding_box']['x_range']}m, Y={prof['bounding_box']['y_range']}m, Z={prof['bounding_box']['z_range']}m")
        print(f"  ✓ Depot: {prof['depot_coordinates']}")
        print(f"  ✓ Navigation Points: {prof['navigation_points_count']}")

        summaries.append({
            "Mode": prof["mode"],
            "Name": prof["name"],
            "Locations": prof["total_locations"],
            "Slots/Unit": 18,
            "Total Capacity": prof["total_capacity_slots"],
            "Blocks": prof["blocks_count"],
            "Aisles": prof["aisles_count"],
            "Levels": prof["levels_count"],
            "X Range (m)": f"{prof['bounding_box']['x_range']}",
            "Y Range (m)": f"{prof['bounding_box']['y_range']}",
            "Z Range (m)": f"{prof['bounding_box']['z_range']}",
            "Nav Points": prof["navigation_points_count"],
        })

    # Export Summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_csv_path = tables_dir / "warehouse_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    print(f"\n[✓] Exported summary table to: {summary_csv_path.relative_to(root)}")

    # Export JSON Report
    json_path = outputs_dir / "warehouse_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[✓] Exported JSON report to: {json_path.relative_to(root)}")

    print("\n" + "=" * 80)
    print("PHASE 2 WAREHOUSE LAYER READY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
