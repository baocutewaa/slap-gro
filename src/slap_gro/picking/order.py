"""Customer order analysis and wave batching utilities for SLAP-GRO."""

from typing import Any, Dict, List
import pandas as pd

from slap_gro.picking.wave import PickingItem, PickingWave, PickingWaveDataset


def batch_customer_orders(orders_df: pd.DataFrame) -> PickingWaveDataset:
    """
    Consolidate customer order lines into PickingWave batches grouped by waveNumber.

    Args:
        orders_df: Customer_Order DataFrame.

    Returns:
        PickingWaveDataset instance containing all consolidated waves.
    """
    wave_col = "waveNumber" if "waveNumber" in orders_df.columns else "wave_number"
    dataset = PickingWaveDataset()

    for w_id, group in orders_df.groupby(wave_col):
        items: List[PickingItem] = []
        for row in group.itertuples(index=False):
            ref = str(getattr(row, "Reference", getattr(row, "reference", "")))
            qty = int(getattr(row, "quantity (units)", getattr(row, "quantity", 1)))
            loc = str(getattr(row, "locations", getattr(row, "location", "")))
            size_val = getattr(row, "Size (US)", getattr(row, "size_us", None))
            size = float(size_val) if pd.notna(size_val) else None
            op = str(getattr(row, "operator", ""))

            items.append(
                PickingItem(
                    reference=ref,
                    quantity=qty,
                    location=loc,
                    size_us=size,
                    operator=op,
                )
            )
        dataset.waves[int(w_id)] = PickingWave(wave_id=int(w_id), items=items)

    return dataset
