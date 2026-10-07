"""Dedicated Storage Allocation Policy for SLAP-GRO."""

from typing import Dict, List, Literal, Optional
import numpy as np

from slap_gro.data.product import ProductCatalog
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.storage_matrix import StorageMatrix, StorageUnit, StoredItem
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


class DedicatedStoragePolicy(StoragePolicy):
    """
    Dedicated Storage Assignment Policy.

    Assigns each SKU to a fixed, dedicated set of storage locations based strictly
    on historical picking demand ranking (highest-velocity SKUs get locations closest to Depot).
    """

    def __init__(
        self,
        source: Literal["observed", "generated"] = "observed",
        matrix: Optional[StorageMatrix] = None,
        file_path: Optional[str] = None,
    ):
        super().__init__(name="dedicated", source=source, matrix=matrix)
        if source == "observed" and self._matrix is None:
            self._matrix = StorageMatrix.from_csv(policy="dedicated", file_path=file_path)

    def assign_storage(
        self,
        warehouse: WarehouseLayout,
        catalog: ProductCatalog,
        demand_profile: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> StorageMatrix:
        """Assign SKUs to fixed locations ordered by demand ranking."""
        depot = warehouse.get_depot()
        all_locations = warehouse.get_all_locations()

        # Sort locations ascending by distance to Depot (closest first)
        sorted_locs = sorted(all_locations, key=lambda loc: loc.distance_to(depot))

        # Determine demand scores for SKUs
        if demand_profile is None:
            # Fallback demand based on ABC class
            weights = {"A": 80.0, "B": 15.0, "C": 5.0}
            demand_profile = {
                p.reference: weights.get(p.abc_class, 5.0) for p in catalog
            }

        # Sort SKUs descending by demand score
        sorted_skus = sorted(
            catalog.references,
            key=lambda ref: demand_profile.get(ref, 0.0),
            reverse=True,
        )

        if not sorted_skus:
            raise ValueError("Cannot assign storage with an empty product catalog")

        units_dict: Dict[str, StorageUnit] = {}
        n_skus = len(sorted_skus)

        for idx, loc in enumerate(sorted_locs):
            # Deterministically map SKU to location based on ranking
            assigned_sku = sorted_skus[idx % n_skus]
            slots = [
                StoredItem(reference=assigned_sku, quantity=10.0)
                for _ in range(loc.capacity or 18)
            ]
            units_dict[loc.id] = StorageUnit(
                location_id=loc.id,
                slots=slots,
            )

        self._matrix = StorageMatrix(policy="dedicated", units=units_dict)
        return self._matrix
