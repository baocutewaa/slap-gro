"""Hybrid Storage Allocation Policy for SLAP-GRO."""

from typing import Dict, List, Literal, Optional
import numpy as np

from slap_gro.data.product import ProductCatalog
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.storage_matrix import StorageMatrix, StorageUnit, StoredItem
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


class HybridStoragePolicy(StoragePolicy):
    """
    Hybrid Storage Assignment Policy.

    Combines Dedicated and Class-Based storage:
      - Class A items (top 20% by turnover) are assigned to fixed, dedicated locations
        closest to the Depot.
      - Remaining items (Class B & C, 80%) are randomized within their respective
        demand-stratified zones.
    """

    def __init__(
        self,
        source: Literal["observed", "generated"] = "observed",
        matrix: Optional[StorageMatrix] = None,
        file_path: Optional[str] = None,
    ):
        super().__init__(name="hybrid", source=source, matrix=matrix)
        if source == "observed" and self._matrix is None:
            self._matrix = StorageMatrix.from_csv(policy="hybrid", file_path=file_path)

    def assign_storage(
        self,
        warehouse: WarehouseLayout,
        catalog: ProductCatalog,
        demand_profile: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> StorageMatrix:
        """Assign Class A to fixed closest locations, randomize Class B and C in zones."""
        rng = np.random.default_rng(seed)
        depot = warehouse.get_depot()
        all_locations = warehouse.get_all_locations()

        # Sort locations ascending by distance to Depot
        sorted_locs = sorted(all_locations, key=lambda loc: loc.distance_to(depot))
        n_total = len(sorted_locs)
        n_a = int(round(n_total * 0.20))
        n_b = int(round(n_total * 0.30))

        zone_a = sorted_locs[:n_a]
        zone_b = sorted_locs[n_a : n_a + n_b]
        zone_c = sorted_locs[n_a + n_b :]

        skus_a = [p.reference for p in catalog.get_by_abc_class("A")] or catalog.references
        skus_b = [p.reference for p in catalog.get_by_abc_class("B")] or catalog.references
        skus_c = [p.reference for p in catalog.get_by_abc_class("C")] or catalog.references

        units_dict: Dict[str, StorageUnit] = {}

        # 1. Zone A: Fixed dedicated assignment for Class A
        n_skus_a = len(skus_a)
        for idx, loc in enumerate(zone_a):
            fixed_sku = skus_a[idx % n_skus_a]
            slots = [
                StoredItem(reference=fixed_sku, quantity=10.0)
                for _ in range(loc.capacity or 18)
            ]
            units_dict[loc.id] = StorageUnit(
                location_id=loc.id,
                slots=slots,
                category_code="A",
            )

        # 2. Zone B: Randomized assignment within Zone B
        for loc in zone_b:
            slots = [
                StoredItem(reference=str(rng.choice(skus_b)), quantity=10.0)
                for _ in range(loc.capacity or 18)
            ]
            units_dict[loc.id] = StorageUnit(
                location_id=loc.id,
                slots=slots,
                category_code="B",
            )

        # 3. Zone C: Randomized assignment within Zone C
        for loc in zone_c:
            slots = [
                StoredItem(reference=str(rng.choice(skus_c)), quantity=10.0)
                for _ in range(loc.capacity or 18)
            ]
            units_dict[loc.id] = StorageUnit(
                location_id=loc.id,
                slots=slots,
                category_code="C",
            )

        self._matrix = StorageMatrix(policy="hybrid", units=units_dict)
        return self._matrix
