"""Class-Based Storage Allocation Policy for SLAP-GRO."""

from typing import Any, Dict, List, Literal, Optional
import numpy as np


from slap_gro.data.product import ProductCatalog
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.storage_matrix import StorageMatrix, StorageUnit, StoredItem
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


class ClassBasedStoragePolicy(StoragePolicy):
    """
    Class-Based Storage Assignment Policy.

    Partitions warehouse locations into 3 zones based on distance to the I/O Depot:
      - Zone A (~20% closest locations): assigned to fast-moving Class A items.
      - Zone B (~30% mid-distance locations): assigned to medium-turnover Class B items.
      - Zone C (~50% farthest locations): assigned to slow-moving Class C items.
    """

    def __init__(
        self,
        source: Literal["observed", "generated"] = "observed",
        matrix: Optional[StorageMatrix] = None,
        file_path: Optional[str] = None,
    ):
        super().__init__(name="class_based", source=source, matrix=matrix)
        if source == "observed" and self._matrix is None:
            self._matrix = StorageMatrix.from_csv(policy="class_based", file_path=file_path)

    def assign_storage(
        self,
        warehouse: WarehouseLayout,
        catalog: ProductCatalog,
        demand_profile: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> StorageMatrix:
        """Assign SKUs into distance-stratified ABC zones."""
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

        def _fill_zone(loc_list: List[Any], sku_pool: List[str], class_label: str):
            for loc in loc_list:
                slots: List[StoredItem] = []
                for _ in range(loc.capacity or 18):
                    sku = str(rng.choice(sku_pool))
                    slots.append(StoredItem(reference=sku, quantity=10.0))
                units_dict[loc.id] = StorageUnit(
                    location_id=loc.id,
                    slots=slots,
                    category_code=class_label,
                )

        _fill_zone(zone_a, skus_a, "A")
        _fill_zone(zone_b, skus_b, "B")
        _fill_zone(zone_c, skus_c, "C")

        self._matrix = StorageMatrix(policy="class_based", units=units_dict)
        return self._matrix
