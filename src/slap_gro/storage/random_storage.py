"""Random Storage Allocation Policy for SLAP-GRO."""

from typing import Dict, List, Literal, Optional
import numpy as np

from slap_gro.data.product import ProductCatalog
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.storage_matrix import StorageMatrix, StorageUnit, StoredItem
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


class RandomStoragePolicy(StoragePolicy):
    """
    Random Storage Assignment Policy.

    Every SKU has equal probability of being assigned to any available storage slot,
    maximizing space utilization without regard to product turnover.
    """

    def __init__(
        self,
        source: Literal["observed", "generated"] = "observed",
        matrix: Optional[StorageMatrix] = None,
        file_path: Optional[str] = None,
    ):
        super().__init__(name="random", source=source, matrix=matrix)
        if source == "observed" and self._matrix is None:
            self._matrix = StorageMatrix.from_csv(policy="random", file_path=file_path)

    def assign_storage(
        self,
        warehouse: WarehouseLayout,
        catalog: ProductCatalog,
        demand_profile: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> StorageMatrix:
        """
        Randomly assign catalog SKUs across warehouse storage units.

        Fills up to 18 slots per location with randomly selected SKUs.
        """
        rng = np.random.default_rng(seed)
        all_locations = warehouse.get_all_locations()
        skus = catalog.references
        if not skus:
            raise ValueError("Cannot assign storage with an empty product catalog")

        units_dict: Dict[str, StorageUnit] = {}
        for loc in all_locations:
            slots: List[StoredItem] = []
            for _ in range(loc.capacity or 18):
                # Sample random SKU
                sku = str(rng.choice(skus))
                slots.append(StoredItem(reference=sku, quantity=10.0))

            units_dict[loc.id] = StorageUnit(
                location_id=loc.id,
                slots=slots,
            )

        self._matrix = StorageMatrix(policy="random", units=units_dict)
        return self._matrix
