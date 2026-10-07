"""Base abstract class for SLAP-GRO storage allocation policies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Sequence, Set
import pandas as pd

from slap_gro.data.product import ProductCatalog
from slap_gro.storage.storage_matrix import StorageMatrix, StorageUnit, StoredItem
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


class StoragePolicy(ABC):
    """
    Abstract base class for storage allocation strategies.

    Supports both:
    1. 'observed': Loads the pre-existing assignment matrix from industrial CSV datasets.
    2. 'generated': Programmatically generates assignment based on warehouse geometry,
       product ABC classifications, and demand profiles.
    """

    def __init__(
        self,
        name: str,
        source: Literal["observed", "generated"] = "observed",
        matrix: Optional[StorageMatrix] = None,
    ):
        self.name = name.lower().replace("-", "_")
        self.source = source
        self._matrix = matrix

    @property
    def matrix(self) -> StorageMatrix:
        """Underlying storage matrix."""
        if self._matrix is None:
            raise ValueError(f"Storage policy '{self.name}' has not been initialized or assigned yet")
        return self._matrix

    def get_locations_for_sku(self, reference: str) -> List[str]:
        """Find all location IDs where a SKU is stored."""
        return self.matrix.get_locations_for_sku(reference)

    def get_items_at_location(self, location_id: str) -> List[StoredItem]:
        """Retrieve stored items at a specified location."""
        return self.matrix.get_items_at_location(location_id)

    def get_all_occupied_locations(self) -> Set[str]:
        """Set of all location IDs that contain at least one stored item."""
        return {loc_id for loc_id, unit in self.matrix.units.items() if not unit.is_empty}

    @abstractmethod
    def assign_storage(
        self,
        warehouse: WarehouseLayout,
        catalog: ProductCatalog,
        demand_profile: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> StorageMatrix:
        """Programmatically assign products to warehouse storage units."""
        pass
