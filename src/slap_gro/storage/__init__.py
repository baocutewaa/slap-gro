"""Storage assignment package for SLAP-GRO."""

from typing import Dict, Literal, Optional

from slap_gro.data.product import ProductCatalog
from slap_gro.storage.base import StoragePolicy
from slap_gro.storage.class_based import ClassBasedStoragePolicy
from slap_gro.storage.dedicated_storage import DedicatedStoragePolicy
from slap_gro.storage.hybrid_storage import HybridStoragePolicy
from slap_gro.storage.random_storage import RandomStoragePolicy
from slap_gro.storage.storage_matrix import (
    StorageMatrix,
    StorageUnit,
    StoredItem,
    parse_slot,
)
from slap_gro.warehouse.warehouse_layout import WarehouseLayout


def get_storage_policy(
    policy: str,
    source: Literal["observed", "generated"] = "observed",
    warehouse: Optional[WarehouseLayout] = None,
    catalog: Optional[ProductCatalog] = None,
    demand_profile: Optional[Dict[str, float]] = None,
    seed: Optional[int] = 42,
    file_path: Optional[str] = None,
) -> StoragePolicy:
    """
    Factory function to retrieve or generate a StoragePolicy instance.

    Args:
        policy: One of 'random', 'class_based', 'dedicated', 'hybrid'.
        source: 'observed' (load pre-assigned CSV matrix) or 'generated' (programmatically assign).
        warehouse: WarehouseLayout instance (required if source='generated').
        catalog: ProductCatalog instance (required if source='generated').
        demand_profile: Optional SKU demand mapping.
        seed: Random seed for reproducibility.
        file_path: Optional custom path to matrix CSV if source='observed'.

    Returns:
        StoragePolicy instance with active .matrix.
    """
    key = policy.lower().replace("-", "_")
    policies = {
        "random": RandomStoragePolicy,
        "class_based": ClassBasedStoragePolicy,
        "dedicated": DedicatedStoragePolicy,
        "hybrid": HybridStoragePolicy,
    }

    if key not in policies:
        raise ValueError(f"Unknown storage policy: '{policy}'. Available policies: {list(policies.keys())}")

    policy_cls = policies[key]

    if source == "observed":
        return policy_cls(source="observed", file_path=file_path)

    if warehouse is None or catalog is None:
        raise ValueError("Must provide 'warehouse' and 'catalog' when source='generated'")

    instance = policy_cls(source="generated")
    instance.assign_storage(
        warehouse=warehouse,
        catalog=catalog,
        demand_profile=demand_profile,
        seed=seed,
    )
    return instance


__all__ = [
    "StoredItem",
    "StorageUnit",
    "StorageMatrix",
    "parse_slot",
    "StoragePolicy",
    "RandomStoragePolicy",
    "ClassBasedStoragePolicy",
    "DedicatedStoragePolicy",
    "HybridStoragePolicy",
    "get_storage_policy",
]
