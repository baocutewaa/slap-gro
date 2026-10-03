"""Data layer package for SLAP-GRO."""

from slap_gro.data.loaders import (
    load_customer_orders,
    load_navigation_points,
    load_orders,
    load_picking_waves,
    load_products,
    load_storage_locations,
    load_storage_matrix,
)
from slap_gro.data.preprocess import (
    clean_string_column,
    normalize_customer_orders,
    normalize_navigation_points,
    normalize_orders,
    normalize_picking_waves,
    normalize_products,
    normalize_storage_locations,
    normalize_storage_matrix,
    parse_coordinate_tuple,
    parse_position_string,
    parse_slot_value,
)
from slap_gro.data.product import Product, ProductCatalog, load_product_catalog
from slap_gro.data.schemas import (
    AuditReportSummary,
    CustomerOrderRecord,
    NavigationPointRecord,
    PickingWaveRecord,
    ProductRecord,
    StorageLocationRecord,
    StorageSlot,
    StorageUnitRecord,
)
from slap_gro.data.validators import (
    validate_customer_orders,
    validate_navigation_points,
    validate_orders,
    validate_picking_waves,
    validate_products,
    validate_storage_locations,
    validate_storage_matrix,
)

__all__ = [
    # Loaders
    "load_products",
    "load_orders",
    "load_customer_orders",
    "load_picking_waves",
    "load_storage_locations",
    "load_navigation_points",
    "load_storage_matrix",
    "load_product_catalog",
    # Normalizers & Parsers
    "normalize_products",
    "normalize_orders",
    "normalize_customer_orders",
    "normalize_picking_waves",
    "normalize_storage_locations",
    "normalize_navigation_points",
    "normalize_storage_matrix",
    "parse_slot_value",
    "parse_coordinate_tuple",
    "parse_position_string",
    "clean_string_column",
    # Domain Entities
    "Product",
    "ProductCatalog",
    # Schemas & Records
    "ProductRecord",
    "StorageLocationRecord",
    "NavigationPointRecord",
    "CustomerOrderRecord",
    "PickingWaveRecord",
    "StorageSlot",
    "StorageUnitRecord",
    "AuditReportSummary",
    # Validators
    "validate_products",
    "validate_orders",
    "validate_storage_locations",
    "validate_navigation_points",
    "validate_customer_orders",
    "validate_picking_waves",
    "validate_storage_matrix",
]

