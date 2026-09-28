"""Data schemas and domain records for SLAP-GRO."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProductRecord:
    """Represents a product item from Product.csv."""

    reference: str
    abc_class: str
    sector: str


@dataclass(frozen=True)
class StorageLocationRecord:
    """Represents a physical storage location from Storage_Location.csv."""

    location_id: str
    x: float
    y: float
    z: int
    block: Optional[str] = None
    aisle: Optional[int] = None
    level: Optional[int] = None

    @classmethod
    def from_raw(cls, location_id: str, x: float, y: float, z: int) -> "StorageLocationRecord":
        """Parse block and aisle from standard location string (e.g., 'A-14-11')."""
        loc = location_id.strip()
        parts = loc.split("-")
        block = parts[0] if len(parts) > 0 else None
        aisle = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        level = z
        return cls(
            location_id=loc,
            x=float(x),
            y=float(y),
            z=int(z),
            block=block,
            aisle=aisle,
            level=level,
        )


@dataclass(frozen=True)
class NavigationPointRecord:
    """Represents a navigation support point from Support_Points_Navigation.csv."""

    label: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CustomerOrderRecord:
    """Represents a customer order line from Customer_Order.csv."""

    customer_code: str
    order_number: int
    order_to_collect: int
    reference: str
    size_us: Optional[float]
    quantity: int
    creation_date: datetime
    wave_number: int
    operator: str


@dataclass(frozen=True)
class PickingWaveRecord:
    """Represents an item pick line from Picking_Wave.csv."""

    wave_number: int
    reference: str
    size_us: float
    quantity_to_pick: int
    location: str
    operator: str


@dataclass
class StorageSlot:
    """Represents a single slot within a storage location unit."""

    slot_id: int
    reference: Optional[str] = None
    quantity: float = 0.0


@dataclass
class StorageUnitRecord:
    """Represents a storage unit row with its 18 slots."""

    location_id: str
    policy: str
    category_code: Optional[str] = None
    slots: List[StorageSlot] = field(default_factory=list)


@dataclass
class AuditReportSummary:
    """Summary of data audit results."""

    total_files_audited: int
    file_summaries: Dict[str, Dict[str, Any]]
    discrepancies: List[Dict[str, Any]]
    all_checks_passed: bool
