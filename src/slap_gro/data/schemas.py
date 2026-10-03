"""Data schemas and domain records for SLAP-GRO."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class ProductRecord:
    """Represents a product item from Product.csv."""

    reference: str
    abc_class: str
    sector: str

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["ProductRecord"]:
        """Convert a normalized Product DataFrame to a list of ProductRecord instances."""
        records = []
        for row in df.itertuples(index=False):
            records.append(
                cls(
                    reference=str(getattr(row, "Reference", getattr(row, "reference", ""))),
                    abc_class=str(getattr(row, "ABCCOD", getattr(row, "abc_class", ""))),
                    sector=str(getattr(row, "Sector", getattr(row, "sector", ""))),
                )
            )
        return records


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

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["StorageLocationRecord"]:
        """Convert a normalized Storage_Location DataFrame to a list of StorageLocationRecord instances."""
        records = []
        for row in df.itertuples(index=False):
            loc_id = str(getattr(row, "originalLocation", getattr(row, "location_id", "")))
            x = float(getattr(row, "x"))
            y = float(getattr(row, "y"))
            z = int(getattr(row, "z"))
            records.append(cls.from_raw(loc_id, x, y, z))
        return records


@dataclass(frozen=True)
class NavigationPointRecord:
    """Represents a navigation support point from Support_Points_Navigation.csv."""

    label: str
    x: float
    y: float
    z: float

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["NavigationPointRecord"]:
        """Convert a normalized NavigationPoint DataFrame to a list of NavigationPointRecord instances."""
        from slap_gro.data.preprocess import parse_coordinate_tuple

        records = []
        for row in df.itertuples(index=False):
            label = str(getattr(row, "labels", getattr(row, "label", "")))
            if hasattr(row, "x") and hasattr(row, "y") and hasattr(row, "z"):
                x = float(getattr(row, "x"))
                y = float(getattr(row, "y"))
                z = float(getattr(row, "z"))
            else:
                raw_pt = getattr(row, "points_specified", "")
                x, y, z = parse_coordinate_tuple(raw_pt)
            records.append(cls(label=label, x=x, y=y, z=z))
        return records



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

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["CustomerOrderRecord"]:
        """Convert a normalized Customer_Order DataFrame to a list of CustomerOrderRecord instances."""
        records = []
        for row in df.itertuples(index=False):
            size_val = getattr(row, "Size (US)", getattr(row, "size_us", None))
            size = float(size_val) if pd.notna(size_val) else None
            date_val = getattr(row, "creationDate", getattr(row, "creation_date", None))
            c_date = pd.to_datetime(date_val).to_pydatetime() if pd.notna(date_val) else datetime.min
            records.append(
                cls(
                    customer_code=str(getattr(row, "codCustomer", getattr(row, "customer_code", ""))),
                    order_number=int(getattr(row, "orderNumber", getattr(row, "order_number", 0))),
                    order_to_collect=int(getattr(row, "orderToCollect", getattr(row, "order_to_collect", 0))),
                    reference=str(getattr(row, "Reference", getattr(row, "reference", ""))),
                    size_us=size,
                    quantity=int(getattr(row, "quantity (units)", getattr(row, "quantity", 1))),
                    creation_date=c_date,
                    wave_number=int(getattr(row, "waveNumber", getattr(row, "wave_number", 0))),
                    operator=str(getattr(row, "operator", "")),
                )
            )
        return records


@dataclass(frozen=True)
class PickingWaveRecord:
    """Represents an item pick line from Picking_Wave.csv."""

    wave_number: int
    reference: str
    size_us: float
    quantity_to_pick: int
    location: str
    operator: str

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["PickingWaveRecord"]:
        """Convert a normalized Picking_Wave DataFrame to a list of PickingWaveRecord instances."""
        records = []
        for row in df.itertuples(index=False):
            records.append(
                cls(
                    wave_number=int(getattr(row, "waveNumber", getattr(row, "wave_number", 0))),
                    reference=str(getattr(row, "reference", "")),
                    size_us=float(getattr(row, "Size (US)", getattr(row, "size_us", 0.0))),
                    quantity_to_pick=int(getattr(row, "quantityToPick (units)", getattr(row, "quantity_to_pick", 1))),
                    location=str(getattr(row, "locations", getattr(row, "location", ""))),
                    operator=str(getattr(row, "operator", "")),
                )
            )
        return records


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

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, policy: str = "random") -> List["StorageUnitRecord"]:
        """Convert a normalized Storage Matrix DataFrame to a list of StorageUnitRecord instances."""
        from slap_gro.data.preprocess import parse_slot_value

        loc_col = "originalLocation" if "originalLocation" in df.columns else (
            "Location" if "Location" in df.columns else None
        )
        non_slot = {loc_col, "ABCCOD", "XYZCOD"}
        slot_cols = [c for c in df.columns if c not in non_slot]

        records = []
        for _, row in df.iterrows():
            loc_id = str(row[loc_col]) if loc_col else ""
            cat_code = str(row["ABCCOD"]) if "ABCCOD" in df.columns else (
                str(row["XYZCOD"]) if "XYZCOD" in df.columns else None
            )

            slots = []
            for idx, c in enumerate(slot_cols, start=1):
                raw_val = row[c]
                parsed = parse_slot_value(raw_val)
                if parsed:
                    sku, qty = parsed
                    slots.append(StorageSlot(slot_id=idx, reference=sku, quantity=qty))
                else:
                    slots.append(StorageSlot(slot_id=idx, reference=None, quantity=0.0))

            records.append(
                cls(
                    location_id=loc_id,
                    policy=policy,
                    category_code=cat_code,
                    slots=slots,
                )
            )
        return records



@dataclass
class AuditReportSummary:
    """Summary of data audit results."""

    total_files_audited: int
    file_summaries: Dict[str, Dict[str, Any]]
    discrepancies: List[Dict[str, Any]]
    all_checks_passed: bool
