"""Storage Matrix domain representation and parsing utilities for SLAP-GRO."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import pandas as pd

from slap_gro.data.preprocess import clean_string_column, parse_slot_value


@dataclass(frozen=True)
class StoredItem:
    """
    Represents an item stored in a specific warehouse storage slot.

    Attributes:
        reference: SKU identifier.
        quantity: Quantity of the item stored in the slot.
    """

    reference: str
    quantity: float

    def __post_init__(self):
        object.__setattr__(self, "reference", str(self.reference).strip())
        object.__setattr__(self, "quantity", float(self.quantity))

    @property
    def is_empty(self) -> bool:
        """Check if slot is unoccupied."""
        return not self.reference or self.quantity <= 0.0


def parse_slot(value: Any) -> Optional[StoredItem]:
    """
    Parse a storage slot cell formatted as 'SKU;quantity' or plain 'SKU'.

    Example:
        '8551FLX;15.0' -> StoredItem(reference='8551FLX', quantity=15.0)
        'TQBVRI;7' -> StoredItem(reference='TQBVRI', quantity=7.0)
        'SKU_001' -> StoredItem(reference='SKU_001', quantity=1.0)
        None / NaN / '' -> None

    Returns:
        StoredItem if non-empty, otherwise None.
    """
    parsed = parse_slot_value(value)
    if parsed is None:
        return None
    sku, qty = parsed
    if not sku:
        return None
    return StoredItem(reference=sku, quantity=qty)


@dataclass
class StorageUnit:
    """
    Represents a physical storage location unit comprising 18 vertical slots.

    Attributes:
        location_id: Unique location identifier (e.g., 'A-14-11').
        slots: List of StoredItem instances (empty slots represented with reference='' or quantity=0.0).
        category_code: Optional ABC or XYZ class associated with the unit.
    """

    location_id: str
    slots: List[StoredItem] = field(default_factory=list)
    category_code: Optional[str] = None

    def __post_init__(self):
        self.location_id = str(self.location_id).strip()

    @property
    def total_slots(self) -> int:
        """Total number of slots (typically 18)."""
        return len(self.slots)

    @property
    def occupied_slots_count(self) -> int:
        """Number of non-empty slots in this unit."""
        return sum(1 for s in self.slots if not s.is_empty)

    @property
    def is_empty(self) -> bool:
        """Check if unit has zero stored items."""
        return self.occupied_slots_count == 0

    def get_items(self) -> List[StoredItem]:
        """Return list of non-empty stored items in this unit."""
        return [s for s in self.slots if not s.is_empty]

    def contains_sku(self, reference: str) -> bool:
        """Check if a specific SKU is stored in this unit."""
        ref = str(reference).strip()
        return any(s.reference == ref and not s.is_empty for s in self.slots)

    def get_quantity(self, reference: str) -> float:
        """Calculate total quantity of a given SKU stored in this unit."""
        ref = str(reference).strip()
        return sum(s.quantity for s in self.slots if s.reference == ref)


@dataclass
class StorageMatrix:
    """
    Complete warehouse storage allocation matrix.

    Maps physical location IDs to StorageUnit objects, supporting fast SKU lookups
    and spatial inventory queries across all 4 policies.
    """

    policy: str
    units: Dict[str, StorageUnit] = field(default_factory=dict)
    _sku_to_locations: Dict[str, List[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.rebuild_sku_index()

    def rebuild_sku_index(self) -> None:
        """Index all locations containing each SKU for fast O(1) retrieval."""
        index: Dict[str, List[str]] = {}
        for loc_id, unit in self.units.items():
            for item in unit.get_items():
                if item.reference:
                    index.setdefault(item.reference, []).append(loc_id)
        self._sku_to_locations = {k: list(dict.fromkeys(v)) for k, v in index.items()}

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self) -> Iterator[StorageUnit]:
        return iter(self.units.values())

    def __contains__(self, location_id: str) -> bool:
        return str(location_id).strip() in self.units

    def __getitem__(self, location_id: str) -> StorageUnit:
        key = str(location_id).strip()
        if key not in self.units:
            raise KeyError(f"Storage location '{key}' not found in matrix")
        return self.units[key]

    def get_unit(self, location_id: str) -> Optional[StorageUnit]:
        """Safely retrieve a storage unit by location ID."""
        return self.units.get(str(location_id).strip())

    def get_items_at_location(self, location_id: str) -> List[StoredItem]:
        """Get all stored items residing at a specific location."""
        unit = self.get_unit(location_id)
        return unit.get_items() if unit else []

    def get_locations_for_sku(self, reference: str) -> List[str]:
        """Find all location IDs where a SKU is stored."""
        return self._sku_to_locations.get(str(reference).strip(), [])

    @property
    def location_ids(self) -> List[str]:
        """List of all storage location IDs in the matrix."""
        return list(self.units.keys())

    @property
    def stored_skus(self) -> Set[str]:
        """Set of all unique SKUs present in the matrix."""
        return set(self._sku_to_locations.keys())

    @property
    def total_capacity(self) -> int:
        """Total slots across all storage units (e.g. 2292 * 18 = 41,256)."""
        return sum(u.total_slots for u in self.units.values())

    @property
    def total_occupied_slots(self) -> int:
        """Count of occupied slots across the entire warehouse."""
        return sum(u.occupied_slots_count for u in self.units.values())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, policy: str = "random") -> "StorageMatrix":
        """
        Build StorageMatrix from a normalized storage DataFrame.

        Detects location column ('originalLocation' or 'Location') and 18 slot columns.
        """
        loc_col = "originalLocation" if "originalLocation" in df.columns else (
            "Location" if "Location" in df.columns else None
        )
        if not loc_col:
            raise ValueError("Dataframe must contain 'originalLocation' or 'Location' column")

        non_slot = {loc_col, "ABCCOD", "XYZCOD"}
        slot_cols = [c for c in df.columns if c not in non_slot]

        units_dict: Dict[str, StorageUnit] = {}
        for _, row in df.iterrows():
            loc_id = str(row[loc_col]).strip()
            cat_code = str(row["ABCCOD"]).strip() if "ABCCOD" in df.columns and pd.notna(row["ABCCOD"]) else (
                str(row["XYZCOD"]).strip() if "XYZCOD" in df.columns and pd.notna(row["XYZCOD"]) else None
            )

            slots: List[StoredItem] = []
            for col in slot_cols:
                item = parse_slot(row[col])
                if item:
                    slots.append(item)
                else:
                    slots.append(StoredItem(reference="", quantity=0.0))

            units_dict[loc_id] = StorageUnit(
                location_id=loc_id,
                slots=slots,
                category_code=cat_code,
            )

        return cls(policy=policy, units=units_dict)

    @classmethod
    def from_csv(
        cls,
        policy: str = "random",
        file_path: Optional[str] = None,
    ) -> "StorageMatrix":
        """Load storage matrix directly from raw or normalized CSV."""
        from slap_gro.data.loaders import load_storage_matrix

        df = load_storage_matrix(policy=policy, file_path=file_path)  # type: ignore
        return cls.from_dataframe(df, policy=policy)

    def to_dataframe(self) -> pd.DataFrame:
        """Export StorageMatrix back to a standard 18-slot DataFrame."""
        rows = []
        for loc_id, unit in self.units.items():
            row_dict: Dict[str, Any] = {"Location": loc_id}
            if unit.category_code:
                row_dict["ABCCOD"] = unit.category_code

            for idx, item in enumerate(unit.slots, start=1):
                col_name = f"slot_{idx}"
                if not item.is_empty:
                    row_dict[col_name] = f"{item.reference};{item.quantity:g}"
                else:
                    row_dict[col_name] = ""
            rows.append(row_dict)
        return pd.DataFrame(rows)
