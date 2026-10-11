"""Picking wave domain entities and repository for SLAP-GRO."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set
import numpy as np
import pandas as pd

from slap_gro.data.product import ProductCatalog
from slap_gro.data.schemas import PickingWaveRecord
from slap_gro.storage.base import StoragePolicy


@dataclass(frozen=True)
class PickingItem:
    """
    Represents an individual picking instruction within a wave.

    Attributes:
        reference: SKU identifier to pick.
        quantity: Number of units to pick (typically 1 in unit-picking dataset).
        location: Storage location ID where the SKU is placed.
        size_us: Optional shoe/item size.
        operator: Picker/operator identifier.
    """

    reference: str
    quantity: int
    location: str
    size_us: Optional[float] = None
    operator: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "reference", str(self.reference).strip())
        object.__setattr__(self, "quantity", int(self.quantity))
        object.__setattr__(self, "location", str(self.location).strip())
        if self.size_us is not None:
            object.__setattr__(self, "size_us", float(self.size_us))
        if self.operator is not None:
            object.__setattr__(self, "operator", str(self.operator).strip())


@dataclass
class PickingWave:
    """
    Represents a consolidated picking wave (batch of items to collect in one tour).

    Attributes:
        wave_id: Unique wave identifier.
        items: List of PickingItem tasks.
    """

    wave_id: int
    items: List[PickingItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[PickingItem]:
        return iter(self.items)

    @property
    def total_picks(self) -> int:
        """Total pick operations (rows) in the wave."""
        return len(self.items)

    @property
    def total_units(self) -> int:
        """Total item units to pick across all instructions."""
        return sum(item.quantity for item in self.items)

    @property
    def unique_skus(self) -> Set[str]:
        """Set of unique SKU references involved in this wave."""
        return {item.reference for item in self.items if item.reference}

    @property
    def unique_locations(self) -> List[str]:
        """
        List of unique storage location IDs visited in this wave,
        preserving first-encountered order.
        """
        seen: Set[str] = set()
        locations: List[str] = []
        for item in self.items:
            if item.location and item.location not in seen:
                seen.add(item.location)
                locations.append(item.location)
        return locations

    @property
    def locations(self) -> List[str]:
        """Alias for unique_locations."""
        return self.unique_locations

    @property
    def skus(self) -> Set[str]:
        """Alias for unique_skus."""
        return self.unique_skus

    def add_item(self, item: PickingItem) -> None:
        """Append a picking instruction to this wave."""
        self.items.append(item)

    @classmethod
    def from_records(cls, wave_id: int, records: Sequence[PickingWaveRecord]) -> "PickingWave":
        """Instantiate wave from a sequence of PickingWaveRecord DTOs."""
        items = [
            PickingItem(
                reference=r.reference,
                quantity=r.quantity_to_pick,
                location=r.location,
                size_us=r.size_us,
                operator=r.operator,
            )
            for r in records
        ]
        return cls(wave_id=int(wave_id), items=items)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, wave_id: int) -> "PickingWave":
        """Build PickingWave from a filtered DataFrame for a specific waveNumber."""
        items: List[PickingItem] = []
        for row in df.itertuples(index=False):
            ref = str(getattr(row, "reference", getattr(row, "Reference", "")))
            qty = int(getattr(row, "quantityToPick (units)", getattr(row, "quantity", 1)))
            loc = str(getattr(row, "locations", getattr(row, "location", "")))
            size_val = getattr(row, "Size (US)", getattr(row, "size_us", None))
            size = float(size_val) if pd.notna(size_val) else None
            op = str(getattr(row, "operator", ""))

            items.append(
                PickingItem(
                    reference=ref,
                    quantity=qty,
                    location=loc,
                    size_us=size,
                    operator=op,
                )
            )
        return cls(wave_id=int(wave_id), items=items)


@dataclass
class PickingWaveDataset:
    """
    Repository of all picking waves in the experiment.

    Supports indexing by wave_id and generating synthetic waves for Mode A.
    """

    waves: Dict[int, PickingWave] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.waves)

    def __getitem__(self, wave_id: int) -> PickingWave:
        key = int(wave_id)
        if key not in self.waves:
            raise KeyError(f"Wave ID {key} not found in dataset")
        return self.waves[key]

    def __iter__(self) -> Iterator[PickingWave]:
        return iter(self.waves.values())

    def get(self, wave_id: int, default: Optional[PickingWave] = None) -> Optional[PickingWave]:
        """Safely retrieve a wave by ID."""
        return self.waves.get(int(wave_id), default)

    def get_wave(self, wave_id: int, default: Optional[PickingWave] = None) -> Optional[PickingWave]:
        """Alias for get(wave_id)."""
        return self.get(wave_id, default)

    @property
    def total_waves(self) -> int:
        """Total number of waves."""
        return len(self.waves)

    @property
    def total_items(self) -> int:
        """Total pick items across all waves."""
        return sum(len(w.items) for w in self.waves.values())

    @property
    def wave_ids(self) -> List[int]:
        """List of all available wave IDs."""
        return list(self.waves.keys())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "PickingWaveDataset":
        """Group full Picking_Wave DataFrame by waveNumber into a dataset."""
        wave_col = "waveNumber" if "waveNumber" in df.columns else "wave_number"
        dataset = cls()
        for w_id, group in df.groupby(wave_col):
            wave = PickingWave.from_dataframe(group, wave_id=int(w_id))
            dataset.waves[int(w_id)] = wave
        return dataset

    @classmethod
    def from_csv(cls, file_path: Optional[str] = None) -> "PickingWaveDataset":
        """Load full picking waves dataset from CSV via unified loader."""
        from slap_gro.data.loaders import load_picking_waves

        df = load_picking_waves(file_path=file_path) if file_path else load_picking_waves()
        return cls.from_dataframe(df)

    @classmethod
    def generate_synthetic(
        cls,
        catalog: ProductCatalog,
        storage_policy: StoragePolicy,
        n_waves: int = 50,
        items_per_wave: int = 25,
        seed: Optional[int] = 42,
    ) -> "PickingWaveDataset":
        """
        Generate synthetic waves for Mode A (Paper Replication).

        Paper setup:
            - 50 waves
            - 25 SKU picks per wave
            - SKUs sampled based on Pareto ABC turnover probabilities:
                * Class A (20% SKU) generates ~80% demand probability
                * Class B (30% SKU) generates ~15% demand probability
                * Class C (50% SKU) generates ~5% demand probability
        """
        rng = np.random.default_rng(seed)
        skus_a = [p.reference for p in catalog.get_by_abc_class("A")] or catalog.references
        skus_b = [p.reference for p in catalog.get_by_abc_class("B")] or catalog.references
        skus_c = [p.reference for p in catalog.get_by_abc_class("C")] or catalog.references

        all_skus = skus_a + skus_b + skus_c
        # Assign individual selection weights
        prob_a = (0.80 / len(skus_a)) if skus_a else 0.0
        prob_b = (0.15 / len(skus_b)) if skus_b else 0.0
        prob_c = (0.05 / len(skus_c)) if skus_c else 0.0

        weights = [prob_a] * len(skus_a) + [prob_b] * len(skus_b) + [prob_c] * len(skus_c)
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        dataset = cls()
        for w_id in range(1, n_waves + 1):
            chosen_skus = rng.choice(all_skus, size=items_per_wave, replace=False, p=norm_weights)
            items: List[PickingItem] = []
            for sku in chosen_skus:
                # Find stored location for SKU from the policy matrix
                locs = storage_policy.get_locations_for_sku(str(sku))
                loc_id = locs[0] if locs else "A-01-01"
                items.append(PickingItem(reference=str(sku), quantity=1, location=loc_id))

            dataset.waves[w_id] = PickingWave(wave_id=w_id, items=items)

        return dataset
