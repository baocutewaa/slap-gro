"""Product domain layer for SLAP-GRO.

Decouples core business logic, picking waves, storage assignment,
and GRO algorithms from raw CSV column names.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Product:
    """
    Domain entity representing an item SKU in the warehouse.

    Decouples domain logic from raw CSV headers ('Reference', 'ABCCOD', 'Sector').

    Attributes:
        reference: Unique SKU identifier (e.g., '8551FLX', 'PROD_0001').
        abc_class: Pareto ABC activity class ('A', 'B', or 'C').
        sector: Product storage zone / sector (e.g., 'PF').
    """

    reference: str
    abc_class: str
    sector: str = "PF"

    def __post_init__(self):
        # Enforce trimmed and uppercase values
        object.__setattr__(self, "reference", str(self.reference).strip())
        object.__setattr__(self, "abc_class", str(self.abc_class).strip().upper())
        object.__setattr__(self, "sector", str(self.sector).strip().upper())

    @property
    def is_class_a(self) -> bool:
        """Check if product belongs to high-velocity Class A."""
        return self.abc_class == "A"

    @property
    def is_class_b(self) -> bool:
        """Check if product belongs to medium-velocity Class B."""
        return self.abc_class == "B"

    @property
    def is_class_c(self) -> bool:
        """Check if product belongs to slow-moving Class C."""
        return self.abc_class == "C"

    @classmethod
    def from_row(cls, row: Any) -> "Product":
        """Create a Product from a pandas Series, namedtuple, or dict."""
        if isinstance(row, dict):
            ref = row.get("Reference") or row.get("reference") or ""
            abc = row.get("ABCCOD") or row.get("abc_class") or row.get("abc") or "C"
            sec = row.get("Sector") or row.get("sector") or "PF"
        else:
            ref = getattr(row, "Reference", getattr(row, "reference", ""))
            abc = getattr(row, "ABCCOD", getattr(row, "abc_class", getattr(row, "abc", "C")))
            sec = getattr(row, "Sector", getattr(row, "sector", "PF"))
        return cls(reference=str(ref), abc_class=str(abc), sector=str(sec))

    @classmethod
    def from_record(cls, record: Any) -> "Product":
        """Create a Product from a ProductRecord DTO."""
        return cls(
            reference=record.reference,
            abc_class=record.abc_class,
            sector=record.sector,
        )


@dataclass
class ProductCatalog:
    """
    Catalog repository managing the collection of products.

    Provides O(1) SKU lookups, ABC class querying, and synthetic catalog generation
    for Paper Replication (Mode A).
    """

    _products: Dict[str, Product] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._products)

    def __iter__(self) -> Iterator[Product]:
        return iter(self._products.values())

    def __contains__(self, reference: str) -> bool:
        return str(reference).strip() in self._products

    def __getitem__(self, reference: str) -> Product:
        key = str(reference).strip()
        if key not in self._products:
            raise KeyError(f"Product '{key}' not found in catalog")
        return self._products[key]

    def add(self, product: Product) -> None:
        """Add or update a product in the catalog."""
        self._products[product.reference] = product

    def get(self, reference: str, default: Optional[Product] = None) -> Optional[Product]:
        """Retrieve a product by SKU identifier with fallback default."""
        return self._products.get(str(reference).strip(), default)

    @property
    def references(self) -> List[str]:
        """List of all product SKU references in catalog."""
        return list(self._products.keys())

    @property
    def abc_distribution(self) -> Dict[str, int]:
        """Counts of products per ABC class ('A', 'B', 'C')."""
        counts = {"A": 0, "B": 0, "C": 0}
        for p in self._products.values():
            if p.abc_class in counts:
                counts[p.abc_class] += 1
            else:
                counts[p.abc_class] = 1
        return counts

    def get_by_abc_class(self, abc_class: str) -> List[Product]:
        """Filter products by their ABC class ('A', 'B', or 'C')."""
        target = abc_class.strip().upper()
        return [p for p in self._products.values() if p.abc_class == target]

    def get_by_sector(self, sector: str) -> List[Product]:
        """Filter products by their sector."""
        target = sector.strip().upper()
        return [p for p in self._products.values() if p.sector == target]

    def to_dataframe(self) -> pd.DataFrame:
        """Export catalog products to a pandas DataFrame."""
        return pd.DataFrame([
            {
                "reference": p.reference,
                "abc_class": p.abc_class,
                "sector": p.sector,
            }
            for p in self._products.values()
        ])

    @classmethod
    def from_products(cls, products: Sequence[Product]) -> "ProductCatalog":
        """Instantiate catalog from an iterable of Product entities."""
        cat = cls()
        for p in products:
            cat.add(p)
        return cat

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ProductCatalog":
        """Build catalog from a normalized or raw pandas DataFrame."""
        cat = cls()
        for row in df.itertuples(index=False):
            prod = Product.from_row(row)
            if prod.reference:
                cat.add(prod)
        return cat

    @classmethod
    def from_csv(cls, file_path: str = "data/raw/Product.csv") -> "ProductCatalog":
        """Load catalog directly from CSV file via unified loader."""
        from slap_gro.data.loaders import load_products

        df = load_products(file_path=file_path)
        return cls.from_dataframe(df)

    @classmethod
    def generate_synthetic(
        cls,
        n_products: int = 2000,
        abc_skew: Sequence[float] = (0.20, 0.30, 0.50),
        sector: str = "PF",
        seed: Optional[int] = 42,
    ) -> "ProductCatalog":
        """
        Generate synthetic products for Mode A (Paper Replication).

        Paper setup:
            - Total items: 2,000 SKUs (e.g. 'SKU_0001' to 'SKU_2000').
            - ABC distribution:
                * Class A: ~20% (400 items)
                * Class B: ~30% (600 items)
                * Class C: ~50% (1,000 items)
        """
        if sum(abc_skew) <= 0:
            raise ValueError("abc_skew probabilities must sum to > 0")

        # Normalize skew proportions
        total_skew = sum(abc_skew)
        probs = [p / total_skew for p in abc_skew]

        n_a = int(round(n_products * probs[0]))
        n_b = int(round(n_products * probs[1]))
        n_c = n_products - (n_a + n_b)

        classes = ["A"] * n_a + ["B"] * n_b + ["C"] * n_c

        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(classes)

        cat = cls()
        for i in range(1, n_products + 1):
            ref = f"SKU_{i:04d}"
            cat.add(Product(reference=ref, abc_class=classes[i - 1], sector=sector))
        return cat


def load_product_catalog(
    file_path: str = "data/raw/Product.csv",
    mode: str = "real",
    n_synthetic_products: int = 2000,
) -> ProductCatalog:
    """
    Factory function to load ProductCatalog based on execution mode.

    Args:
        file_path: Path to Product.csv (for 'real' mode).
        mode: 'real' (loads 208 catalog SKUs) or 'paper' (generates 2,000 synthetic SKUs).
        n_synthetic_products: Number of products if mode is 'paper'.

    Returns:
        ProductCatalog instance.
    """
    if mode.lower() in {"paper", "paper_replication", "synthetic"}:
        return ProductCatalog.generate_synthetic(n_products=n_synthetic_products)
    return ProductCatalog.from_csv(file_path=file_path)
