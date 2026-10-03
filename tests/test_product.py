"""Tests for SLAP-GRO Phase 5: Product Layer."""

import pytest
import pandas as pd
from dataclasses import FrozenInstanceError

from slap_gro.data.product import Product, ProductCatalog, load_product_catalog
from slap_gro.data.loaders import load_products
from slap_gro.data.schemas import ProductRecord


# ==============================================================================
# 1. Product Entity Model Unit Tests
# ==============================================================================

def test_product_initialization_and_normalization():
    """Verify Product entity normalizes attributes to uppercase and strips whitespace."""
    p = Product(reference="  8551FLX  ", abc_class="  a ", sector=" pf ")
    assert p.reference == "8551FLX"
    assert p.abc_class == "A"
    assert p.sector == "PF"
    assert p.is_class_a is True
    assert p.is_class_b is False
    assert p.is_class_c is False


def test_product_immutability():
    """Verify Product is immutable (frozen dataclass)."""
    p = Product(reference="8551FLX", abc_class="A", sector="PF")
    with pytest.raises(FrozenInstanceError):
        p.reference = "ANOTHER_SKU"  # type: ignore


def test_product_hashability():
    """Verify Product instances can be placed in sets or used as dict keys."""
    p1 = Product(reference="SKU_001", abc_class="A", sector="PF")
    p2 = Product(reference="SKU_001", abc_class="A", sector="PF")
    p3 = Product(reference="SKU_002", abc_class="B", sector="PF")

    prod_set = {p1, p2, p3}
    assert len(prod_set) == 2
    assert p1 in prod_set

    prod_dict = {p1: 100, p3: 200}
    assert prod_dict[p2] == 100


def test_product_from_row_and_record():
    """Verify Product factory methods from raw dictionary, Series, and ProductRecord."""
    # From dict
    p_dict = Product.from_row({"Reference": "SKU_D", "ABCCOD": "b", "Sector": "pf"})
    assert p_dict.reference == "SKU_D"
    assert p_dict.abc_class == "B"
    assert p_dict.sector == "PF"

    # From Series
    series = pd.Series({"Reference": "SKU_S", "ABCCOD": "C", "Sector": "PF"})
    p_series = Product.from_row(series)
    assert p_series.reference == "SKU_S"
    assert p_series.is_class_c is True

    # From ProductRecord DTO
    dto = ProductRecord(reference="SKU_DTO", abc_class="A", sector="PF")
    p_dto = Product.from_record(dto)
    assert p_dto.reference == "SKU_DTO"
    assert p_dto.is_class_a is True


# ==============================================================================
# 2. ProductCatalog Repository Tests
# ==============================================================================

def test_product_catalog_from_csv():
    """Verify ProductCatalog loads 208 products from real Product.csv."""
    catalog = ProductCatalog.from_csv()
    assert len(catalog) == 208
    sample_sku = catalog.references[0]
    assert sample_sku in catalog
    assert "UNKNOWN_SKU" not in catalog

    prod = catalog[sample_sku]
    assert isinstance(prod, Product)
    assert prod.reference == sample_sku


    # O(1) get with default
    assert catalog.get("NON_EXISTENT", None) is None


def test_product_catalog_abc_distribution():
    """Verify catalog correctly summarizes ABC distribution."""
    catalog = ProductCatalog.from_csv()
    dist = catalog.abc_distribution
    assert set(dist.keys()) == {"A", "B", "C"}
    assert sum(dist.values()) == 208
    assert dist["A"] > 0
    assert dist["B"] > 0
    assert dist["C"] > 0


def test_product_catalog_filtering():
    """Verify filtering catalog by ABC class and sector."""
    catalog = ProductCatalog.from_csv()
    class_a_items = catalog.get_by_abc_class("A")
    assert len(class_a_items) == catalog.abc_distribution["A"]
    assert all(p.is_class_a for p in class_a_items)

    pf_items = catalog.get_by_sector("PF")
    assert len(pf_items) == 208


def test_product_catalog_to_dataframe():
    """Verify exporting catalog back to a clean pandas DataFrame."""
    catalog = ProductCatalog.from_csv()
    df = catalog.to_dataframe()
    assert len(df) == 208
    assert set(df.columns) == {"reference", "abc_class", "sector"}
    assert df["reference"].nunique() == 208


# ==============================================================================
# 3. Synthetic Catalog (Mode A - Paper Replication) Tests
# ==============================================================================

def test_synthetic_product_catalog_generation():
    """Verify Mode A synthetic catalog generates 2,000 products with Pareto skew."""
    catalog = ProductCatalog.generate_synthetic(
        n_products=2000,
        abc_skew=(0.20, 0.30, 0.50),
        seed=42,
    )
    assert len(catalog) == 2000
    assert "SKU_0001" in catalog
    assert "SKU_2000" in catalog

    dist = catalog.abc_distribution
    assert dist["A"] == 400
    assert dist["B"] == 600
    assert dist["C"] == 1000
    assert sum(dist.values()) == 2000


def test_load_product_catalog_factory():
    """Verify load_product_catalog factory returns correct mode catalog."""
    real_cat = load_product_catalog(mode="real")
    assert len(real_cat) == 208

    paper_cat = load_product_catalog(mode="paper")
    assert len(paper_cat) == 2000


def test_load_products_as_catalog_flag():
    """Verify load_products loader supports as_catalog=True."""
    catalog = load_products(as_catalog=True)
    assert isinstance(catalog, ProductCatalog)
    assert len(catalog) == 208
