from __future__ import annotations
import dagster as dg
import polars as pl

@dg.asset(key_prefix=["dims"], name="multicol", description="Dimension MULTICOL (parquet)")
def dim_multicol(context) -> pl.DataFrame:
    store = context.resources.store
    return store.read_parquet("MULTICOL_202510150301.parquet")


@dg.asset(key_prefix=["dims"], name="uf", description="Référentiel UF (parquet)")
def dim_uf(context) -> pl.DataFrame:
    store = context.resources.store
    df = store.read_parquet("liste_UF.parquet")
    # normalisation minimale (au cas où espaces/encodage)
    return df.with_columns([
        pl.col("UF").cast(pl.Utf8).str.strip_chars(),
        pl.col("SITE_UF").cast(pl.Utf8).str.strip_chars(),
        pl.col("TYPE_UF").cast(pl.Utf8).str.strip_chars(),
    ])


@dg.asset(key_prefix=["dims"], name="mappingbox", description="Capacités box (CSV)")
def dim_mappingbox(context) -> pl.DataFrame:
    store = context.resources.store
    df = store.read_parquet("MAPPINGBOX.parquet")
    # si besoin: trims sur colonnes string
    for c, t in df.schema.items():
        if t == pl.Utf8:
            df = df.with_columns(pl.col(c).str.strip_chars())
    return df