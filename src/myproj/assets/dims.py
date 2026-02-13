from __future__ import annotations
import dagster as dg
import polars as pl
from myproj.utils.debug import log_df, log_msg


@dg.asset(
    key_prefix=["dims"],
    name="uf",
    io_manager_key="minio",
    description="Référentiel UF (parquet)",
    required_resource_keys={"store"},
)
def dim_uf(context) -> pl.DataFrame:
    store = context.resources.store
    df = (
        store
        .read_parquet("liste_UF.parquet")
        .with_columns([
            pl.col("UF").cast(pl.Utf8).str.strip_chars(),
            pl.col("SITE_UF").cast(pl.Utf8).str.strip_chars(),
            pl.col("TYPE_UF").cast(pl.Utf8).str.strip_chars(),
        ])
    )
    log_df(context, df, step="dims", name="uf", keys=["UF"])
    return df


@dg.asset(
    key_prefix=["dims"],
    name="mappingbox",
    io_manager_key="minio",
    description="Capacités box (CSV)",
    required_resource_keys={"store"},
)
def dim_mappingbox(context) -> pl.DataFrame:
    store = context.resources.store

    # ⚠️ ici tu lis un parquet, mais ton fichier dans data est MAPPINGBOX.csv
    # donc soit tu changes en read_csv, soit tu convertis en parquet.
    df = store.read_parquet("MAPPINGBOX.parquet")

    for c, t in df.schema.items():
        if t == pl.Utf8:
            df = df.with_columns(pl.col(c).str.strip_chars())
    log_df(context, df, step="dims", name="mappingbox", keys=["SITE_UF"])
    return df