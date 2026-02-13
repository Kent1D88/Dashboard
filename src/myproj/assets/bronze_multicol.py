import dagster as dg
from dagster import MonthlyPartitionsDefinition
import polars as pl
import datetime as dt

from myproj.utils.debug import log_df, log_msg, debug_enabled

# ============================================================
# Monthly partitions (YYYY-MM)
# ============================================================
MULTICOL_PARTITIONS = MonthlyPartitionsDefinition(
    start_date="2010-01-01",
    timezone="Europe/Paris",
)

# ============================================================
# Bronze MULTICOL (DEV – parquet only)
# ============================================================
@dg.asset(
    key_prefix=["bronze"],
    name="multicol",
    partitions_def=MULTICOL_PARTITIONS,
    io_manager_key="minio",
    description="Bronze – MULTICOL (dev parquet)",
    required_resource_keys={"store"},
)
def bronze_multicol(context) -> pl.DataFrame:
    """
    Bronze extraction of MULTICOL table.
    Dev mode: reads local parquet and filters by month partition.
    """

    partition_key = context.partition_key  # ex: "2025-10"
    year, month = map(int, partition_key.split("-"))

    start = dt.datetime(year, month, 1)

    if month == 12:
        end = dt.datetime(year + 1, 1, 1)
    else:
        end = dt.datetime(year, month + 1, 1)

    context.log.info(f"[BRONZE MULTICOL] Loading partition {partition_key}")

    store = context.resources.store

    # Lecture brute parquet
    df = store.read_parquet("MULTICOL_202510150301.parquet")

    # Assure que ENTREE_DATE est bien datetime
    if df.schema.get("ENTREE_DATE") != pl.Datetime:
        df = df.with_columns(
            pl.col("ENTREE_DATE")
            .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False)
        )

    # Filtrage partition mensuelle
    df = df.filter(
        (pl.col("ENTREE_DATE") >= start)
        & (pl.col("ENTREE_DATE") < end)
    )

    context.log.info(f"[BRONZE MULTICOL] Rows for {partition_key}: {df.height}")

    log_df(context, df, step="bronze", name="multicol", keys=["IPP", "IEP"])

    return df