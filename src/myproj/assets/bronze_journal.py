from __future__ import annotations
import dagster as dg
import polars as pl
import datetime as dt

from myproj.utils.journal_parse import parse_journal
from myproj.utils.debug import log_df, log_msg


_BATCH_SIZE = 1_000_000
URQUAL_PARTITIONS = dg.DynamicPartitionsDefinition(name="urqual_journal")

JOURNAL_MONTHLY = dg.MonthlyPartitionsDefinition(
    start_date="2010-01-01",
    timezone="Europe/Paris",
)

@dg.asset(
    key_prefix=["bronze"],
    name="journal_raw",
    partitions_def=URQUAL_PARTITIONS,
    io_manager_key="minio",
    required_resource_keys={"store"},
    description="BRONZE — journal.csv découpé en partitions NOREC, persisté sur MinIO",
)
def bronze_journal_raw(context) -> pl.DataFrame:
    store = context.resources.store

    part = int(context.partition_key)
    start = (part - 1) * _BATCH_SIZE + 1
    end = start + _BATCH_SIZE - 1

    df = (
        store.read_parquet("JOURNAL_202510150301.parquet")
        .with_columns([
            pl.col("NOREC").cast(pl.Int64),
            pl.col("LIBELLE").cast(pl.Utf8),
        ])
        .filter(pl.col("NOREC").is_between(start, end))
        .unique()
    )

    context.log.info(f"Partition {part} | NOREC [{start},{end}] | rows={df.height}")
    log_df(context, df, step="bronze", name="journal_partition", keys=["NOREC"])
    return df


@dg.asset(
    key_prefix=["bronze"],
    name="journal_monthly",
    partitions_def=JOURNAL_MONTHLY,
    io_manager_key="minio",
    required_resource_keys={"store"},
    description="BRONZE — journal.csv agrégé par mois, persisté sur MinIO",
    ins={
        "journal_raw": dg.AssetIn(key=dg.AssetKey(["bronze", "journal_raw"]))
    },
)
def bronze_journal_monthly(context, journal_raw: pl.DataFrame) -> pl.DataFrame:
    
    partition_key = context.partition_key  # "YYYY-MM"
    year, month = map(int, partition_key.split("-"))

    start = dt.datetime(year, month, 1)
    end = (
        dt.datetime(year + 1, 1, 1)
        if month == 12
        else dt.datetime(year, month + 1, 1)
    )

    # parse ici
    df = parse_journal(
        df_journal=journal_raw,
        drop_col=True,
        normalize_keys=True,
    )

    # filtrage mensuel sur dt
    df = df.filter(
        (pl.col("dt") >= start) &
        (pl.col("dt") < end)
    )

    return df