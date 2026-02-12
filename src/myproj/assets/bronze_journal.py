from __future__ import annotations
import dagster as dg
import polars as pl

from myproj.utils.debug import log_df, log_msg

_BATCH_SIZE = 1_000_000
URQUAL_PARTITIONS = dg.DynamicPartitionsDefinition(name="urqual_journal")

@dg.asset(
    key_prefix=["bronze"],
    name="journal",
    partitions_def=URQUAL_PARTITIONS,
    io_manager_key="minio",
    required_resource_keys={"store"},
    description="BRONZE — journal.csv découpé en partitions NOREC, persisté sur MinIO",
)
def bronze_journal(context) -> pl.DataFrame:
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