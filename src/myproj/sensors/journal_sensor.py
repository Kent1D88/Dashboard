from __future__ import annotations
import dagster as dg
import polars as pl

from myproj.assets.bronze_journal import URQUAL_PARTITIONS, _BATCH_SIZE

@dg.sensor(
    asset_selection=[dg.AssetKey(["bronze", "journal"])],
    minimum_interval_seconds=43200,
    required_resource_keys={"store"},
)
def bronze_journal_sensor(context) -> dg.SensorResult:
    store = context.resources.store

    df = (
        store.read_csv("JOURNAL_202510150301.csv")
        .select(pl.col("NOREC").cast(pl.Int64))
    )

    max_rec = df.select(pl.col("NOREC").max()).item()
    if max_rec is None:
        context.log.info("No records found.")
        return dg.SensorResult()

    partition_count = int(max_rec // _BATCH_SIZE) + 1
    partitions = [str(i) for i in range(1, partition_count + 1)]

    context.log.info(f"MAX(NOREC)={max_rec} → {len(partitions)} partitions")
    return dg.SensorResult(
        dynamic_partitions_requests=[URQUAL_PARTITIONS.build_add_request(partitions)]
    )