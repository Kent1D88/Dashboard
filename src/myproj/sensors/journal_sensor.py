from __future__ import annotations
import dagster as dg
import polars as pl

from dagster import RunsFilter, DagsterRunStatus
from myproj.assets.bronze_journal import URQUAL_PARTITIONS, _BATCH_SIZE


@dg.sensor(
    asset_selection=[
        dg.AssetKey(["bronze", "journal"]),
        dg.AssetKey(["silver", "stay_delays"]),
    ],
    minimum_interval_seconds=300,
    required_resource_keys={"store"},
)
def bronze_journal_sensor(context) -> dg.SensorResult:
    store = context.resources.store

    df = (
        store.read_parquet("JOURNAL_202510150301.parquet")
        .select(pl.col("NOREC").cast(pl.Int64))
    )

    max_rec = df.select(pl.col("NOREC").max()).item()

    if max_rec is None:
        return dg.SkipReason("No records found in JOURNAL parquet")

    partition_count = int(max_rec // _BATCH_SIZE) + 1
    context.log.info(f"MAX(NOREC)={max_rec} → {partition_count} partitions")

    last_seen = int(context.cursor) if context.cursor else 0

    # -----------------------------
    # 1️⃣ Nouvelles partitions
    # -----------------------------
    new_partitions = []
    if partition_count > last_seen:
        new_partitions = [
            str(i)
            for i in range(last_seen + 1, partition_count + 1)
        ]

    # -----------------------------
    # 2️⃣ Toujours rematérialiser les 2 dernières partitions
    # -----------------------------
    recent_partitions = [
        str(i)
        for i in range(max(1, partition_count - 1), partition_count + 1)
    ]

    target_partitions = sorted(set(new_partitions + recent_partitions))

    if not target_partitions:
        return dg.SkipReason("Nothing to materialize")

    # -----------------------------
    # 3️⃣ Garde anti-run concurrent
    # -----------------------------
    active_partitions = set()

    active_runs = context.instance.get_runs(
        RunsFilter(
            statuses=[
                DagsterRunStatus.STARTED,
                DagsterRunStatus.QUEUED,
            ]
        )
    )

    for run in active_runs:
        part = run.tags.get("dagster/partition")
        if part:
            active_partitions.add(part)

    run_reqs = []

    for p in target_partitions:
        if p in active_partitions:
            context.log.info(f"Partition {p} already running — skip")
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[
                    dg.AssetKey(["bronze", "journal"]),
                    dg.AssetKey(["silver", "stay_delays"]),
                ],
                run_key=f"bronze_silver_{p}",
            )
        )

    if not run_reqs:
        return dg.SkipReason("All target partitions already running")

    # Ajouter partitions dynamiques si nouvelles
    dynamic_requests = []
    if new_partitions:
        dynamic_requests.append(
            URQUAL_PARTITIONS.build_add_request(new_partitions)
        )

    context.update_cursor(str(partition_count))

    return dg.SensorResult(
        dynamic_partitions_requests=dynamic_requests,
        run_requests=run_reqs,
    )