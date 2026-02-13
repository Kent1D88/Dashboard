from __future__ import annotations

import datetime as dt
import dagster as dg
import polars as pl
from dagster import RunsFilter, DagsterRunStatus

from myproj.assets.bronze_journal import URQUAL_PARTITIONS, _BATCH_SIZE


_JOURNAL_PARQUET = "JOURNAL_202510150301.parquet"
_DYNAMIC_PARTITIONS_NAME = "urqual_journal"
_MIN_INTERVAL_SECONDS = 300  # 5 minutes


_GOLD_ASSETS: list[dg.AssetKey] = [
    dg.AssetKey(["gold", "entries"]),
    dg.AssetKey(["gold", "delays"]),
    dg.AssetKey(["gold", "quality"]),
    dg.AssetKey(["gold", "edor_hourly"]),
]

def _get_partition_count_from_journal(store) -> int | None:
    df = store.read_parquet(_JOURNAL_PARQUET).select(pl.col("NOREC").cast(pl.Int64))
    max_rec = df.select(pl.col("NOREC").max()).item()
    if max_rec is None:
        return None
    return int(max_rec // _BATCH_SIZE) + 1


def _get_active_partitions(context: dg.SensorEvaluationContext) -> set[str]:
    active = set()
    runs = context.instance.get_runs(
        RunsFilter(statuses=[DagsterRunStatus.STARTED, DagsterRunStatus.QUEUED])
    )
    for run in runs:
        part = run.tags.get("dagster/partition")
        if part:
            active.add(part)
    return active


def _last_two_partitions(partition_count: int) -> list[str]:
    if partition_count <= 0:
        return []
    if partition_count == 1:
        return ["1"]
    return [str(partition_count - 1), str(partition_count)]

def _last_two_months(now: dt.datetime | None = None) -> list[str]:
    """Return partition keys for previous month and current month as 'YYYY-MM'."""
    if now is None:
        now = dt.datetime.now()

    cur = now.replace(day=1)
    # previous month: go to day 1 then step back 1 day
    prev = (cur - dt.timedelta(days=1)).replace(day=1)

    return [prev.strftime("%Y-%m"), cur.strftime("%Y-%m")]

# ============================================================
# 🟫 BRONZE SENSOR
# - new partition? materialize latest + previous
# - no new?        materialize latest + previous
# - also add dynamic partitions if missing
# ============================================================

@dg.sensor(
    asset_selection=[dg.AssetKey(["bronze", "journal"])],
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,
    required_resource_keys={"store"},
)
def bronze_sensor(context) -> dg.SensorResult:
    store = context.resources.store

    partition_count = _get_partition_count_from_journal(store)
    if partition_count is None:
        return dg.SkipReason("No records found in JOURNAL parquet")

    context.log.info(f"[BRONZE] MAX(NOREC) → partition_count={partition_count}")

    last_seen = int(context.cursor) if context.cursor else 0
    has_new = partition_count > last_seen

    # ✅ STOP rule: toujours matérialiser latest + previous
    target_partitions = _last_two_partitions(partition_count)

    # Ajouter partitions dynamiques si de nouvelles partitions sont apparues
    dynamic_requests = []
    if has_new:
        new_parts = [str(i) for i in range(last_seen + 1, partition_count + 1)]
        context.log.info(f"[BRONZE] New partitions detected: {new_parts}")
        dynamic_requests.append(URQUAL_PARTITIONS.build_add_request(new_parts))
    else:
        context.log.info("[BRONZE] No new partition detected")

    # garde anti-run concurrent
    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            context.log.info(f"[BRONZE] Partition {p} already running — skip")
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[dg.AssetKey(["bronze", "journal"])],
                # ⚠️ pas de run_key : on veut rejouer à chaque tick
            )
        )

    if not run_reqs and not dynamic_requests:
        return dg.SkipReason("[BRONZE] Nothing to do (latest/previous already running)")

    # Cursor = dernière partition_count vue
    context.update_cursor(str(partition_count))

    context.log.info(f"[BRONZE] Refresh partitions: {target_partitions}")
    return dg.SensorResult(
        dynamic_partitions_requests=dynamic_requests,
        run_requests=run_reqs,
    )


# ============================================================
# 🟪 SILVER SENSOR
# - toujours matérialiser latest + previous
# - basé sur la liste des partitions dynamiques existantes
# ============================================================

@dg.sensor(
    asset_selection=[dg.AssetKey(["silver", "stay_delays"])],
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,
)
def silver_sensor(context) -> dg.SensorResult:
    partitions = context.instance.get_dynamic_partitions(_DYNAMIC_PARTITIONS_NAME)
    if not partitions:
        return dg.SkipReason("[SILVER] No bronze partitions yet")

    parts_sorted = sorted(partitions, key=int)
    target_partitions = parts_sorted[-2:] if len(parts_sorted) >= 2 else parts_sorted

    context.log.info(f"[SILVER] Refresh partitions: {target_partitions}")

    # garde anti-run concurrent
    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            context.log.info(f"[SILVER] Partition {p} already running — skip")
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[dg.AssetKey(["silver", "stay_delays"])],
                # ⚠️ pas de run_key : on veut rejouer à chaque tick
            )
        )

    if not run_reqs:
        return dg.SkipReason("[SILVER] Nothing to refresh (latest/previous already running)")

    return dg.SensorResult(run_requests=run_reqs)

# ============================================================
# 🟨 GOLD SENSOR
# - toujours matérialiser latest + previous
# - basé sur la liste des partitions dynamiques existantes
# ============================================================

@dg.sensor(
    asset_selection=dg.AssetSelection.keys(*_GOLD_ASSETS),
    minimum_interval_seconds=900,  # 15 minutes
)
def gold_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    """Refresh GOLD monthly partitions for current + previous month.

    Assumes GOLD assets are partitioned by month with partition keys formatted as 'YYYY-MM'.
    """
    target_partitions = _last_two_months()
    context.log.info(f"[GOLD] Refresh monthly partitions: {target_partitions}")

    # garde anti-run concurrent
    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            context.log.info(f"[GOLD] Partition {p} already running — skip")
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=_GOLD_ASSETS,
                # ⚠️ pas de run_key : on veut rejouer à chaque tick
            )
        )

    if not run_reqs:
        return dg.SkipReason("[GOLD] Nothing to refresh (current/previous already running)")

    return dg.SensorResult(run_requests=run_reqs)
