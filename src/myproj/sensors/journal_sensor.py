from __future__ import annotations

import datetime as dt
import dagster as dg
import polars as pl
from dagster import RunsFilter, DagsterRunStatus

from myproj.assets.bronze_journal import URQUAL_PARTITIONS, _BATCH_SIZE

# ============================================================
# 🔧 CONFIG
# ============================================================

REFRESH_N_MONTHS = 2   # ← changer 1 / 2 / 6 ici
_MIN_INTERVAL_SECONDS = 300
_JOURNAL_PARQUET = "JOURNAL_202510150301.parquet"
_DYNAMIC_PARTITIONS_NAME = "urqual_journal"

_GOLD_ASSETS: list[dg.AssetKey] = [
    dg.AssetKey(["gold", "entries"]),
    dg.AssetKey(["gold", "delays"]),
    dg.AssetKey(["gold", "quality"]),
    dg.AssetKey(["gold", "edor_hourly"]),
]

# ============================================================
# 🔧 UTILS
# ============================================================

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


def _last_n_months(n: int, now: dt.datetime | None = None) -> list[str]:
    if now is None:
        now = dt.datetime.now()

    months = []
    current = now.replace(day=1)

    for _ in range(n):
        months.append(current.strftime("%Y-%m"))
        current = (current - dt.timedelta(days=1)).replace(day=1)

    return sorted(months)

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

def _last_n_months(n: int) -> list[str]:
    now = dt.datetime.now().replace(day=1)
    months = []
    for _ in range(n):
        months.append(now.strftime("%Y-%m"))
        now = (now - dt.timedelta(days=1)).replace(day=1)
    return sorted(months)

# ============================================================
# 🟫 1️⃣ BRONZE RAW SENSOR (NOREC)
# ============================================================

@dg.sensor(
    asset_selection=[dg.AssetKey(["bronze", "journal"])],
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,
    required_resource_keys={"store"},
)
def bronze_journal_raw_sensor(context) -> dg.SensorResult:
    store = context.resources.store

    partition_count = _get_partition_count_from_journal(store)
    if partition_count is None:
        return dg.SkipReason("No records found in JOURNAL parquet")

    context.log.info(f"[BRONZE RAW] MAX(NOREC) → partition_count={partition_count}")

    last_seen = int(context.cursor) if context.cursor else 0
    has_new = partition_count > last_seen

    target_partitions = _last_two_partitions(partition_count)

    dynamic_requests = []
    if has_new:
        new_parts = [str(i) for i in range(last_seen + 1, partition_count + 1)]
        dynamic_requests.append(URQUAL_PARTITIONS.build_add_request(new_parts))

    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[dg.AssetKey(["bronze", "journal"])],
            )
        )

    context.update_cursor(str(partition_count))

    if not run_reqs and not dynamic_requests:
        return dg.SkipReason("[BRONZE RAW] Nothing to refresh")

    return dg.SensorResult(
        dynamic_partitions_requests=dynamic_requests,
        run_requests=run_reqs,
    )

# ============================================================
# 🟫 2️⃣ BRONZE MONTHLY SENSOR
# ============================================================

@dg.sensor(
    asset_selection=[dg.AssetKey(["bronze", "journal_monthly"])],
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,
)
def bronze_journal_monthly_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    target_partitions = _last_n_months(REFRESH_N_MONTHS)

    context.log.info(f"[BRONZE MONTHLY] Refresh: {target_partitions}")

    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[dg.AssetKey(["bronze", "journal_monthly"])],
            )
        )

    if not run_reqs:
        return dg.SkipReason("[BRONZE MONTHLY] Nothing to refresh")

    return dg.SensorResult(run_requests=run_reqs)

# ============================================================
# 🟪 3️⃣ SILVER SENSOR (MONTHLY)
# ============================================================

@dg.sensor(
    asset_selection=[dg.AssetKey(["silver", "stay_delays"])],
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,
)
def silver_journal_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    target_partitions = _last_n_months(REFRESH_N_MONTHS)

    context.log.info(f"[SILVER] Refresh: {target_partitions}")

    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[dg.AssetKey(["silver", "stay_delays"])],
            )
        )

    if not run_reqs:
        return dg.SkipReason("[SILVER] Nothing to refresh")

    return dg.SensorResult(run_requests=run_reqs)

# ============================================================
# 🟨 4️⃣ GOLD SENSOR (MONTHLY)
# ============================================================

@dg.sensor(
    asset_selection=dg.AssetSelection.keys(*_GOLD_ASSETS),
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,
)
def gold_journal_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    target_partitions = _last_n_months(REFRESH_N_MONTHS)

    context.log.info(f"[GOLD] Refresh: {target_partitions}")

    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=_GOLD_ASSETS,
            )
        )

    if not run_reqs:
        return dg.SkipReason("[GOLD] Nothing to refresh")

    return dg.SensorResult(run_requests=run_reqs)