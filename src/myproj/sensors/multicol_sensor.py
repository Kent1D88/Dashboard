from __future__ import annotations

import datetime as dt
import dagster as dg
import polars as pl
from dagster import RunsFilter, DagsterRunStatus

from myproj.assets.bronze_journal import URQUAL_PARTITIONS, _BATCH_SIZE

# 🔧 Pilotage simple ici
REFRESH_N_MONTHS = 2   # ← changer 1 / 2 / 6 ici
_MIN_INTERVAL_SECONDS = 300

def _last_n_months(n: int, now: dt.datetime | None = None) -> list[str]:
    if now is None:
        now = dt.datetime.now()

    months = []
    current = now.replace(day=1)

    for _ in range(n):
        months.append(current.strftime("%Y-%m"))
        current = (current - dt.timedelta(days=1)).replace(day=1)

    return sorted(months)

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


@dg.sensor(
    asset_selection=[dg.AssetKey(["bronze", "multicol"])],
    minimum_interval_seconds=_MIN_INTERVAL_SECONDS,  # 15 min
    required_resource_keys={"store"},
)
def bronze_multicol_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    target_partitions = _last_n_months(REFRESH_N_MONTHS)

    context.log.info(f"[MULTICOL] Refresh partitions: {target_partitions}")

    active_partitions = _get_active_partitions(context)

    run_reqs: list[dg.RunRequest] = []
    for p in target_partitions:
        if p in active_partitions:
            context.log.info(f"[MULTICOL] Partition {p} already running — skip")
            continue

        run_reqs.append(
            dg.RunRequest(
                partition_key=p,
                asset_selection=[dg.AssetKey(["multicol"])],
            )
        )

    if not run_reqs:
        return dg.SkipReason("[MULTICOL] Nothing to refresh")

    return dg.SensorResult(run_requests=run_reqs)