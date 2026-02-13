from __future__ import annotations
import datetime as dt
import dagster as dg
import polars as pl

import myproj.utils.dashboard_functions as U
from myproj.utils.debug import log_df, log_msg

from dagster import MonthlyPartitionsDefinition

# Monthly partitions starting from historical beginning
GOLD_MONTHLY_PARTITIONS = MonthlyPartitionsDefinition(start_date="2015-01-01")

PERIOD_ENTRIES = "hour"
PERIOD_DELAYS = "hour"
PERIOD_QUALITY = "day"
PERIOD_EDOR = "hour"

SILVER_TABLE = "silver_stay_delays"

# Partition-aware loader for silver
def _load_silver(clickhouse_client, partition_key: str) -> pl.LazyFrame:
    key = partition_key
    if len(key) == 7:  # "YYYY-MM"
        key = key + "-01"

    start_date = dt.date.fromisoformat(key)
    start_dt = dt.datetime.combine(start_date, dt.time.min)

    if start_date.month == 12:
        end_dt = dt.datetime(start_date.year + 1, 1, 1)
    else:
        end_dt = dt.datetime(start_date.year, start_date.month + 1, 1)

    sql = f"""
        SELECT
            SITE_UF, IST,
            dt_DATERDV, dt_HPIOA, dt_HPMED, dt_HPECINF, dt_HDECIS, dt_DHSORTIESAU,
            is_finish, has_dp, has_ccmu, mode_sortie_raw, mode_sortie, is_hospit,
            d_entree_ioa, d_entree_med, d_entree_orient, d_entree_sortie, d_orient_sortie,
            d_decision_hospit_sortie, d_decision_rad_sortie,
            ioa_lt15, med_lt60
        FROM {SILVER_TABLE}
        WHERE dt_DATERDV >= toDateTime('{start_dt:%Y-%m-%d %H:%M:%S}')
          AND dt_DATERDV <  toDateTime('{end_dt:%Y-%m-%d %H:%M:%S}')
    """
    return pl.from_arrow(clickhouse_client.query_arrow(sql)).lazy()


# --------------------------------------------------------
# ENTRIES
# --------------------------------------------------------

@dg.asset(
    key_prefix=["gold"],
    name="entries",
    io_manager_key="clickhouse_gold_entries",
    required_resource_keys={"clickhouse_client"},
    partitions_def=GOLD_MONTHLY_PARTITIONS,
    deps=[dg.AssetDep(SILVER_TABLE, partition_mapping=dg.AllPartitionMapping())],
)
def gold_entries(context) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client
    
    partition_key = context.partition_key
    lf = _load_silver(clickhouse_client, partition_key)
    
    query = U.DashboardQuery(period=PERIOD_ENTRIES, sites=None, date_start=None, date_end=None)
    log_msg(context, step="gold", msg="monthly_refresh", partition=partition_key, period=query.period)
    
    df = U.entry_volumes_by_site_period(lf_stay=lf, query=query).collect()

    df = df.with_columns(pl.lit(dt.datetime.utcnow()).alias("computed_at"))

    log_df(context, df, step="gold", name="gold_entries_out", keys=["SITE_UF"])
    context.log.info(f"df rows = {df.height}")

    return df


# --------------------------------------------------------
# DELAYS
# --------------------------------------------------------

@dg.asset(
    key_prefix=["gold"],
    name="delays",
    io_manager_key="clickhouse_gold_delays",
    required_resource_keys={"clickhouse_client"},
    partitions_def=GOLD_MONTHLY_PARTITIONS,
    deps=[dg.AssetDep(SILVER_TABLE, partition_mapping=dg.AllPartitionMapping())],
    )
def gold_delays(context) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    partition_key = context.partition_key
    lf = _load_silver(clickhouse_client, partition_key)
    query = U.DashboardQuery(period=PERIOD_DELAYS, sites=None, date_start=None, date_end=None)
    log_msg(context, step="gold", msg="monthly_refresh", partition=partition_key, period=query.period)
    
    df = U.delay_stats_by_site_period(lf_stay_with_delays=lf, query=query).collect()

    df = df.with_columns(pl.lit(dt.datetime.utcnow()).alias("computed_at"))

    log_df(context, df, step="gold", name="gold_delays_out", keys=["SITE_UF"])
    context.log.info(f"df rows = {df.height}")

    return df


# --------------------------------------------------------
# QUALITY
# --------------------------------------------------------

@dg.asset(
    key_prefix=["gold"],
    name="quality",
    io_manager_key="clickhouse_gold_quality",
    required_resource_keys={"clickhouse_client"},
    partitions_def=GOLD_MONTHLY_PARTITIONS,
    deps=[dg.AssetDep(SILVER_TABLE, partition_mapping=dg.AllPartitionMapping())],
)
def gold_quality(context) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    partition_key = context.partition_key
    lf = _load_silver(clickhouse_client, partition_key)
    query = U.DashboardQuery(period=PERIOD_QUALITY, sites=None, date_start=None, date_end=None)
    log_msg(context, step="gold", msg="monthly_refresh", partition=partition_key, period=query.period)
    
    df = U.completeness_stats_by_site_period(lf_stay=lf, query=query).collect()

    df = df.with_columns(pl.lit(dt.datetime.utcnow()).alias("computed_at"))

    log_df(context, df, step="gold", name="gold_quality_out", keys=["SITE_UF"])
    context.log.info(f"df rows = {df.height}")

    return df


# --------------------------------------------------------
# EDOR
# --------------------------------------------------------

@dg.asset(
    key_prefix=["gold"],
    name="edor_hourly",
    io_manager_key="clickhouse_gold_edor",
    required_resource_keys={"clickhouse_client"},
    ins={"mappingbox": dg.AssetIn(key=dg.AssetKey(["dims", "mappingbox"]))},
    partitions_def=GOLD_MONTHLY_PARTITIONS,
    deps=[dg.AssetDep(SILVER_TABLE, partition_mapping=dg.AllPartitionMapping())],
)
def gold_edor_hourly(context, mappingbox: pl.DataFrame) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    partition_key = context.partition_key
    lf = _load_silver(clickhouse_client, partition_key)
    query = U.DashboardQuery(period=PERIOD_EDOR, sites=None, date_start=None, date_end=None)
    log_msg(context, step="gold", msg="monthly_refresh", partition=partition_key, period=query.period)

    lf_presence = U.build_hourly_presence(lf_stay=lf, query=query)
    lf_capacity = U.build_capacity_per_site_hour(
        lf_mapping_box=mappingbox.lazy(),
        categories=["BOX_CS", "SAUV"],
    )

    df = U.compute_edor(
        lf_presence=lf_presence,
        lf_capacity_per_site_hour=lf_capacity,
    ).collect()

    df = df.with_columns(pl.lit(dt.datetime.utcnow()).alias("computed_at"))

    log_df(context, df, step="gold", name="gold_edor_hourly_out", keys=["SITE_UF"])
    context.log.info(f"df rows = {df.height}")

    return df