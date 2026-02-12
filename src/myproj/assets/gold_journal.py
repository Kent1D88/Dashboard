from __future__ import annotations
import datetime as dt
import dagster as dg
import polars as pl

import myproj.utils.dashboard_functions as U


GOLD_WINDOW_DAYS = 30
GOLD_PERIOD = "day"
SILVER_TABLE = "silver_stay_delays"


def _build_query(window_days: int, period: str) -> U.DashboardQuery:
    today = dt.date.today()
    return U.DashboardQuery(
        period=period,
        sites=None,
        date_start=today - dt.timedelta(days=window_days),
        date_end=today,
    )


def _load_silver(clickhouse_client, query: U.DashboardQuery) -> pl.LazyFrame:
    sql = f"""
        SELECT
            SITE_UF, IPPDATE_multicol,
            dt_DATERDV, dt_HPIOA, dt_HPMED, dt_HPECINF, dt_HDECIS, dt_DHSORTIESAU,
            is_finish, has_dp, has_ccmu, mode_sortie_raw, mode_sortie, is_hospit,
            d_entree_ioa, d_entree_med, d_entree_orient, d_entree_sortie, d_orient_sortie,
            d_decision_hospit_sortie, d_decision_rad_sortie,
            ioa_lt15, med_lt60
        FROM {SILVER_TABLE}
        WHERE dt_DATERDV >= toDateTime('{query.date_start} 00:00:00')
          AND dt_DATERDV <= toDateTime('{query.date_end} 23:59:59')
    """
    return pl.from_arrow(clickhouse_client.query_arrow(sql)).lazy()


@dg.asset(
    key_prefix=["gold"],
    name="entries",
    io_manager_key="clickhouse_gold_entries",
    required_resource_keys={"clickhouse_client"},
    deps=[dg.AssetKey(["silver", "stay_delays"])],
    description="GOLD — volumes d'entrée -> ClickHouse.gold_entries",
)
def gold_entries(context) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    query = _build_query(GOLD_WINDOW_DAYS, GOLD_PERIOD)
    lf = _load_silver(clickhouse_client, query)
    df = U.entry_volumes_by_site_period(lf, query).collect()
    return df.with_columns(pl.lit(dt.datetime.now()).alias("computed_at"))


@dg.asset(
    key_prefix=["gold"],
    name="delays",
    io_manager_key="clickhouse_gold_delays",
    required_resource_keys={"clickhouse_client"},
    deps=[dg.AssetKey(["silver", "stay_delays"])],
    description="GOLD — stats délais -> ClickHouse.gold_delays",
)
def gold_delays(context) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    query = _build_query(GOLD_WINDOW_DAYS, GOLD_PERIOD)
    lf = _load_silver(clickhouse_client, query)
    df = U.delay_stats_by_site_period(lf, query).collect()
    return df.with_columns(pl.lit(dt.datetime.now()).alias("computed_at"))


@dg.asset(
    key_prefix=["gold"],
    name="quality",
    io_manager_key="clickhouse_gold_quality",
    required_resource_keys={"clickhouse_client"},
    deps=[dg.AssetKey(["silver", "stay_delays"])],
    description="GOLD — qualité -> ClickHouse.gold_quality",
)
def gold_quality(context) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    query = _build_query(GOLD_WINDOW_DAYS, GOLD_PERIOD)
    lf = _load_silver(clickhouse_client, query)
    df = U.completeness_stats_by_site_period(lf, query).collect()
    return df.with_columns(pl.lit(dt.datetime.now()).alias("computed_at"))


@dg.asset(
    key_prefix=["gold"],
    name="edor_hourly",
    io_manager_key="clickhouse_gold_edor",
    required_resource_keys={"clickhouse_client"},
    ins={
            "mappingbox": dg.AssetIn(key=dg.AssetKey(["dims", "mappingbox"])),
        },    
    deps=[dg.AssetKey(["silver", "stay_delays"])],
    description="GOLD — EDOR horaire -> ClickHouse.gold_edor_hourly",
)
def gold_edor_hourly(context, mappingbox: pl.DataFrame) -> pl.DataFrame:
    clickhouse_client = context.resources.clickhouse_client

    query = _build_query(GOLD_WINDOW_DAYS, GOLD_PERIOD)
    lf = _load_silver(clickhouse_client, query)

    lf_presence = U.build_hourly_presence(lf, query=query)
    lf_capacity = U.build_capacity_per_site_hour(mappingbox.lazy(), categories=["BOX_CS", "SAUV"])

    df = U.compute_edor(lf_presence, lf_capacity).collect()
    return df.with_columns(pl.lit(dt.datetime.now()).alias("computed_at"))