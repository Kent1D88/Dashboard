"""dashboard_functions.py: Core pipeline and helpers for CDC dashboard metrics.

This module provides a set of functions for constructing and analyzing stay-level and hourly
metrics for emergency department dashboards, using Polars LazyFrame for scalable, efficient
processing on large datasets. The main pipeline is:
    build_stay_table -> add_delay_columns -> delay_stats_by_site_period / completeness_stats_by_site_period / build_hourly_presence -> compute_edor
    build_hourly_staff_counts -> compute_patient_staff_ratios

Families of metrics covered:
    - Entry volumes (day/week/month + hour-of-day)
    - Age and triage distributions
    - Waiting indicators (no IOA / no MED)
    - Box occupancy (occupied boxes vs capacity)
    - Hospitalization rate per period
    - Event timestamp parsing, delay computation, site/date filtering, and completeness statistics

Functions are compatible with LazyFrame and optimized for large-scale data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Literal, Sequence

import polars as pl
# =========================
# Staff config
# =========================

@dataclass(frozen=True)
class DashboardStaffConfig:
    """Configuration for inferring hourly staff presence from journal actions.

    Presence is inferred by at least one logged staff action in a given hour (approximation),
    unless a better signal is provided.

    Attributes
    ----------
    user_col : str, default "USER_ID"
        Column containing the staff user identifier.
    dt_staff_col : str, default "dt"
        Column with the timestamp of the staff action (already parsed as datetime).
    active_action_codes : Optional[Sequence[str]], default None
        If provided, restrict staff presence inference to these action codes (e.g., only count
        actions that are relevant for presence, such as direct patient care).
    count_distinct_per_hour : bool, default True
        If True, count unique users per site/date/hour/role; if False, count all actions (not recommended).
    """
    user_col: str = "USER_ID"
    dt_staff_col: str = "dt"  # timestamp of the journal action attributed to staff
    active_action_codes: Optional[Sequence[str]] = None  # if provided, restrict staff presence inference to these action codes
    count_distinct_per_hour: bool = True  # count unique users per hour


# =========================
# Config
# =========================
Period = Literal["hour", "day", "week", "month"]

_TRUNC_MAP: dict[Period, str] = {"hour": "1h", "day": "1d", "week": "1w", "month": "1mo"}

@dataclass(frozen=True)
class DashboardQuery:
    """Query parameters for dashboard metrics.

    Attributes
    ----------
    period : {'day', 'week', 'month'}, default 'day'
        Granularity for period aggregation (used for truncation).
    sites : Sequence[str] or None, optional
        If provided, restricts results to these site codes; None means all sites.
    date_start : pl.Datetime or None, optional
        Inclusive lower bound for the anchor datetime column (typically 'dt_DATERDV').
    date_end : pl.Datetime or None, optional
        Exclusive upper bound for the anchor datetime column.

    Properties
    ----------
    trunc : str
        Polars-style truncation string corresponding to the selected period.
    """
    period: Period = "day"
    sites: Optional[Sequence[str]] = None
    date_start: Optional[pl.Datetime] = None
    date_end: Optional[pl.Datetime] = None

    @property
    def trunc(self) -> str:
        """Return the truncation string for the selected period."""
        return _TRUNC_MAP[self.period]


@dataclass(frozen=True)
class DashboardSchema:
    """Schema configuration for parsed JOURNAL data.

    Attributes
    ----------
    stay_col : str
        Column name for unique stay identifier (after parse_journal, typically 'IPPDATE_multicol').
    site_col : str
        Column name for site/unit identifier (typically 'SITE_UF').
    action_code_col : str
        Column name for action/event code (typically 'ACTION_CODE').
    action_detail_col : str
        Column name for event detail (often stores timestamps as strings, typically 'ACTION_DETAIL').
    dt_col : str
        Column name for already-parsed datetime (used for events like EVT_LOC).
    dt_formats : tuple of str
        Supported datetime formats for parsing ACTION_DETAIL.
    """
    stay_col: str = "IPPDATE_multicol"
    site_col: str = "SITE_UF"
    action_code_col: str = "ACTION_CODE"
    action_detail_col: str = "ACTION_DETAIL"
    dt_col: str = "dt"
    dt_formats: tuple[str, str] = ("%d/%m/%Y %H:%M", "%Y%m%d%H%M%S")


@dataclass(frozen=True)
class DashboardEventCodes:
    """Action codes for key events and flags in the dashboard pipeline.

    Attributes
    ----------
    enter : str
        Code for patient entry (timestamp from ACTION_DETAIL).
    ioa : str
        Code for IOA time (timestamp from ACTION_DETAIL).
    med : str
        Code for physician time (timestamp from ACTION_DETAIL).
    inf : str
        Code for nurse time (timestamp from ACTION_DETAIL).
    decision : str
        Code for orientation/decision (timestamp from ACTION_DETAIL).
    exit : str
        Code for exit/discharge (timestamp from ACTION_DETAIL).
    diagp : str
        Code indicating presence of primary diagnosis (flag, not a timestamp).
    ccmu : str
        Code indicating presence of CCMU (flag, not a timestamp).
    mode_sortie_codes : tuple of str
        Codes for disposition/orientation (values in ACTION_DETAIL, used to determine 'is_hospit').
    loc : str
        Code for patient location (timestamp from dt_col, not ACTION_DETAIL).

    Notes
    -----
    Timestamps for enter/ioa/med/inf/decision/exit are parsed from ACTION_DETAIL using dt_formats.
    Location ('loc') timestamps use the dt_col (already parsed).
    """
    enter: str = "DATERDV"
    ioa: str = "HPECIAO"
    med: str = "HPECMED"
    inf: str = "HPECINF"
    decision: str = "HEURDECIS1"
    exit: str = "DHSORTIESAU"
    diagp: str = "DIAGP"
    ccmu: str = "CCMU"
    mode_sortie_codes: tuple[str, str] = ("MODEDESORTIE", "REALSORTIE")
    loc: str = "EVT_LOC"


# =========================
# Helpers (expr builders)
# =========================

def _expr_parse_event_dt(
    detail_col: str,
    formats: tuple[str, str],
    out: str = "event_dt",
) -> pl.Expr:
    """Parse event timestamp from ACTION_DETAIL using two supported formats.

    Supported formats are tried in order and the first successful parse is used.
    If strict=False, parsing will not error on invalid formats and will yield null.

    Parameters
    ----------
    detail_col : str
        Column name containing the timestamp string (typically ACTION_DETAIL).
    formats : tuple of str
        Tuple of two datetime formats to try (e.g., ("%d/%m/%Y %H:%M", "%Y%m%d%H%M%S")).
    out : str, default "event_dt"
        Name for the output column/expression.

    Returns
    -------
    pl.Expr
        Expression producing the parsed datetime, or null if parsing fails.
    """
    f1, f2 = formats
    return (
        pl.coalesce([
            pl.col(detail_col).str.strptime(pl.Datetime, f1, strict=False),
            pl.col(detail_col).str.strptime(pl.Datetime, f2, strict=False),
        ])
        .alias(out)
    )


def _expr_any_code(action_code_col: str, code: str, out: str) -> pl.Expr:
    """Return True if at least one row for this code exists within the group.

    Used in group aggregations to flag presence of specific event codes.
    """
    return (pl.col(action_code_col) == code).any().alias(out)


def _expr_min_dt_for_code(event_dt_col: str, action_code_col: str, code: str, out: str) -> pl.Expr:
    """Compute the minimum event_dt for a given ACTION_CODE within a group.

    Used to extract the first occurrence timestamp for an event type in group-by context.
    """
    return (
        pl.col(event_dt_col)
        .filter(pl.col(action_code_col) == code)
        .min()
        .alias(out)
    )

def apply_site_filter(
    lf: pl.LazyFrame,
    query: DashboardQuery,
    *,
    site_col: str = "SITE_UF",
) -> pl.LazyFrame:
    """Filter LazyFrame by sites if query.sites is provided.

    If query.sites is None, returns the LazyFrame unchanged (all sites included).
    """
    if query.sites is None:
        return lf
    return lf.filter(pl.col(site_col).is_in(list(query.sites)))


def apply_date_window(
    lf: pl.LazyFrame,
    query: DashboardQuery,
    *,
    anchor_col: str = "dt_DATERDV",
) -> pl.LazyFrame:
    """Filter LazyFrame to a date window on a datetime column.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input data.
    query : DashboardQuery
        Query object with date_start and/or date_end.
    anchor_col : str, default "dt_DATERDV"
        Name of the datetime column to filter (must be a parsed datetime).

    Returns
    -------
    pl.LazyFrame
        Filtered to date_start <= anchor_col < date_end, if set.
    """
    if query.date_start is not None:
        lf = lf.filter(pl.col(anchor_col) >= pl.lit(query.date_start))
    if query.date_end is not None:
        lf = lf.filter(pl.col(anchor_col) < pl.lit(query.date_end))
    return lf


def add_periode_column(
    lf: pl.LazyFrame,
    query: DashboardQuery,
    *,
    anchor_col: str = "dt_DATERDV",
    out_col: str = "periode",
) -> pl.LazyFrame:
    """Add a truncated period column to the frame.

    The anchor_col (datetime) is truncated to the period specified in query and output as out_col.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input data.
    query : DashboardQuery
        Query object specifying period.
    anchor_col : str, default "dt_DATERDV"
        Datetime column to truncate.
    out_col : str, default "periode"
        Name of the output period column.
    """
    return lf.with_columns(pl.col(anchor_col).dt.truncate(query.trunc).alias(out_col))

# =========================
# Build stay-level dataframe (1 row / stay)
# =========================

def build_stay_table(
    lf_journal: pl.LazyFrame,
    *,
    schema: DashboardSchema = DashboardSchema(),
    codes: DashboardEventCodes = DashboardEventCodes(),
    # Optionnel : limiter très tôt aux codes utiles (perf énorme)
    restrict_to_codes: Optional[Iterable[str]] = None,
) -> pl.LazyFrame:
    """Construct stay-level table (one row per (site, stay)) with key event datetimes and flags.

    This function parses event timestamps (from ACTION_DETAIL for relevant codes, using
    schema.dt_formats; for EVT_LOC and similar, uses schema.dt_col directly), and aggregates
    per-stay/per-site to produce:
      - dt_* columns: entry, IOA, physician, nurse, decision, exit times (as min(event_dt) per code)
      - flags: is_finish, has_dp, has_ccmu (True if code present in stay)
      - mode_sortie_raw/mode_sortie: raw and normalized disposition/orientation
      - is_hospit: bool, True if disposition indicates hospitalization

    Performance tip: restrict_to_codes can be used to filter very early for only relevant codes.

    Parameters
    ----------
    lf_journal : pl.LazyFrame
        Parsed JOURNAL dataframe (after parse_journal).
    schema : DashboardSchema, optional
        Schema describing column names and datetime formats.
    codes : DashboardEventCodes, optional
        Codes for key events.
    restrict_to_codes : Iterable[str], optional
        If provided, restricts input to these action codes for performance.

    Returns
    -------
    pl.LazyFrame
        Stay-level table: one row per (site, stay), with event datetimes and flags.
    """
    stay = schema.stay_col
    site = schema.site_col
    ac = schema.action_code_col
    ad = schema.action_detail_col

    # --- 0) Sélection minimaliste
    lf = lf_journal.select([site, stay, ac, ad, schema.dt_col])

    # --- 1) Optionnel : filtrer très tôt sur les codes utiles
    if restrict_to_codes is not None:
        code_list = list(restrict_to_codes)
        lf = lf.filter(pl.col(ac).is_in(code_list))

    # --- 2) Parse datetime des events (dans ACTION_DETAIL) uniquement pour les codes concernés
    # On ne parse pas pour tout : on construit event_dt puis on l'utilise dans des min() conditionnels.
    lf = lf.with_columns(
        _expr_parse_event_dt(ad, schema.dt_formats, out="event_dt")
    )

    # --- 3) Table "séjour" : 1 ligne par (site, stay)
    lf_stay = (
        lf.group_by([site, stay])
        .agg([
            # timestamps
            _expr_min_dt_for_code("event_dt", ac, codes.enter,   "dt_DATERDV"),
            _expr_min_dt_for_code("event_dt", ac, codes.ioa,     "dt_HPIOA"),
            _expr_min_dt_for_code("event_dt", ac, codes.med,     "dt_HPMED"),
            _expr_min_dt_for_code("event_dt", ac, codes.inf,     "dt_HPECINF"),
            _expr_min_dt_for_code("event_dt", ac, codes.decision,"dt_HDECIS"),
            _expr_min_dt_for_code("event_dt", ac, codes.exit,    "dt_DHSORTIESAU"),

            # flags
            _expr_any_code(ac, codes.exit, "is_finish"),
            _expr_any_code(ac, codes.diagp, "has_dp"),
            _expr_any_code(ac, codes.ccmu, "has_ccmu"),

            # mode sortie (texte) + is_hospit (bool)
            pl.col(ad)
              .filter(pl.col(ac).is_in(list(codes.mode_sortie_codes)))
              .first()
              .cast(pl.Utf8)
              .alias("mode_sortie_raw"),
        ])
        .with_columns(
            pl.col("mode_sortie_raw").str.to_lowercase().alias("mode_sortie")
        )
        .with_columns(
            pl.when(pl.col("mode_sortie").is_null())
              .then(None)
            .when(pl.col("mode_sortie").str.contains("hos"))
              .then(True)
            .when(pl.col("mode_sortie").str.contains("dom"))
              .then(False)
            .when(pl.col("mode_sortie").str.contains("sans"))
              .then(False)
            .when(pl.col("mode_sortie").str.contains("transf"))
              .then(False)
            .otherwise(None)
            .alias("is_hospit")
        )
    )

    return lf_stay


# =========================
# (Optionnel) Last location per stay
# =========================

def build_last_location_table(
    lf_journal: pl.LazyFrame,
    lf_mapping_box: pl.LazyFrame,
    *,
    schema: DashboardSchema = DashboardSchema(),
    codes: DashboardEventCodes = DashboardEventCodes(),
) -> pl.LazyFrame:
    """Return last known location per stay.

    For each (site, stay), returns the last known location category and box (from EVT_LOC events),
    using dt_col for ordering. Requires mapping_box to have columns SITE_UF, BOX, CATEGORY.

    Returns
    -------
    pl.LazyFrame
        Columns: site, stay, dt_last_loc, category_last, box_last.
    """
    stay = schema.stay_col
    site = schema.site_col
    ac = schema.action_code_col
    ad = schema.action_detail_col
    dt = schema.dt_col

    lf_loc = (
        lf_journal
        .select([stay, site, ac, ad, dt])
        .filter(pl.col(ac) == codes.loc)
        .join(
            lf_mapping_box.select([site, pl.col("BOX"), pl.col("CATEGORY")]),
            left_on=[ad, site],
            right_on=["BOX", site],
            how="left",
        )
        .group_by([site, stay])
        .agg([
            pl.col(dt).max().alias("dt_last_loc"),
            pl.col("CATEGORY").sort_by(dt).last().alias("category_last"),
            pl.col(ad).sort_by(dt).last().alias("box_last"),
        ])
    )
    return lf_loc


# =========================
# Derive delays + stats (from df_stay)
# =========================

def add_delay_columns(lf_stay: pl.LazyFrame) -> pl.LazyFrame:
    """Add delay columns (in minutes) to stay-level table.

    Produces the following columns (all in minutes, may be negative if timestamps are inconsistent):
      - d_entree_ioa: entry → IOA
      - d_entree_med: entry → physician
      - d_entree_orient: entry → orientation/decision
      - d_entree_sortie: entry → exit
      - d_orient_sortie: orientation → exit
      - d_decision_hospit_sortie: decision → exit (if is_hospit True)
      - d_decision_rad_sortie: decision → exit (if is_hospit False)
      - ioa_lt15: True if d_entree_ioa < 15
      - med_lt60: True if d_entree_med < 60

    Parameters
    ----------
    lf_stay : pl.LazyFrame
        Stay-level table from build_stay_table.

    Returns
    -------
    pl.LazyFrame
        With added delay columns (minutes).
    """
    return lf_stay.with_columns([
        (pl.col("dt_HPIOA") - pl.col("dt_DATERDV")).dt.total_minutes().alias("d_entree_ioa"),
        (pl.col("dt_HPMED") - pl.col("dt_DATERDV")).dt.total_minutes().alias("d_entree_med"),
        (pl.col("dt_HDECIS") - pl.col("dt_DATERDV")).dt.total_minutes().alias("d_entree_orient"),
        (pl.col("dt_DHSORTIESAU") - pl.col("dt_DATERDV")).dt.total_minutes().alias("d_entree_sortie"),
        (pl.col("dt_DHSORTIESAU") - pl.col("dt_HDECIS")).dt.total_minutes().alias("d_orient_sortie"),
        pl.when(pl.col("is_hospit") == True)
          .then((pl.col("dt_DHSORTIESAU") - pl.col("dt_HDECIS")).dt.total_minutes())
          .otherwise(None)
          .alias("d_decision_hospit_sortie"),
        pl.when(pl.col("is_hospit") == False)
          .then((pl.col("dt_DHSORTIESAU") - pl.col("dt_HDECIS")).dt.total_minutes())
          .otherwise(None)
          .alias("d_decision_rad_sortie"),
    ]).with_columns([
        (pl.col("d_entree_ioa") < 15).alias("ioa_lt15"),
        (pl.col("d_entree_med") < 60).alias("med_lt60"),
    ])


def delay_stats_by_site_period(
    lf_stay_with_delays: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    anchor_col: str = "dt_DATERDV",
) -> pl.LazyFrame:
    """Compute delay statistics by (site, period).

    Requires input to have delay columns (see add_delay_columns) and ioa_lt15/med_lt60 flags.
    Output columns: for each delay, n/mean/std/median/q25/q75; plus pct_ioa_lt15, pct_med_lt60.

    Parameters
    ----------
    lf_stay_with_delays : pl.LazyFrame
        Output of add_delay_columns.
    query : DashboardQuery
        Query parameters for filtering and period.
    schema : DashboardSchema, optional
        Schema for column names.
    anchor_col : str, default "dt_DATERDV"
        Datetime column for period and date filtering.

    Returns
    -------
    pl.LazyFrame
        One row per (site, periode) with stats.
    """
    site_col = schema.site_col

    lf = (
        lf_stay_with_delays
        .pipe(apply_site_filter, query, site_col=site_col)
        .pipe(apply_date_window, query, anchor_col=anchor_col)
        .pipe(add_periode_column, query, anchor_col=anchor_col, out_col="periode")
    )

    delay_cols = [
        "d_entree_ioa",
        "d_entree_med",
        "d_entree_orient",
        "d_entree_sortie",
        "d_orient_sortie",
        "d_decision_hospit_sortie",
        "d_decision_rad_sortie",
    ]

    agg: list[pl.Expr] = []
    for c in delay_cols:
        agg += [
            pl.col(c).drop_nulls().len().alias(f"{c}_n"),
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).median().alias(f"{c}_median"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
        ]

    # Percent IOA < 15 min and MED < 60 min (of those with non-null delays)
    agg += [
        (pl.col("ioa_lt15").cast(pl.Int64).sum() / pl.col("d_entree_ioa").drop_nulls().len())
          .alias("pct_ioa_lt15"),
        (pl.col("med_lt60").cast(pl.Int64).sum() / pl.col("d_entree_med").drop_nulls().len())
          .alias("pct_med_lt60"),
    ]

    return (
        lf.group_by([site_col, "periode"])
          .agg(agg)
          .sort([site_col, "periode"])
    )


def completeness_stats_by_site_period(
    lf_stay: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    anchor_col: str = "dt_DATERDV",
) -> pl.LazyFrame:
    """Compute completeness statistics for finished stays by (site, period).

    Uses the is_finish flag to select completed stays, then computes:
      - pct_no_dp: percent of finished stays missing DIAGP (has_dp False)
      - pct_no_ccmu: percent missing CCMU (has_ccmu False)

    Parameters
    ----------
    lf_stay : pl.LazyFrame
        Stay-level table from build_stay_table.
    query : DashboardQuery
        Query parameters for filtering and period.
    schema : DashboardSchema, optional
        Schema for column names.
    anchor_col : str, default "dt_DATERDV"
        Datetime column for period and date filtering.

    Returns
    -------
    pl.LazyFrame
        One row per (site, periode) with pct_no_dp and pct_no_ccmu.
    """
    site_col = schema.site_col

    lf = (
        lf_stay
        .filter(pl.col("is_finish") == True)
        .pipe(apply_site_filter, query, site_col=site_col)
        .pipe(apply_date_window, query, anchor_col=anchor_col)
        .pipe(add_periode_column, query, anchor_col=anchor_col, out_col="periode")
    )

    return (
        lf.group_by([site_col, "periode"])
          .agg([
              (1 - (pl.col("has_dp").cast(pl.Int64).sum() / pl.len())).alias("pct_no_dp"),
              (1 - (pl.col("has_ccmu").cast(pl.Int64).sum() / pl.len())).alias("pct_no_ccmu"),
          ])
          .sort([site_col, "periode"])
    )


# =========================
# (Optionnel) Hourly presence (for EDOR / occupancy)
# =========================


def build_hourly_presence(
    lf_stay: pl.LazyFrame,
    *,
    query: Optional[DashboardQuery] = None,
    schema: DashboardSchema = DashboardSchema(),
    start_col: str = "dt_DATERDV",
    end_col: str = "dt_DHSORTIESAU",
    keep_open_stays: bool = True,
) -> pl.LazyFrame:
    """Compute hourly patient presence by site/date/hour.

    For each stay, generates an hour_list (datetime range from start_col to end_col, 1h interval)
    and explodes to count patient presence per hour. If keep_open_stays is False, stays with null
    end_col (open stays) are excluded.

    Note: Filtering (site/date) should be performed before exploding for performance.

    Parameters
    ----------
    lf_stay : pl.LazyFrame
        Stay-level table.
    query : DashboardQuery or None, optional
        If provided, used to filter sites and date window.
    schema : DashboardSchema, optional
        Schema for column names.
    start_col : str, default "dt_DATERDV"
        Start datetime column.
    end_col : str, default "dt_DHSORTIESAU"
        End datetime column (may be null for open stays).
    keep_open_stays : bool, default True
        If False, exclude stays with null end_col.

    Returns
    -------
    pl.LazyFrame
        Columns: site, date, hour_int, n_patients (number present at each hour).
    """
    site_col = schema.site_col
    lf = lf_stay

    if query is not None:
        lf = lf.pipe(apply_site_filter, query, site_col=site_col)
        lf = lf.pipe(apply_date_window, query, anchor_col=start_col)

    if not keep_open_stays:
        lf = lf.filter(pl.col(end_col).is_not_null())

    # ⚠️ Explosion : à n'utiliser qu'après réduction temporelle si besoin
    return (
        lf.select([site_col, start_col, end_col])
          .with_columns(
              pl.datetime_ranges(
                  start=pl.col(start_col),
                  end=pl.col(end_col),
                  interval="1h",
              ).alias("hour_list")
          )
          .explode("hour_list")
          .with_columns([
              pl.col("hour_list").dt.date().alias("date"),
              pl.col("hour_list").dt.hour().alias("hour_int"),
          ])
          .group_by([site_col, "date", "hour_int"])
          .len()
          .rename({"len": "n_patients"})
          .sort([site_col, "date", "hour_int"])
    )


def compute_edor(
    lf_presence: pl.LazyFrame,
    lf_capacity_per_site_hour: pl.LazyFrame,
    *,
    site_col: str = "SITE_UF",
) -> pl.LazyFrame:
    """Compute EDOR (patient/capacity ratio) per site/date/hour.

    Joins patient presence with site/hourly capacity. EDOR is defined as n_patients / capacity,
    with 0.0 if capacity is zero or missing.

    Parameters
    ----------
    lf_presence : pl.LazyFrame
        Output of build_hourly_presence (must have columns: site_col, date, hour_int, n_patients).
    lf_capacity_per_site_hour : pl.LazyFrame
        Table of site/hour_int/capacity (number of available boxes).
    site_col : str, default "SITE_UF"
        Site column name.

    Returns
    -------
    pl.LazyFrame
        Columns: site_col, date, hour_int, n_patients, capacity, EDOR.
    """
    return (
        lf_presence
        .join(lf_capacity_per_site_hour, on=[site_col, "hour_int"], how="left")
        .with_columns(pl.col("capacity").fill_null(0))
        .with_columns(
            pl.when(pl.col("capacity") > 0)
              .then(pl.col("n_patients") / pl.col("capacity"))
              .otherwise(0.0)
              .alias("EDOR")
        )
    )

# =========================
# Additional dashboard metrics
# =========================

def entry_volumes_by_site_period(
    lf_stay: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    anchor_col: str = "dt_DATERDV",
    out_col: str = "n_entries",
) -> pl.LazyFrame:
    """Compute unique stay counts (entries) by site and period.

    This uses the stay-level table (one row per stay) and counts the number of stays
    (unique IPPDATE_multicol) per (SITE_UF, periode), where `periode` is derived by
    truncating `anchor_col` according to `query.period`.

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, periode, n_entries.
    """
    site_col = schema.site_col
    lf = (
        lf_stay
        .pipe(apply_site_filter, query, site_col=site_col)
        .pipe(apply_date_window, query, anchor_col=anchor_col)
        .pipe(add_periode_column, query, anchor_col=anchor_col, out_col="periode")
        .group_by([site_col, "periode"])
        .agg([pl.len().alias(out_col)])
        .sort([site_col, "periode"])
    )
    return lf


def entry_volumes_by_site_hour_of_day(
    lf_stay: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    anchor_col: str = "dt_DATERDV",
    out_col: str = "n_entries",
) -> pl.LazyFrame:
    """Compute entry volumes by site and hour-of-day.

    Counts unique stays per (SITE_UF, hour_int), where hour_int is extracted from
    `anchor_col` (0..23). This is useful to characterize arrival patterns.

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, hour_int, n_entries.
    """
    site_col = schema.site_col
    lf = (
        lf_stay
        .pipe(apply_site_filter, query, site_col=site_col)
        .pipe(apply_date_window, query, anchor_col=anchor_col)
        .with_columns(pl.col(anchor_col).dt.hour().alias("hour_int"))
        .group_by([site_col, "hour_int"])
        .agg([pl.len().alias(out_col)])
        .sort([site_col, "hour_int"])
    )
    return lf


def age_distribution_by_site_period(
    lf_journal: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    age_code: str = "AGERESU",
    anchor_code: str = "DATERDV",
    age_out: str = "AGE",
) -> pl.LazyFrame:
    """Compute age distribution by site and period.

    Extracts age from ACTION_DETAIL for rows with ACTION_CODE==AGERESU, and associates
    each stay to a period using the entry timestamp from DATERDV.

    Notes
    -----
    - Age parsing is heuristic: digits are extracted from ACTION_DETAIL (e.g. "45 ans").
    - This function expects the parsed journal to contain `dt` (schema.dt_col) for ordering,
      but the period anchor is computed from the parsed DATERDV (ACTION_DETAIL parsed with
      schema.dt_formats).

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, periode, AGE, n (count of stays with that age).
    """
    site = schema.site_col
    stay = schema.stay_col
    ac = schema.action_code_col
    ad = schema.action_detail_col

    # Minimal selection
    lf = lf_journal.select([site, stay, ac, ad])
    lf = lf.filter(pl.col(ac).is_in([age_code, anchor_code]))
    lf = lf.with_columns(
        _expr_parse_event_dt(ad, schema.dt_formats, out="event_dt")
    )
    # Group by (site, stay) and aggregate dt_DATERDV and age_raw
    lf = (
        lf.group_by([site, stay])
        .agg([
            pl.col("event_dt")
              .filter(pl.col(ac) == anchor_code)
              .min()
              .alias("dt_DATERDV"),
            pl.col(ad)
              .filter(pl.col(ac) == age_code)
              .first()
              .alias("age_raw"),
        ])
    )
    lf = (
        lf.pipe(apply_site_filter, query, site_col=site)
        .pipe(apply_date_window, query, anchor_col="dt_DATERDV")
        .pipe(add_periode_column, query, anchor_col="dt_DATERDV", out_col="periode")
        .with_columns(
            pl.col("age_raw")
              .cast(pl.Utf8)
              .str.extract(r"(\d+)", 1)
              .cast(pl.UInt64)
              .alias(age_out)
        )
        .group_by([site, "periode", age_out])
        .agg([pl.len().alias("n")])
        .sort([site, "periode", age_out])
    )
    return lf


def triage_distribution_by_site_period(
    lf_journal: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    triage_code: str = "GRAV",
    anchor_code: str = "DATERDV",
    triage_out: str = "NIV_TRI",
) -> pl.LazyFrame:
    """Compute triage level distribution (GRAV) by site and period.

    Extracts triage level from ACTION_DETAIL for rows with ACTION_CODE==GRAV, and assigns
    stays to a period using the entry timestamp (DATERDV).

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, periode, NIV_TRI, n.
    """
    site = schema.site_col
    stay = schema.stay_col
    ac = schema.action_code_col
    ad = schema.action_detail_col

    lf = lf_journal.select([site, stay, ac, ad])
    lf = lf.filter(pl.col(ac).is_in([triage_code, anchor_code]))
    lf = lf.with_columns(
        _expr_parse_event_dt(ad, schema.dt_formats, out="event_dt")
    )
    lf = (
        lf.group_by([site, stay])
        .agg([
            pl.col("event_dt")
              .filter(pl.col(ac) == anchor_code)
              .min()
              .alias("dt_DATERDV"),
            pl.col(ad)
              .filter(pl.col(ac) == triage_code)
              .first()
              .alias("triage_raw"),
        ])
    )
    lf = (
        lf.pipe(apply_site_filter, query, site_col=site)
        .pipe(apply_date_window, query, anchor_col="dt_DATERDV")
        .pipe(add_periode_column, query, anchor_col="dt_DATERDV", out_col="periode")
        .with_columns(
            pl.col("triage_raw")
              .cast(pl.Utf8)
              .str.extract(r"(\d+)", 1)
              .cast(pl.UInt64)
              .alias(triage_out)
        )
        .group_by([site, "periode", triage_out])
        .agg([pl.len().alias("n")])
        .sort([site, "periode", triage_out])
    )
    return lf


def waiting_counts_by_site_period(
    lf_stay: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    anchor_col: str = "dt_DATERDV",
    only_open_stays: bool = True,
) -> pl.LazyFrame:
    """Count stays waiting for IOA and/or physician by site and period.

    A stay is considered:
    - waiting_ioa if dt_HPIOA is null
    - waiting_med if dt_HPMED is null

    If `only_open_stays` is True, the computation is restricted to stays with null
    dt_DHSORTIESAU (i.e., not finished).

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, periode, n_open, n_waiting_ioa, n_waiting_med.
    """
    site_col = schema.site_col
    lf = lf_stay
    if only_open_stays:
        lf = lf.filter(pl.col("dt_DHSORTIESAU").is_null())
    lf = (
        lf.pipe(apply_site_filter, query, site_col=site_col)
        .pipe(apply_date_window, query, anchor_col=anchor_col)
        .pipe(add_periode_column, query, anchor_col=anchor_col, out_col="periode")
        .group_by([site_col, "periode"])
        .agg([
            pl.len().alias("n_open"),
            pl.col("dt_HPIOA").is_null().cast(pl.Int64).sum().alias("n_waiting_ioa"),
            pl.col("dt_HPMED").is_null().cast(pl.Int64).sum().alias("n_waiting_med"),
        ])
        .sort([site_col, "periode"])
    )
    return lf


def build_capacity_per_site_hour(
    lf_mapping_box: pl.LazyFrame,
    *,
    site_col: str = "SITE_UF",
    box_col: str = "BOX",
    category_col: str = "CATEGORY",
    open_col: str = "dt_START",
    close_col: str = "dt_END",
    categories: Sequence[str] = ("BOX_CS", "SAUV"),
) -> pl.LazyFrame:
    """Build hourly capacity (number of available boxes) by site and hour.

    The mapping table is expected to contain per-box opening hours as strings ("HH:MM").
    Capacity is computed as the number of unique boxes available for each (SITE_UF, hour_int).

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, hour_int, capacity.
    """
    lf = lf_mapping_box.filter(pl.col(category_col).is_in(list(categories)))
    lf = (
        lf
        .with_columns([
            pl.col(open_col).str.split_exact(":", 1).struct.field("field_0").cast(pl.UInt64).alias("open_hour"),
            pl.col(close_col).str.split_exact(":", 1).struct.field("field_0").cast(pl.UInt64).alias("close_hour"),
        ])
        .with_columns(
            pl.int_ranges(
                start=pl.col("open_hour"),
                end=pl.col("close_hour") + 1,
                step=1,
                eager=False
            ).alias("hour_int")
        )
        .explode("hour_int")
        .group_by([site_col, "hour_int"])
        .agg([
            pl.col(box_col).n_unique().alias("capacity")
        ])
    )
    return lf


def hourly_box_occupancy(
    lf_journal: pl.LazyFrame,
    lf_mapping_box: pl.LazyFrame,
    lf_stay: pl.LazyFrame,
    *,
    query: Optional[DashboardQuery] = None,
    schema: DashboardSchema = DashboardSchema(),
    codes: DashboardEventCodes = DashboardEventCodes(),
    categories: Sequence[str] = ("BOX_CS", "SAUV"),
) -> pl.LazyFrame:
    """Compute occupied boxes per site/date/hour from location traces.

    This approximates box occupancy by:
    1) taking EVT_LOC events,
    2) restricting to boxes in `categories` via the mapping table,
    3) building intervals per stay between successive location timestamps,
    4) exploding to hours and counting unique boxes seen as occupied.

    Notes
    -----
    - Requires `lf_stay` to provide dt_DHSORTIESAU for interval end when a location has no next event.
    - For open stays (missing dt_DHSORTIESAU), those final intervals are dropped.

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, date, hour_int, occupied_boxes.
    """
    site = schema.site_col
    stay = schema.stay_col
    ac = schema.action_code_col
    ad = schema.action_detail_col
    dt = schema.dt_col

    # Mapping: only keep boxes in requested categories
    lf_box_map = lf_mapping_box.filter(pl.col("CATEGORY").is_in(list(categories))).select([site, "BOX", "CATEGORY"])
    # 1. Extract location events
    lf_loc = (
        lf_journal
        .select([stay, site, ac, ad, dt])
        .filter(pl.col(ac) == codes.loc)
        .join(
            lf_box_map,
            left_on=[ad, site],
            right_on=["BOX", site],
            how="inner",
        )
    )
    # 2. Sort and compute dt_start/dt_next per stay
    lf_loc2 = (
        lf_loc
        .sort([stay, dt])
        .with_columns([
            pl.col(dt).alias("dt_start"),
            pl.col(dt).shift(-1).over(stay).alias("dt_next"),
        ])
    )
    # 3. Join with dt_DHSORTIESAU from stay table
    lf_loc3 = (
        lf_loc2
        .join(
            lf_stay.select([site, stay, "dt_DHSORTIESAU"]),
            on=[site, stay],
            how="left"
        )
        .with_columns(
            pl.when(pl.col("dt_next").is_null())
            .then(pl.col("dt_DHSORTIESAU"))
            .otherwise(pl.col("dt_next"))
            .alias("dt_end")
        )
        .filter(pl.col("dt_end").is_not_null())
    )
    # 4. Optionally filter by site/date window using dt_start as anchor
    if query is not None:
        lf_loc3 = lf_loc3.pipe(apply_site_filter, query, site_col=site)
        lf_loc3 = lf_loc3.pipe(apply_date_window, query, anchor_col="dt_start")
    # 5. Explode hours
    lf_hours = (
        lf_loc3
        .with_columns(
            pl.datetime_ranges(
                start=pl.col("dt_start"),
                end=pl.col("dt_end"),
                interval="1h",
            ).alias("hour_list")
        )
        .explode("hour_list")
        .with_columns([
            pl.col("hour_list").dt.date().alias("date"),
            pl.col("hour_list").dt.hour().alias("hour_int"),
        ])
    )
    # 6. Group by and count unique occupied boxes
    lf_occ = (
        lf_hours
        .group_by([site, "date", "hour_int"])
        .agg([
            pl.col(ad).n_unique().alias("occupied_boxes")
        ])
        .sort([site, "date", "hour_int"])
    )
    return lf_occ


def compute_box_occupancy_rate(
    lf_occupied: pl.LazyFrame,
    lf_capacity: pl.LazyFrame,
    *,
    site_col: str = "SITE_UF",
) -> pl.LazyFrame:
    """Join occupied boxes with capacity and compute occupancy rate.

    Occupancy rate is defined as occupied_boxes / capacity per site/date/hour.

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, date, hour_int, occupied_boxes, capacity, taux_occupation.
    """
    return (
        lf_occupied
        .join(lf_capacity, on=[site_col, "hour_int"], how="left")
        .with_columns(pl.col("capacity").fill_null(0))
        .with_columns(
            pl.when(pl.col("capacity") > 0)
            .then(pl.col("occupied_boxes") / pl.col("capacity"))
            .otherwise(0.0)
            .alias("taux_occupation")
        )
    )


def hospitalization_rate_by_site_period(
    lf_stay: pl.LazyFrame,
    query: DashboardQuery,
    *,
    schema: DashboardSchema = DashboardSchema(),
    anchor_col: str = "dt_DATERDV",
    only_finished: bool = True,
) -> pl.LazyFrame:
    """Compute hospitalization rate (is_hospit) by site and period.

    By default, restricts to finished stays (dt_DHSORTIESAU not null) to avoid bias.

    Returns
    -------
    pl.LazyFrame
        Columns: SITE_UF, periode, taux_hospit.
    """
    site_col = schema.site_col
    lf = lf_stay
    if only_finished:
        lf = lf.filter(pl.col("dt_DHSORTIESAU").is_not_null())
    lf = (
        lf.pipe(apply_site_filter, query, site_col=site_col)
        .pipe(apply_date_window, query, anchor_col=anchor_col)
        .pipe(add_periode_column, query, anchor_col=anchor_col, out_col="periode")
        .group_by([site_col, "periode"])
        .agg([
            (
                pl.col("is_hospit").cast(pl.Int64).sum()
                / pl.col("is_hospit").drop_nulls().len()
            ).alias("taux_hospit")
        ])
        .sort([site_col, "periode"])
    )
    return lf
# =========================
# Hourly staff counts (wide)
# =========================

def build_hourly_staff_counts(
    lf_journal: pl.LazyFrame,
    lf_user_role_map: pl.LazyFrame,
    *,
    query: Optional[DashboardQuery] = None,
    schema: DashboardSchema = DashboardSchema(),
    staff: DashboardStaffConfig = DashboardStaffConfig(),
) -> pl.LazyFrame:
    """Compute hourly staff counts by site/date/hour/role from journal actions.

    For each hour, infers staff presence by detecting at least one logged action per user in that hour,
    optionally restricting to specific action codes. Joins with a user-to-role mapping to assign each
    user a role (e.g., "MED", "INF"), and aggregates counts per site/date/hour/role.

    Parameters
    ----------
    lf_journal : pl.LazyFrame
        Parsed JOURNAL dataframe (must have columns for user, action code, and timestamp).
    lf_user_role_map : pl.LazyFrame
        Table mapping user IDs to roles (columns: user_col, "role").
    query : DashboardQuery, optional
        If provided, applies site and date filtering.
    schema : DashboardSchema, optional
        Schema for site and action code column names.
    staff : DashboardStaffConfig, optional
        Configuration for staff inference.

    Returns
    -------
    pl.LazyFrame
        Columns: site, date, hour_int, n_med, n_inf (0-filled if missing or absent for a slot).
        Other roles are ignored for simplicity.
    """
    site_col = schema.site_col
    user_col = staff.user_col
    dt_col = staff.dt_staff_col
    action_code_col = schema.action_code_col

    # Minimal selection
    cols = [site_col, user_col, action_code_col, dt_col]
    lf = lf_journal.select(cols)

    # Restrict to active action codes if provided
    if staff.active_action_codes is not None:
        lf = lf.filter(pl.col(action_code_col).is_in(list(staff.active_action_codes)))

    # Join user role map (left join: keep only users with a role)
    lf = (
        lf.join(
            lf_user_role_map.select([user_col, "role"]),
            on=user_col,
            how="left",
        )
        .filter(pl.col("role").is_not_null())
    )

    # Apply site/date filtering if query is provided
    if query is not None:
        lf = apply_site_filter(lf, query, site_col=site_col)
        lf = apply_date_window(lf, query, anchor_col=dt_col)

    # Compute hour/date columns
    lf = lf.with_columns([
        pl.col(dt_col).dt.truncate("1h").alias("hour_dt"),
    ]).with_columns([
        pl.col("hour_dt").dt.date().alias("date"),
        pl.col("hour_dt").dt.hour().alias("hour_int"),
    ])

    # Aggregate: group by site/date/hour/role
    group_keys = [site_col, "date", "hour_int", "role"]
    if staff.count_distinct_per_hour:
        agg_expr = pl.col(user_col).n_unique().alias("n_staff")
    else:
        agg_expr = pl.len().alias("n_staff")

    lf = lf.group_by(group_keys).agg(agg_expr)

    # Pivot to wide: columns per role (e.g., MED, INF)
    lf = (
        lf.pivot(
            values="n_staff",
            index=[site_col, "date", "hour_int"],
            columns="role",
            aggregate_function="first"
        )
    )

    # Fill missing role columns (MED/INF) with 0 and rename
    for role, outcol in [("MED", "n_med"), ("INF", "n_inf")]:
        if role not in lf.schema:
            lf = lf.with_columns(pl.lit(0).alias(role))
        lf = lf.with_columns(pl.col(role).fill_null(0).cast(pl.Int64).alias(role))
        lf = lf.rename({role: outcol})

    # Sort for consistency
    lf = lf.sort([site_col, "date", "hour_int"])
    return lf


# =========================
# Patient/staff ratios
# =========================

def compute_patient_staff_ratios(
    lf_presence: pl.LazyFrame,
    lf_staff_counts: pl.LazyFrame,
    *,
    site_col: str = "SITE_UF",
) -> pl.LazyFrame:
    """Compute ratios of patients per staff and staff per 10 patients by site/date/hour.

    Joins patient hourly presence and staff hourly counts. For each hour:
      - patients_per_med = n_patients / n_med (None if n_med==0)
      - patients_per_inf = n_patients / n_inf (None if n_inf==0)
      - med_per_10_patients = n_med / n_patients * 10 (None if n_patients==0)
      - inf_per_10_patients = n_inf / n_patients * 10 (None if n_patients==0)

    Parameters
    ----------
    lf_presence : pl.LazyFrame
        Output of build_hourly_presence (columns: site_col, date, hour_int, n_patients).
    lf_staff_counts : pl.LazyFrame
        Output of build_hourly_staff_counts (columns: site_col, date, hour_int, n_med, n_inf).
    site_col : str, default "SITE_UF"
        Site column name.

    Returns
    -------
    pl.LazyFrame
        Columns: site_col, date, hour_int, n_patients, n_med, n_inf, patients_per_med,
        patients_per_inf, med_per_10_patients, inf_per_10_patients.
    """
    # Join on site/date/hour
    lf = (
        lf_presence.join(
            lf_staff_counts,
            on=[site_col, "date", "hour_int"],
            how="left"
        )
        .with_columns([
            pl.col("n_med").fill_null(0).cast(pl.Int64),
            pl.col("n_inf").fill_null(0).cast(pl.Int64),
        ])
        .with_columns([
            pl.when(pl.col("n_med") > 0)
              .then(pl.col("n_patients") / pl.col("n_med"))
              .otherwise(None)
              .alias("patients_per_med"),
            pl.when(pl.col("n_inf") > 0)
              .then(pl.col("n_patients") / pl.col("n_inf"))
              .otherwise(None)
              .alias("patients_per_inf"),
            pl.when(pl.col("n_patients") > 0)
              .then(pl.col("n_med") / pl.col("n_patients") * 10)
              .otherwise(None)
              .alias("med_per_10_patients"),
            pl.when(pl.col("n_patients") > 0)
              .then(pl.col("n_inf") / pl.col("n_patients") * 10)
              .otherwise(None)
              .alias("inf_per_10_patients"),
        ])
        .select([
            site_col, "date", "hour_int",
            "n_patients", "n_med", "n_inf",
            "patients_per_med", "patients_per_inf",
            "med_per_10_patients", "inf_per_10_patients",
        ])
    )
    return lf