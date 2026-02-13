from __future__ import annotations
import datetime as dt
import dagster as dg
import polars as pl

from myproj.assets.bronze_journal import JOURNAL_MONTHLY

# Tes fonctions métier (à adapter aux vrais chemins)
import myproj.utils.journal_parse as JP
import myproj.utils.dashboard_functions as U
from myproj.utils.ch_types import ensure_dt64us, ensure_strings
from myproj.utils.debug import log_df, log_msg, debug_enabled

def _next_month(partition_key: str) -> str:
    year, month = map(int, partition_key.split("-"))
    if month == 12:
        return f"{year+1}-01"
    return f"{year}-{month+1:02d}"

@dg.asset(
    key_prefix=["silver"],
    name="stay_delays",
    io_manager_key="clickhouse_silver",
    partitions_def=JOURNAL_MONTHLY,
    description="SILVER — stay+delays (par partition NOREC) -> ClickHouse.silver_stay_delays",
    ins={
        "journal": dg.AssetIn(key=dg.AssetKey(["bronze", "journal_monthly"])),
        "multicol": dg.AssetIn(key=dg.AssetKey(["bronze", "multicol"])),
        "uf": dg.AssetIn(key=dg.AssetKey(["dims", "uf"])),
    },
)
def silver_stay_delays(
    context,
    journal_monthly: pl.DataFrame,
    multicol: pl.DataFrame,
    uf: pl.DataFrame,
) -> pl.DataFrame:
    # sanity check (utile en debug)
    context.log.info("Starting silver transformation")

    context.log.info(f"journal shape={journal_monthly.shape}")
    context.log.info(f"multicol shape={multicol.shape}")

    if debug_enabled("silver"):
        context.log.debug(f"journal columns={journal_monthly.columns}")
        context.log.debug(f"multicol columns={multicol.columns}")
        context.log.debug(f"uf columns={uf.columns}")
        log_df(context, journal_monthly, step="silver", name="input_journal", keys=["NOREC"])
        log_df(context, multicol, step="silver", name="input_multicol", keys=["IPP", "IPPDATE"])
        log_df(context, uf, step="silver", name="input_uf", keys=["SITE_UF"])
        
    part = context.partition_key  # "YYYY-MM"
    
    next_month = _next_month(partition_key=part)

    context.log.info(
        f"SILVER monthly={part} lookahead={next_month}"
    )
    
    # Charger le mois courant
    df_current = journal_monthly
    
    # Charger le mois suivant
    try:
        df_next = context.asset_partition_for_input(
            "journal_monthly",
            partition_key=next_month
        )
    except Exception:
        df_next = None

    if df_next is not None:
        df_events = pl.concat([df_current, df_next])
    else:
        df_events = df_current
    
    df_journal, _ = JP.add_concordance_bt_JOURNAL_MULTICOL(
        df_journal=df_events,
        df_multicol=multicol,
        drop_na_ippdate=True,
    )
    log_df(context, df_journal, step="silver", name="after_concordance")
    
    df_journal, _ = JP.add_UF_name(
        df_journal=df_journal,
        df_UF=uf,
        uf_select="SAU",
    )
    log_df(context, df_journal, step="silver", name="after_add_uf")
    
    # stay + delays
    lf_stay = U.build_stay_table(
        lf_journal=df_journal.lazy(),
        restrict_to_codes=[
            "DATERDV",
            "HPECIAO",
            "HPECMED",
            "HPECINF",
            "HEURDECIS1",
            "DHSORTIESAU",
            "DIAGP",
            "CCMU",
            "MODEDESORTIE",
            "REALSORTIE",
        ],
    )

    df_out = U.add_delay_columns(lf_stay=lf_stay).collect()

    # tracking (conforme à ton DDL silver)
    now = dt.datetime.now()
    df_out = df_out.with_columns([
        pl.lit(now).alias("ingested_at"),
        pl.lit(part).alias("journal_partition"),
        pl.lit(context.run_id).alias("ingest_run_id"),
    ])
    
    df_out = ensure_dt64us(df_out)

    # Optionnel : éviter des surprises de types ClickHouse
    df_out = ensure_strings(df_out, ["SITE_UF", "IPPDATE_multicol", "mode_sortie_raw", "mode_sortie"])
    log_df(context, df_out, step="silver", name="stay_delays_output", keys=["SITE_UF", "IPPDATE_multicol", "dt_DATERDV"])
    context.log.info(f"Columns after stay: {df_out.columns}")
    context.log.info(f"IST null count: {df_out.select(pl.col('IST').is_null().sum())}")
    return df_out