from __future__ import annotations
import datetime as dt
import dagster as dg
import polars as pl

from myproj.assets.bronze_journal import URQUAL_PARTITIONS, _BATCH_SIZE

# Tes fonctions métier (à adapter aux vrais chemins)
import myproj.utils.journal_parse as JP
import myproj.utils.dashboard_functions as U
from myproj.utils.ch_types import ensure_dt64us, ensure_strings
from myproj.utils.debug import log_df, log_msg, debug_enabled


@dg.asset(
    key_prefix=["silver"],
    name="stay_delays",
    io_manager_key="clickhouse_silver",
    partitions_def=URQUAL_PARTITIONS,
    description="SILVER — stay+delays (par partition NOREC) -> ClickHouse.silver_stay_delays",
    ins={
        "journal": dg.AssetIn(key=dg.AssetKey(["bronze", "journal"])),
        "multicol": dg.AssetIn(key=dg.AssetKey(["dims", "multicol"])),
        "uf": dg.AssetIn(key=dg.AssetKey(["dims", "uf"])),
    },
)
def silver_stay_delays(
    context,
    journal: pl.DataFrame,
    multicol: pl.DataFrame,
    uf: pl.DataFrame,
) -> pl.DataFrame:
    # sanity check (utile en debug)
    context.log.info("Starting silver transformation")

    context.log.info(f"journal shape={journal.shape}")
    context.log.info(f"multicol shape={multicol.shape}")

    if debug_enabled("silver"):
        context.log.debug(f"journal columns={journal.columns}")
        context.log.debug(f"multicol columns={multicol.columns}")
        context.log.debug(f"uf columns={uf.columns}")
        log_df(context, journal, step="silver", name="input_journal", keys=["NOREC"])
        log_df(context, multicol, step="silver", name="input_multicol", keys=["IPP", "IPPDATE"])
        log_df(context, uf, step="silver", name="input_uf", keys=["SITE_UF"])
        
    part = int(context.partition_key)
    start = (part - 1) * _BATCH_SIZE
    end = start + _BATCH_SIZE - 1
    context.log.info(f"SILVER stay_delays partition={part} NOREC[{start},{end}]")

    # Parse + enrich
    df_journal = JP.parse_journal(df_journal=journal, drop_col=True, normalize_keys=True)
    log_df(context, df_journal, step="silver", name="after_parse_journal", keys=["NOREC"])
    
    df_journal, _ = JP.add_concordance_bt_JOURNAL_MULTICOL(
        df_journal=df_journal,
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
        pl.lit(part).cast(pl.Int32).alias("norec_partition"),
        pl.lit(start).cast(pl.Int64).alias("norec_start"),
        pl.lit(end).cast(pl.Int64).alias("norec_end"),
        pl.lit(context.run_id).alias("ingest_run_id"),
    ])
    
    df_out = ensure_dt64us(df_out)

    # Optionnel : éviter des surprises de types ClickHouse
    df_out = ensure_strings(df_out, ["SITE_UF", "IPPDATE_multicol", "mode_sortie_raw", "mode_sortie"])
    log_df(context, df_out, step="silver", name="stay_delays_output", keys=["SITE_UF", "IPPDATE_multicol", "dt_DATERDV"])
    
    return df_out