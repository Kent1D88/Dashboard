CREATE TABLE IF NOT EXISTS silver_stay_delays (
    SITE_UF String,
    IPPDATE_multicol String,

    dt_DATERDV DateTime64(6),
    dt_HPIOA DateTime64(6),
    dt_HPMED DateTime64(6),
    dt_HPECINF DateTime64(6),
    dt_HDECIS DateTime64(6),
    dt_DHSORTIESAU DateTime64(6),

    is_finish Bool,
    has_dp Bool,
    has_ccmu Bool,

    mode_sortie_raw String,
    mode_sortie String,
    is_hospit Bool,

    d_entree_ioa Int64,
    d_entree_med Int64,
    d_entree_orient Int64,
    d_entree_sortie Int64,
    d_orient_sortie Int64,
    d_decision_hospit_sortie Int64,
    d_decision_rad_sortie Int64,

    ioa_lt15 Bool,
    med_lt60 Bool,

    ingested_at DateTime64(6),
    norec_partition Int32,
    norec_start Int64,
    norec_end Int64,
    ingest_run_id String
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(dt_DATERDV)
ORDER BY (SITE_UF, dt_DATERDV, IPPDATE_multicol);