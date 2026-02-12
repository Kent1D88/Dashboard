CREATE TABLE IF NOT EXISTS gold_entries (
  SITE_UF String,
  periode DateTime64(6),
  n_entries UInt32,
  computed_at DateTime64(6)
)
ENGINE = ReplacingMergeTree(computed_at)
PARTITION BY toYYYYMM(periode)
ORDER BY (SITE_UF, periode);

CREATE TABLE IF NOT EXISTS gold_delays (
  SITE_UF String,
  periode DateTime64(6),

  d_entree_ioa_n UInt32,
  d_entree_ioa_mean Float64,
  d_entree_ioa_std Float64,
  d_entree_ioa_median Float64,
  d_entree_ioa_q25 Float64,
  d_entree_ioa_q75 Float64,

  d_entree_med_n UInt32,
  d_entree_med_mean Float64,
  d_entree_med_std Float64,
  d_entree_med_median Float64,
  d_entree_med_q25 Float64,
  d_entree_med_q75 Float64,

  d_entree_orient_n UInt32,
  d_entree_orient_mean Float64,
  d_entree_orient_std Float64,
  d_entree_orient_median Float64,
  d_entree_orient_q25 Float64,
  d_entree_orient_q75 Float64,

  d_entree_sortie_n UInt32,
  d_entree_sortie_mean Float64,
  d_entree_sortie_std Float64,
  d_entree_sortie_median Float64,
  d_entree_sortie_q25 Float64,
  d_entree_sortie_q75 Float64,

  d_orient_sortie_n UInt32,
  d_orient_sortie_mean Float64,
  d_orient_sortie_std Float64,
  d_orient_sortie_median Float64,
  d_orient_sortie_q25 Float64,
  d_orient_sortie_q75 Float64,

  d_decision_hospit_sortie_n UInt32,
  d_decision_hospit_sortie_mean Float64,
  d_decision_hospit_sortie_std Float64,
  d_decision_hospit_sortie_median Float64,
  d_decision_hospit_sortie_q25 Float64,
  d_decision_hospit_sortie_q75 Float64,

  d_decision_rad_sortie_n UInt32,
  d_decision_rad_sortie_mean Float64,
  d_decision_rad_sortie_std Float64,
  d_decision_rad_sortie_median Float64,
  d_decision_rad_sortie_q25 Float64,
  d_decision_rad_sortie_q75 Float64,

  pct_ioa_lt15 Float64,
  pct_med_lt60 Float64,

  computed_at DateTime64(6)
)
ENGINE = ReplacingMergeTree(computed_at)
PARTITION BY toYYYYMM(periode)
ORDER BY (SITE_UF, periode);

CREATE TABLE IF NOT EXISTS gold_quality (
  SITE_UF String,
  periode DateTime64(6),
  pct_no_dp Float64,
  pct_no_ccmu Float64,
  computed_at DateTime64(6)
)
ENGINE = ReplacingMergeTree(computed_at)
PARTITION BY toYYYYMM(periode)
ORDER BY (SITE_UF, periode);

CREATE TABLE IF NOT EXISTS gold_edor_hourly (
  SITE_UF String,
  date Date,
  hour_int UInt8,
  n_patients UInt32,
  capacity UInt32,
  EDOR Float64,
  computed_at DateTime64(6)
)
ENGINE = ReplacingMergeTree(computed_at)
PARTITION BY toYYYYMM(date)
ORDER BY (SITE_UF, date, hour_int);