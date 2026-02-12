import polars as pl

def ensure_dt64us(df: pl.DataFrame, prefix: str = "dt_") -> pl.DataFrame:
    # force toutes les colonnes dt_* en Datetime(us)
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return df
    return df.with_columns([pl.col(c).cast(pl.Datetime("us")) for c in cols])

def ensure_strings(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in existing])