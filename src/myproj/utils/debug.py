from __future__ import annotations

import os
import json
from typing import Iterable, Optional

import polars as pl
import dagster as dg


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_set(name: str) -> set[str]:
    v = os.getenv(name, "").strip()
    if not v:
        return set()
    return {x.strip() for x in v.split(",") if x.strip()}


def debug_enabled(step: str) -> bool:
    """Global + per-step switch."""
    if not _env_bool("DASH_DEBUG", False):
        return False
    allowed = _env_set("DASH_DEBUG_STEPS")  # if empty => all steps
    return (not allowed) or (step in allowed)


def log_df(
    context: dg.AssetExecutionContext,
    df: pl.DataFrame,
    *,
    step: str,
    name: str,
    keys: Optional[Iterable[str]] = None,
    sample_rows: Optional[int] = None,
) -> None:
    """Structured, compact, Dagster-friendly Polars debug."""
    if not debug_enabled(step):
        return

    n = sample_rows if sample_rows is not None else _env_int("DASH_DEBUG_ROWS", 5)
    keys = list(keys) if keys else []

    # Basic stats
    meta: dict = {
        "debug": True,
        "step": step,
        "name": name,
        "rows": df.height,
        "cols": df.width,
        "columns": df.columns,
        "dtypes": {k: str(v) for k, v in df.schema.items()},
    }

    # Null rates (top few)
    try:
        nulls = (
            df.null_count()
            .transpose(include_header=True, header_name="col", column_names=["nulls"])
            .with_columns((pl.col("nulls") / max(df.height, 1)).alias("null_rate"))
            .sort("null_rate", descending=True)
        )
        meta["nulls_top"] = nulls.head(10).to_dicts()
    except Exception as e:
        meta["nulls_top_error"] = repr(e)

    # Key uniqueness checks
    if keys:
        missing = [c for c in keys if c not in df.columns]
        meta["keys"] = {"requested": keys, "missing": missing}
        if not missing and df.height > 0:
            try:
                dup = df.select(pl.len() - pl.n_unique(pl.struct(keys))).item()
                meta["keys"]["duplicates_count"] = int(dup)
            except Exception as e:
                meta["keys"]["duplicates_error"] = repr(e)

    # Sample
    try:
        sample = df.head(n)
        meta["sample_head"] = sample.to_dicts()
    except Exception as e:
        meta["sample_head_error"] = repr(e)

    context.log.debug(json.dumps(meta, ensure_ascii=False, default=str))


def log_msg(context: dg.AssetExecutionContext, *, step: str, msg: str, **fields) -> None:
    if not debug_enabled(step):
        return
    payload = {"debug": True, "step": step, "msg": msg, **fields}
    context.log.debug(json.dumps(payload, ensure_ascii=False, default=str))