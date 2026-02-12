from __future__ import annotations
from pathlib import Path
import dagster as dg
import polars as pl

class PolarsParquetIOManager(dg.IOManager):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def _path_for_output(self, context: dg.OutputContext) -> Path:
        # écrit sous base_dir / asset_key_path / partition.parquet
        parts = list(context.asset_key.path)
        p = self.base_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)

        if context.has_partition_key:
            return p / f"{context.partition_key}.parquet"
        return p / "data.parquet"

    def handle_output(self, context: dg.OutputContext, obj: pl.DataFrame):
        path = self._path_for_output(context)
        obj.write_parquet(path)
        context.add_output_metadata({"path": str(path), "rows": obj.height})

    def load_input(self, context: dg.InputContext) -> pl.DataFrame:
        # charge "data.parquet" ou partition spécifique si upstream partitionné
        parts = list(context.asset_key.path)
        p = self.base_dir.joinpath(*parts)
        if context.has_partition_key:
            return pl.read_parquet(p / f"{context.partition_key}.parquet")
        return pl.read_parquet(p / "data.parquet")