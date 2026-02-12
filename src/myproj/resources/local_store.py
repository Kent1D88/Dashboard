from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import polars as pl

@dataclass
class LocalStore:
    root: Path

    def read_parquet(self, relpath: str) -> pl.DataFrame:
        path = (self.root / relpath).resolve()
        return pl.read_parquet(path)

    def read_csv(self, relpath: str) -> pl.DataFrame:
        path = (self.root / relpath).resolve()
        return pl.read_csv(path)