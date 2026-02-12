from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import polars as pl

@dataclass(frozen=True)
class LocalStore:
    root: Path  # ex: Path("data")

    def _abs(self, relpath: str) -> str:
        p = (self.root / relpath).resolve()
        return str(p)

    def read_csv(self, relpath: str, **kwargs) -> pl.DataFrame:
        path = self._abs(relpath)
        return pl.read_csv(path, **kwargs)

    def read_parquet(self, relpath: str, **kwargs) -> pl.DataFrame:
        path = self._abs(relpath)
        return pl.read_parquet(path, **kwargs)