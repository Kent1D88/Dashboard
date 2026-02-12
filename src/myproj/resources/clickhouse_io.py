from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import dagster as dg
import polars as pl
from clickhouse_connect import get_client


@dataclass
class PolarsClickHouseIOManager(dg.IOManager):
    host: str
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "default"
    # Si None: table = "__".join(asset_key.path)
    table: Optional[str] = None

    def _client(self):
        return get_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
        )

    def _table_for(self, context_asset_key: dg.AssetKey) -> str:
        return self.table or "__".join(context_asset_key.path)

    def handle_output(self, context: dg.OutputContext, obj: pl.DataFrame):
        table = self._table_for(context.asset_key)

        if obj.is_empty():
            context.add_output_metadata({"clickhouse_table": table, "rows": 0, "skipped": True})
            return

        # Insert Arrow (rapide + types DateTime64 OK si déjà conformes à tes DDL)
        self._client().insert_arrow(table, obj.to_arrow())

        context.add_output_metadata({"clickhouse_table": table, "rows": obj.height})

    def load_input(self, context: dg.InputContext) -> pl.DataFrame:
        table = self._table_for(context.asset_key)
        arrow_tbl = self._client().query_arrow(f"SELECT * FROM {table}")
        return pl.from_arrow(arrow_tbl)