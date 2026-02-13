from __future__ import annotations
import os
from pathlib import Path

import dagster as dg
from clickhouse_connect import get_client

from myproj.resources.local_store import LocalStore
from myproj.resources.minio_io import PolarsMinioParquetIOManager
from myproj.resources.clickhouse_io import PolarsClickHouseIOManager

from myproj.sensors.dashboard_sensor import silver_sensor, bronze_sensor, gold_sensor

from myproj.assets.bronze_journal import bronze_journal
from myproj.assets.silver_journal import silver_stay_delays
from myproj.assets.gold_journal import gold_entries, gold_delays, gold_quality, gold_edor_hourly
from myproj.assets.dims import dim_multicol, dim_uf, dim_mappingbox

@dg.resource
def store() -> LocalStore:
    return LocalStore(root=Path("data"))


@dg.resource
def clickhouse_client():
    return get_client(
        host=os.environ.get("CH_HOST", "localhost"),
        port=int(os.environ.get("CH_PORT", "8123")),
        username=os.environ.get("CH_USER", "admin"),
        password=os.environ.get("CH_PASSWORD", "admin"),
        database=os.environ.get("CH_DB", "default"),
    )


defs = dg.Definitions(
    assets=[
        bronze_journal,
        dim_multicol, dim_uf,
        silver_stay_delays,
        dim_mappingbox,
        gold_entries, gold_delays, gold_quality, gold_edor_hourly,
    ],
    sensors=[bronze_sensor,
             silver_sensor, 
             gold_sensor,
             ],
    resources={
        "store": store,
        "clickhouse_client": clickhouse_client,

        # IO managers (la clé = io_manager_key dans @asset)
        "minio": PolarsMinioParquetIOManager(
            endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9100"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            bucket=os.environ.get("MINIO_BUCKET", "lake"),
            prefix=os.environ.get("MINIO_PREFIX", "dashboard"),
            use_ssl=False,
        ),

        "clickhouse_silver": PolarsClickHouseIOManager(
            host=os.environ.get("CH_HOST", "localhost"),
            port=int(os.environ.get("CH_PORT", "8124")),
            database=os.environ.get("CH_DB", "default"),
            username=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            table="silver_stay_delays",
        ),
        
        "clickhouse_gold_entries": PolarsClickHouseIOManager(
            table="gold_entries", 
            host=os.environ.get("CH_HOST","localhost"), 
            port=int(os.environ.get("CH_PORT","8124")),
            database=os.environ.get("CH_DB", "default"),
            username=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            ),
        
        "clickhouse_gold_delays": PolarsClickHouseIOManager(
            table="gold_delays", 
            host=os.environ.get("CH_HOST","localhost"), 
            port=int(os.environ.get("CH_PORT","8124")),
            database=os.environ.get("CH_DB", "default"),
            username=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            ),
        
        "clickhouse_gold_quality": PolarsClickHouseIOManager(
            table="gold_quality", 
            host=os.environ.get("CH_HOST","localhost"),
            port=int(os.environ.get("CH_PORT","8124")),
            database=os.environ.get("CH_DB", "default"),
            username=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            ),
        
        "clickhouse_gold_edor": PolarsClickHouseIOManager(
            table="gold_edor_hourly",
            host=os.environ.get("CH_HOST","localhost"),
            port=int(os.environ.get("CH_PORT","8124")),
            database=os.environ.get("CH_DB", "default"),
            username=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            ),
    },
)