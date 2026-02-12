from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import boto3
import dagster as dg
import polars as pl


@dataclass
class PolarsMinioParquetIOManager(dg.IOManager):
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str
    prefix: str = "dashboard"  # racine des objets (ex: dashboard/bronze/journal/1.parquet)
    region_name: str = "us-east-1"
    use_ssl: bool = False

    def _client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name,
            use_ssl=self.use_ssl,
        )

    def _key(self, asset_key: dg.AssetKey, partition_key: Optional[str]) -> str:
        path = "/".join(asset_key.path)  # ex: bronze/journal
        if partition_key:
            return f"{self.prefix}/{path}/{partition_key}.parquet"
        return f"{self.prefix}/{path}/data.parquet"

    def handle_output(self, context: dg.OutputContext, obj: pl.DataFrame):
        pkey = context.partition_key if context.has_partition_key else None
        key = self._key(context.asset_key, pkey)

        buf = io.BytesIO()
        obj.write_parquet(buf)
        buf.seek(0)

        self._client().put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

        context.add_output_metadata({"minio_bucket": self.bucket, "minio_key": key, "rows": obj.height})

    def load_input(self, context: dg.InputContext) -> pl.DataFrame:
        pkey = context.partition_key if context.has_partition_key else None
        key = self._key(context.asset_key, pkey)

        obj = self._client().get_object(Bucket=self.bucket, Key=key)
        data = obj["Body"].read()
        return pl.read_parquet(io.BytesIO(data))