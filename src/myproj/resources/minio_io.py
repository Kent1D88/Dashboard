from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
from botocore.exceptions import ClientError

import boto3
import dagster as dg
import polars as pl

from myproj.utils.debug import debug_enabled

@dataclass
class PolarsMinioParquetIOManager(dg.IOManager):
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str
    prefix: str = "dashboard"
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
        path = "/".join(asset_key.path)
        if partition_key is not None:
            return f"{self.prefix}/{path}/partition={partition_key}.parquet"
        return f"{self.prefix}/{path}/data.parquet"

    def handle_output(self, context: dg.OutputContext, obj: pl.DataFrame):
        # Ici, c’est bien la partition de L’ASSET produit, donc ok.
        pkey = context.partition_key if context.has_partition_key else None
        key = self._key(context.asset_key, pkey)
        if debug_enabled("io"):
            context.log.debug(
                f"[MinIO][WRITE] bucket={self.bucket} key={key} "
                f"rows={obj.height} cols={obj.width}"
            )
        
        buf = io.BytesIO()
        obj.write_parquet(buf)
        buf.seek(0)

        self._client().put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

        context.add_output_metadata(
            {"minio_bucket": self.bucket, "minio_key": key, "rows": obj.height}
        )

    def load_input(self, context: dg.InputContext) -> pl.DataFrame:
        asset_is_partitioned = getattr(context, "has_asset_partitions", False)
        pkey = context.partition_key if (asset_is_partitioned and context.has_partition_key) else None

        # nouvelle convention
        key = self._key(context.asset_key, pkey)
        context.log.info(f"[MinIO load_input] bucket={self.bucket} key={key}")
        if debug_enabled("io"):
            context.log.debug(
             f"[MinIO][READ] bucket={self.bucket} key={key}"
            )
            
        s3 = self._client()
        try:
            obj = s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            # fallback ancienne convention (uniquement si partitionné)
            if code in {"NoSuchKey", "404"} and pkey is not None:
                legacy_key = f"{self.prefix}/{'/'.join(context.asset_key.path)}/{pkey}.parquet"
                context.log.warning(f"[MinIO load_input] key not found, trying legacy key={legacy_key}")
                obj = s3.get_object(Bucket=self.bucket, Key=legacy_key)
            else:
                raise

        data = obj["Body"].read()
        df = pl.read_parquet(io.BytesIO(data))
        
        if debug_enabled("io"):
            context.log.debug(
                f"[MinIO][READ DONE] rows={df.height} cols={df.width}"
            )
            
        return df