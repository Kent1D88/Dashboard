#!/usr/bin/env bash
set -euo pipefail

export DAGSTER_HOME="$(pwd)/dagster_home"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# MinIO
export MINIO_ENDPOINT=http://localhost:9100
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=lake
export MINIO_PREFIX=dashboard

# ClickHouse
export CH_HOST=localhost
export CH_PORT=8123
export CH_DB=default
export CH_USER=admin
export CH_PASSWORD=admin
export CH_ORDER_BY=NOREC

# Debug
export DASH_DEBUG=1
export DASH_DEBUG_STEPS=bronze,silver,io,gold
export DASH_DEBUG_ROWS=3

mkdir -p "$DAGSTER_HOME"
touch "$DAGSTER_HOME/dagster.yaml"  # enlève le warning "no dagster.yaml"

dagster dev -m myproj.definitions