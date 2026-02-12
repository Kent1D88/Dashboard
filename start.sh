export DAGSTER_HOME="$(pwd)/dagster_home"
export IO_BACKEND=minio
export MINIO_ENDPOINT=http://localhost:9100
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=lake
export MINIO_PREFIX=dashboard

export IO_BACKEND=clickhouse
export CH_HOST=localhost
export CH_PORT=8124
export CH_DB=default
export CH_USER=default
export CH_PASSWORD=
export CH_ORDER_BY=NOREC

mkdir -p "$DAGSTER_HOME"

dagster dev -m myproj.definitions

