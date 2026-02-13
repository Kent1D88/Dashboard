# =========================================
# Dashboard infra control
# =========================================

COMPOSE=docker compose -f docker-compose.yaml

# Volumes locaux
MINIO_VOL=../volumes/minio
CLICKHOUSE_VOL=../volumes/clickhouse

.PHONY: up down restart logs ps reset prune-volumes check

# -----------------------------------------
# Start services
# -----------------------------------------
up:
	@echo "Starting services..."
	$(COMPOSE) up -d
	@echo "Services started."

# -----------------------------------------
# Stop services
# -----------------------------------------
down:
	@echo "Stopping services..."
	$(COMPOSE) down
	@echo "Services stopped."

# -----------------------------------------
# Restart services
# -----------------------------------------
restart:
	@echo "Restarting services..."
	$(COMPOSE) down
	$(COMPOSE) up -d
	@echo "Services restarted."

# -----------------------------------------
# Show running containers
# -----------------------------------------
ps:
	$(COMPOSE) ps

# -----------------------------------------
# Logs (follow)
# -----------------------------------------
logs:
	$(COMPOSE) logs -f

# -----------------------------------------
# Remove DB volumes (DANGEROUS)
# -----------------------------------------
prune-volumes:
	@echo "⚠️  Removing MinIO and ClickHouse volumes..."
	$(COMPOSE) down
	rm -rf $(MINIO_VOL)
	rm -rf $(CLICKHOUSE_VOL)
	@echo "Volumes removed."

# -----------------------------------------
# Full reset (containers + volumes)
# -----------------------------------------
reset:
	@echo "⚠️  FULL RESET"
	$(COMPOSE) down
	rm -rf $(MINIO_VOL)
	rm -rf $(CLICKHOUSE_VOL)
	$(COMPOSE) up -d
	@echo "Reset complete."

# -----------------------------------------
# Health check
# -----------------------------------------
check:
	@echo "Checking services..."
	@echo ""
	@echo "MinIO API:"
	curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9100 || true
	@echo ""
	@echo "MinIO Console:"
	curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9101 || true
	@echo ""w
	@echo "ClickHouse:"
	curl -s -u admin:admin 'http://localhost:8123/?query=SELECT%201' || true
	@echo ""
	$(COMPOSE) ps

# -----------------------------------------
# Full clean + restart infra + dagster
# -----------------------------------------
dev-fresh:
	@echo "========================================="
	@echo "FRESH DEV ENVIRONMENT"
	@echo "========================================="

	@echo "Cleaning dagster_home..."
	rm -rf ../../dagster_home
	mkdir -p ../../dagster_home

	@echo "Stopping containers..."
	$(COMPOSE) down -v --remove-orphans

	@echo "Removing volumes..."
	rm -rf $(MINIO_VOL)
	rm -rf $(CLICKHOUSE_VOL)

	@echo "Starting services..."
	$(COMPOSE) up -d

	@echo "Waiting for services to become healthy..."

	@echo "Waiting for MinIO API..."
	@until curl -s http://localhost:9100/minio/health/live > /dev/null; do \
		echo "  MinIO not ready..."; \
		sleep 2; \
	done

	@echo "Waiting for ClickHouse..."
	@until curl -s -u admin:admin 'http://localhost:8123/?query=SELECT%201' > /dev/null; do \
		echo "  ClickHouse not ready..."; \
		sleep 2; \
	done

	@echo "Services ready."
	@echo "Starting Dagster..."
	cd ../../ && ./start.sh