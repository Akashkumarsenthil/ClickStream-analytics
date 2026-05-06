# ============================================================
# Makefile — E-Commerce Clickstream Analytics
# DATA 228 Spring 2026 | Team 3
# ============================================================

.PHONY: help setup download ingest batch speed serve ml-ctr ml-als ml-p2v ml-seg \
        benchmark api dashboard test quality pipeline clean docker-up docker-down

SHELL := /bin/bash
SPARK_SUBMIT := spark-submit --master local[*] --driver-memory 8g

ICEBERG_PKG := org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0
DELTA_PKG := io.delta:delta-spark_2.12:3.1.0
HUDI_PKG := org.apache.hudi:hudi-spark3.5-bundle_2.12:0.14.1

# ─── Help ───
help: ## Show available targets
	@echo ""
	@echo "  E-Commerce Clickstream Analytics — Make Targets"
	@echo "  ================================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ─── Setup ───
setup: ## Install dependencies and create directories
	@chmod +x scripts/*.sh
	@./scripts/setup_env.sh

download: ## Download datasets (Criteo + REES46)
	@./scripts/download_data.sh

# ─── Ingestion ───
ingest: ingest-criteo ingest-rees46 ## Run all data ingestion

ingest-criteo: ## Ingest Criteo data to Bronze
	$(SPARK_SUBMIT) ingestion/criteo_ingest.py --days 2

ingest-rees46: ## Ingest REES46 data to Bronze
	$(SPARK_SUBMIT) ingestion/rees46_ingest.py

# ─── Lambda Architecture Layers ───
batch: ## Run Batch Layer (Apache Iceberg)
	$(SPARK_SUBMIT) --packages $(ICEBERG_PKG) batch_layer/iceberg_batch.py

speed: ## Run Speed Layer (Delta Lake + Streaming)
	$(SPARK_SUBMIT) --packages $(DELTA_PKG) speed_layer/delta_streaming.py

serve: ## Run Serving Layer (Apache Hudi)
	$(SPARK_SUBMIT) --packages $(HUDI_PKG) serving_layer/hudi_serving.py

layers: batch speed serve ## Run all architecture layers

# ─── ML Models ───
ml-ctr: ## Train CTR models (LR + GBT)
	$(SPARK_SUBMIT) ml/ctr/ctr_model.py

ml-als: ## Train ALS Recommender
	$(SPARK_SUBMIT) ml/recommender/als_recommender.py

ml-p2v: ## Train Product2Vec
	$(SPARK_SUBMIT) ml/session/product2vec.py

ml-seg: ## Train Customer Segmentation (K-Means)
	$(SPARK_SUBMIT) ml/segmentation/customer_segments.py

ml: ml-ctr ml-als ml-p2v ml-seg ## Train all ML models

# ─── Benchmark ───
benchmark: ## Run table format benchmark
	$(SPARK_SUBMIT) --packages $(ICEBERG_PKG),$(DELTA_PKG),$(HUDI_PKG) \
		benchmark/format_benchmark.py

# ─── Data Quality ───
quality: ## Run data quality validation
	$(SPARK_SUBMIT) scripts/data_quality.py

# ─── API & Dashboard ───
api: ## Start FastAPI server
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard
	streamlit run dashboard/app.py

mlflow-ui: ## Start MLflow tracking UI
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns

# ─── Testing ───
test: ## Run all tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ -v --tb=short --cov=. --cov-report=html

# ─── Full Pipeline ───
pipeline: ingest layers ml benchmark ## Run complete pipeline
	@echo ""
	@echo "✅ Full pipeline complete!"
	@echo "Start API:       make api"
	@echo "Start Dashboard: make dashboard"

# ─── Docker ───
docker-up: ## Start all services via Docker Compose
	docker-compose up -d
	@echo "Services:"
	@echo "  Spark Master:  http://localhost:8080"
	@echo "  API:           http://localhost:8000"
	@echo "  Dashboard:     http://localhost:8501"
	@echo "  MLflow:        http://localhost:5000"

docker-down: ## Stop all Docker services
	docker-compose down

docker-build: ## Build Docker images
	docker-compose build

# ─── Cleanup ───
clean: ## Remove generated data and artifacts
	rm -rf data/bronze/* data/silver/* data/gold/*
	rm -rf data/iceberg_warehouse data/delta_tables data/hudi_tables
	rm -rf ml_artifacts mlruns benchmark_results
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache htmlcov
	find . -name "*.pyc" -delete
	@echo "Cleaned all generated data and artifacts"

clean-data: ## Remove raw data downloads
	rm -rf data/raw/criteo/* data/raw/rees46/*
	@echo "Cleaned raw data"
