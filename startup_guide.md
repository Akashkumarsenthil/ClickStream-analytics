# E-Commerce Clickstream Analytics Startup Guide

Welcome to the project! This guide will help you get everything up and running smoothly. The project is heavily automated using standard `Makefile` commands and Docker Compose.

## 1. Quick Local Environment Setup

If you want to run scripts locally (like Spark models, tests, or manual ingestion), first make sure your virtual environment is ready and your data is downloaded.

```bash
# Set up Python environment and dependencies
make setup

# Download datasets (Criteo and REES46)
make download
```

> [!TIP]
> The `make setup` command invokes a script (`scripts/setup_env.sh`) that sets up directories and installs all Python dependencies in `requirements.txt`.

## 2. Docker: The Easy Way

The fastest way to get the core platform layers (Spark Master, API Server, Dashboard, and MLflow) running is using Docker Compose.

```bash
# Build and bring up all containers in the background
make docker-up
```

### Access Points

Once `make docker-up` completes, you can access the following services in your browser:

- **Dashboard (Streamlit):** [http://localhost:8501](http://localhost:8501)
- **API Server (FastAPI Interface):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking UI:** [http://localhost:5000](http://localhost:5000)
- **Spark Master UI:** [http://localhost:8080](http://localhost:8080)

To shut things down later:

```bash
make docker-down
```

## 3. Running the Data Pipelines

You can selectively run specific parts of the Lambda Architecture locally via Spark Submit:

- **Ingestion (Bronze):** `make ingest` (runs both Criteo and REES46 ingestion)
- **Batch Layer (Iceberg):** `make batch`
- **Speed Layer (Delta):** `make speed`
- **Serving Layer (Hudi):** `make serve`
- **All Layers:** `make layers`

## 4. Training Machine Learning Models

To train specific machine learning models on the ingested data, run:

```bash
make ml-ctr   # Train Click-Through Rate models
make ml-als   # Train ALS Recommender
make ml-p2v   # Train Product2Vec model
make ml-seg   # Train Customer Segmentation using K-Means

# Train ALL models sequentially
make ml
```

## 5. One-Shot Full Pipeline

If you want the entire system to run from scratch (Ingest → Lambda Architecture Layers → ML Training → Benchmark):

```bash
make pipeline
```

> [!IMPORTANT]
> The full pipeline requires lots of memory as it triggers Apache Iceberg, Delta Lake, Hudi Spark operations as well as Model Training. Make sure your Docker or local Spark cluster is provisioned thoroughly.

## 6. Utilities and Cleanup

If you need to wipe out the data artifacts or generated outputs to start fresh:

- **Remove processed data:** `make clean`
- **Wipe out raw downloads:** `make clean-data`
- **Run Data Quality Tests:** `make quality`
