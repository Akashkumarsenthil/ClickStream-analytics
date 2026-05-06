# E-Commerce Clickstream Analytics & Recommendation Engine at Scale

**DATA 228 — Spring 2026 | Team 3**

> Akash Kumar · Shriram Dundigalla · Pramod Satya Dindukurthi Centhur Velan R.S.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                      │
│   Criteo 1TB Click Logs (~150-350 GB subset)                            │
│   REES46 eCommerce Behavior (~14 GB, 285M events)                       │
└──────────────┬───────────────────────────────────┬───────────────────────┘
               │                                   │
       ┌───────▼───────┐                   ┌───────▼───────┐
       │  Bronze Layer  │                   │  Bronze Layer  │
       │   (Parquet)    │                   │   (Parquet)    │
       └───────┬───────┘                   └───────┬───────┘
               │                                   │
    ┌──────────┼──────────────────┬─────────────────┼──────────┐
    │          │                  │                 │          │
┌───▼────┐ ┌──▼─────┐    ┌──────▼──────┐   ┌─────▼────┐ ┌───▼────┐
│ Batch  │ │ Speed  │    │   Serving   │   │   ML     │ │Bench-  │
│ Layer  │ │ Layer  │    │   Layer     │   │ Models   │ │ mark   │
│Iceberg │ │ Delta  │    │   Hudi      │   │ MLflow   │ │Compare │
└───┬────┘ └──┬─────┘    └──────┬──────┘   └─────┬────┘ └───┬────┘
    │         │                 │                 │          │
    └─────────┴────────┬────────┴─────────────────┴──────────┘
                       │
              ┌────────▼────────┐
              │    FastAPI       │
              │  Serving Layer   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Streamlit     │
              │   Dashboard      │
              └─────────────────┘
```

## Project Structure

```
ecommerce-clickstream-analytics/
├── config/
│   └── settings.py              # Central configuration
├── ingestion/
│   ├── criteo_ingest.py         # Criteo 1TB → Bronze (Parquet)
│   └── rees46_ingest.py         # REES46 → Bronze (Parquet)
├── batch_layer/
│   └── iceberg_batch.py         # Apache Iceberg: historical analytics
├── speed_layer/
│   └── delta_streaming.py       # Delta Lake + Structured Streaming
├── serving_layer/
│   └── hudi_serving.py          # Apache Hudi: mutable user profiles
├── ml/
│   ├── ctr/
│   │   └── ctr_model.py         # CTR prediction (Criteo)
│   ├── recommender/
│   │   └── als_recommender.py   # Collaborative filtering (ALS)
│   ├── session/
│   │   └── product2vec.py       # Session-based Product2Vec
│   └── segmentation/
│       └── customer_segments.py # K-Means customer segmentation
├── api/
│   └── main.py                  # FastAPI serving endpoints
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── benchmark/
│   └── format_benchmark.py      # Iceberg vs Delta vs Hudi benchmark
├── scripts/
│   ├── download_data.sh         # Data download automation
│   ├── setup_env.sh             # Environment setup
│   └── run_pipeline.sh          # Full pipeline orchestration
├── tests/
│   └── test_pipeline.py         # Integration tests
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Quick Start

### 1. Environment Setup
```bash
# Clone and setup
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Or manual setup
pip install -r requirements.txt
```

### 2. Download Data
```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

### 3. Run Full Pipeline
```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

### 4. Run Individual Components
```bash
# Ingestion
spark-submit ingestion/criteo_ingest.py
spark-submit ingestion/rees46_ingest.py

# Batch Layer (Iceberg)
spark-submit batch_layer/iceberg_batch.py

# Speed Layer (Delta + Streaming)
spark-submit speed_layer/delta_streaming.py

# Serving Layer (Hudi)
spark-submit serving_layer/hudi_serving.py

# ML Models
spark-submit ml/ctr/ctr_model.py
spark-submit ml/recommender/als_recommender.py
spark-submit ml/session/product2vec.py
spark-submit ml/segmentation/customer_segments.py

# Benchmark
spark-submit benchmark/format_benchmark.py

# API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Dashboard
streamlit run dashboard/app.py
```

## Datasets

| Dataset | Records | Size | Features |
|---------|---------|------|----------|
| Criteo 1TB Click Logs | ~4.3B (using 150-350GB subset) | ~1.3 TB | 1 label + 13 numerical + 26 categorical |
| REES46 eCommerce | ~285M events | ~14 GB | event_time, event_type, product_id, category, brand, price, user_id |

## ML Models

| Model | Dataset | Task | Metrics |
|-------|---------|------|---------|
| Logistic Regression / GBT | Criteo | CTR Prediction | AUC-ROC, Log Loss |
| ALS Collaborative Filtering | REES46 | Product Recommendations | Precision@K, Recall@K, NDCG@K |
| Product2Vec (Word2Vec) | REES46 | Similar Products | Cosine Similarity, Hit Rate |
| K-Means Clustering | REES46 | Customer Segmentation | Silhouette Score, Inertia |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend/{user_id}` | GET | Personalized product recommendations |
| `/predict/ctr` | POST | Click-through rate prediction |
| `/trending` | GET | Real-time trending products |
| `/user/{user_id}/segment` | GET | Customer segment classification |
| `/similar/{product_id}` | GET | Similar products via Product2Vec |
| `/health` | GET | API health check |

## Tech Stack

- **Compute**: Apache Spark 3.5+, PySpark
- **Table Formats**: Apache Iceberg, Delta Lake, Apache Hudi
- **ML**: Spark MLlib, MLflow
- **Serving**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Storage**: Parquet (Bronze), Iceberg/Delta/Hudi (Silver/Gold)
