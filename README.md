# E-Commerce Clickstream Analytics & Recommendation Engine at Scale

**DATA 228 — Big Data Technologies | Spring 2026**  
**Team 3:** Akash Kumar · Shriram Dundigalla · Pramod Satya Dindukurthi · Centhur Velan R.S.

---

## Overview

This project implements a production-style e-commerce clickstream analytics and recommendation platform on real large-scale data. The system ingests 110M+ historical e-commerce events, processes them through a multi-layer lakehouse architecture on Amazon S3, trains four machine learning models, benchmarks Apache Iceberg, Delta Lake, and Apache Hudi, and serves results through FastAPI and a live Streamlit dashboard. A GitHub Pages demo site emits live click events to simulate real-time ingestion.

---

## Implementation Status

| Component | Status | Verified Output |
|---|---|---|
| AWS S3 bucket | ✅ Complete | `clickstream-analytics-akash` |
| EC2 compute | ✅ Complete | `r5.4xlarge` — 16 vCPUs, 128 GB RAM |
| Raw REES46 upload | ✅ Complete | Oct + Nov 2019 CSVs in `raw-csv/rees46/` |
| Bronze layer | ✅ Complete | Parquet in `bronze/rees46/` and `bronze/criteo/` |
| Apache Iceberg batch layer | ✅ Complete | Metadata + snapshot files in `batch/iceberg/` |
| Delta Lake speed layer | ✅ Complete | Event table + trending in `speed/delta/` |
| Apache Hudi serving layer | ✅ Complete | 15.1M user profiles in `serving/hudi/` |
| ML pipeline | ✅ Complete | CTR, ALS, Product2Vec, K-Means — all trained |
| Benchmarking | ✅ Complete | Iceberg vs Delta vs Hudi results saved to S3 |
| FastAPI serving layer | ✅ Complete | 7 endpoints tested via public API tunnel |
| Live event ingestion | ✅ Complete | GitHub Pages events written to `live-events/` |
| Streamlit dashboard | ✅ Complete | Live dashboard with analytics, ML, benchmark, and event feed |

---

## Architecture

```
                    Historical Data Sources
          ┌─────────────────────────────────────┐
          │  REES46 eCommerce Behavior Dataset  │
          │  Criteo-style CTR Dataset           │
          └──────────────┬──────────────────────┘
                         │
                         ▼
              Raw CSV Layer — Amazon S3
          s3://clickstream-analytics-akash/raw-csv/
                         │
                         ▼
              Bronze Layer — Parquet on S3
          ┌─────────────────────────────────────┐
          │  bronze/rees46/   (110M events)     │
          │  bronze/criteo/   (5M rows)         │
          └──────────────┬──────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   Batch Layer      Speed Layer     Serving Layer
  Apache Iceberg    Delta Lake      Apache Hudi
  batch/iceberg/    speed/delta/    serving/hudi/
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
                  Gold / ML Layer
          ┌─────────────────────────────────────┐
          │  CTR Prediction    (Criteo)         │
          │  ALS Recommender   (REES46)         │
          │  Product2Vec       (REES46 sessions)│
          │  K-Means Segments  (user profiles)  │
          └──────────────┬──────────────────────┘
                         │
                         ▼
                  Serving Layer
          ┌─────────────────────────────────────┐
          │  FastAPI Endpoints                  │
          │  Streamlit Dashboard                │
          └──────────────┬──────────────────────┘
                         │
                         ▼
              Live Demo Integration
          ┌─────────────────────────────────────┐
          │  GitHub Pages Product Clicks        │
          │  FastAPI /live-event                │
          │  S3 live-events/ JSON               │
          └─────────────────────────────────────┘
```

---

## Data Sources

### REES46 eCommerce Behavior Dataset

Real user-product interaction events from a multi-category online store. Powers product recommendations, trending analytics, customer segmentation, and user profile aggregation.

| Field | Description |
|---|---|
| `event_time` | Timestamp of user action |
| `event_type` | `view`, `cart`, or `purchase` |
| `product_id` | Product identifier |
| `category_code` | Product category |
| `brand` | Product brand |
| `price` | Product price |
| `user_id` | User identifier |

**Verified scale:** 109,950,743 events processed (Oct + Nov 2019, both months)

### Criteo-Style CTR Dataset

Binary classification dataset used for click-through-rate prediction. Schema matches the Criteo 1TB Click Logs format.

| Column | Description |
|---|---|
| `label` | Binary click label |
| `I1–I13` | Numerical features |
| `C1–C26` | Hashed categorical features |

**Verified scale:** 5,000,000 Criteo-schema rows processed

---

## S3 Data Layout

```
s3://clickstream-analytics-akash/
├── raw-csv/
│   └── rees46/
│       ├── 2019-Oct.csv
│       └── 2019-Nov.csv
├── bronze/
│   ├── rees46/
│   │   └── event_date=YYYY-MM-DD/*.snappy.parquet
│   └── criteo/
│       └── *.snappy.parquet
├── batch/
│   └── iceberg/
│       └── clickstream/
│           ├── rees46_events/
│           │   ├── data/
│           │   └── metadata/
│           └── criteo_clicks/
│               ├── data/
│               └── metadata/
├── speed/
│   └── zone/delta/
│       ├── events/
│       └── trending/
├── serving/
│   └── hudi/
│       └── user_profiles/
│           └── .hoodie/
├── ml-artifacts/
│   └── ml_summary.json
├── benchmark/
│   └── results.json
└── live-events/
    └── year=YYYY/month=MM/day=DD/event_<uuid>.json
```

---

## Pipeline Scripts

| Script | Purpose | Input | Output |
|---|---|---|---|
| [`01_bronze.py`](https://github.com/Akashkumarsenthil/ClickStream-analytics/blob/main/pipeline/01_bronze.py) | Convert raw data to Bronze Parquet | Raw CSV + generated Criteo-schema data | `bronze/rees46/`, `bronze/criteo/` |
| [`02_iceberg.py`](https://github.com/Akashkumarsenthil/ClickStream-analytics/blob/main/pipeline/02_iceberg.py) | Create Iceberg batch tables | Bronze Parquet | `batch/iceberg/` |
| [`03_delta.py`](https://github.com/Akashkumarsenthil/ClickStream-analytics/blob/main/pipeline/03_delta.py) | Create Delta speed layer and trending products | Bronze REES46 Parquet | `speed/delta/events/`, `speed/delta/trending/` |
| [`04_hudi.py`](https://github.com/Akashkumarsenthil/ClickStream-analytics/blob/main/pipeline/04_hudi.py) | Build mutable user profiles | Bronze REES46 Parquet | `serving/hudi/user_profiles/` |
| [`05_ml.py`](https://github.com/Akashkumarsenthil/ClickStream-analytics/blob/main/pipeline/05_ml.py) | Train ML models and save summary | Bronze + Hudi data | `ml-artifacts/ml_summary.json` |
| [`06_benchmark.py`](https://github.com/Akashkumarsenthil/ClickStream-analytics/blob/main/pipeline/06_benchmark.py) | Benchmark table formats | Iceberg, Delta, Hudi | `benchmark/results.json` |

---

## Lakehouse Layers

### Bronze Layer — Parquet on S3

Cleaned, columnar Parquet versions of the raw datasets, partitioned by `event_date` for efficient downstream reads.

```
REES46 rows:   109,950,743
Criteo rows:     5,000,000
Total Bronze:  114,950,743
```

### Batch Layer — Apache Iceberg

Historical analytics with time-travel, schema evolution, and ACID transactions.

Tables created:
- `local.clickstream.rees46_events`
- `local.clickstream.criteo_clicks`

**Verified SQL output:**
![Iceberg Query Results](docs/images/iceberg_query.png)

### Speed Layer — Delta Lake

Real-time event processing simulation with ACID transactions, transaction log, and partition pruning on `event_type`.

**Verified Trending output:**
![Delta Trending Results](docs/images/delta_query.png)

### Serving Layer — Apache Hudi

Mutable user profiles using Merge-on-Read (MOR) tables with upsert support for incremental profile updates.

Output: `serving/hudi/user_profiles/` — **15,095,144 user profiles**

**Verified Hudi Profile output:**
![Hudi Query Results](docs/images/hudi_query.png)

---

## Machine Learning Layer

| Model | Dataset | Purpose | Verified Result |
|---|---|---|---|
| CTR Prediction (GBT) | Criteo-schema data | Predict click probability | AUC-ROC: 0.5010 |
| ALS Recommender | REES46 user-product interactions | Recommend products to users | Sample recommendations generated |
| Product2Vec (Word2Vec) | REES46 product sequences | Find similar products | 37,157 product embeddings |
| K-Means Segmentation | Hudi user profiles | Segment customers | 5 customer clusters |

ML summary saved to `s3://clickstream-analytics-akash/ml-artifacts/ml_summary.json`:

```json
{
  "ctr_auc_roc": 0.5010295405943604,
  "als_sample_recommendation_users": 10,
  "product_embedding_count": 37157,
  "kmeans_segment_counts": {
    "0": 1768689,
    "1": 184448,
    "2": 16177,
    "3": 1566,
    "4": 29120
  }
}
```

---

## Benchmark Results

Query benchmark across all three table formats on representative workloads running on a single-node `r5.4xlarge` EC2 against S3-backed tables.

| Format | Average Query Time | Runs | Role |
|---|---|---|---|
| Delta Lake | **4.17s** | 6.92s, 2.89s, 2.69s | Speed layer / trending analytics |
| Apache Iceberg | 5.01s | 9.25s, 3.06s, 2.73s | Batch analytics / historical SQL |
| Apache Hudi | 6.52s | 7.29s, 6.22s, 6.05s | Serving layer / user profiles |

**Verified Benchmark timing:**
![Benchmark Timing](docs/images/benchmark_results.png)

---

## FastAPI Serving Layer

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API health and project metadata |
| `/trending` | GET | Delta speed-layer trending products |
| `/recommend/{user_id}` | GET | ALS-style product recommendations |
| `/predict/ctr` | POST | CTR click probability prediction |
| `/user/{user_id}/segment` | GET | K-Means customer segment |
| `/similar/{product_id}` | GET | Product2Vec similar products |
| `/live-event` | POST | Write GitHub Pages click event to S3 |

---

## Live GitHub Pages Integration

A GitHub Pages product demo page emits REES46-compatible click events through FastAPI into S3, simulating a real e-commerce frontend.

**Live event flow:**
```
GitHub Pages product click
        ↓
FastAPI /live-event
        ↓
S3 live-events/ JSON
        ↓
Streamlit dashboard reads live event feed
        ↓
Historical baseline + live event overlay
```

---

## Streamlit Dashboard

Interactive analytics application presenting all pipeline outputs in a single interface.

**Live URL:** https://clickstream-analytics-xzmq8edm4qh3uzkgm7epgl.streamlit.app

---

## Technology Stack

| Category | Technologies |
|---|---|
| Cloud Storage | Amazon S3 |
| Compute | Amazon EC2 `r5.4xlarge` |
| Processing | Apache Spark 3.4, PySpark |
| File Format | Parquet (Snappy compression) |
| Lakehouse Formats | Apache Iceberg, Delta Lake, Apache Hudi |
| Machine Learning | Spark MLlib, MLflow |
| API Serving | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Live Demo Frontend | GitHub Pages, JavaScript |
| Cloud Access | AWS CLI, boto3 |
| Development | Python 3.10, Java 11 |

---

## Verified Scale

| Metric | Value |
|---|---|
| REES46 events processed | 109,950,743 |
| Criteo-schema rows processed | 5,000,000 |
| Total Bronze records | 114,950,743 |
| Hudi user profiles | 15,095,144 |
| Product2Vec embeddings | 37,157 |
| ML models trained | 4 |
| API endpoints | 7 |
| Table formats benchmarked | 3 |

---

## Notes and Limitations

- The REES46 dataset was processed at full scale using both available months (Oct + Nov 2019), totalling 109.95M events.
- Benchmark results are workload-specific and reflect single-node EC2 size, S3 round-trip latency, table layout, and query type.
- The GitHub Pages integration simulates a real e-commerce frontend by emitting REES46-compatible events through FastAPI into S3.

---

## Summary

This project demonstrates a complete end-to-end lakehouse-based clickstream analytics system using real e-commerce event data at scale. The pipeline proves:

- Large-scale clickstream ingestion to Amazon S3
- Bronze Parquet conversion with 114.95M total records
- Apache Iceberg for historical analytics and time-travel queries
- Delta Lake for speed-layer trending analytics with partition pruning
- Apache Hudi for mutable user-profile serving with Merge-on-Read upserts
- CTR prediction, ALS recommendations, Product2Vec embeddings, and K-Means segmentation
- ML and benchmark artifacts persisted to S3 with MLflow tracking
- FastAPI endpoints serving analytics and model outputs
- GitHub Pages live clickstream simulation with real-time S3 event ingestion
- Streamlit dashboard for real-time monitoring and project presentation
