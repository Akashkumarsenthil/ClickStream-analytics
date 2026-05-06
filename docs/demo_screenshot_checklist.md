# Final Demo Screenshot Checklist

Use these screenshots to prove scale, table formats, ML, serving, and the live simulation.

## Minimum set

1. EC2 instance running: `r5.4xlarge`, `us-east-1`, 200 GB gp3.
2. S3 raw files: `raw-csv/rees46/2019-Oct.csv` and `2019-Nov.csv`.
3. Bronze Parquet proof: `bronze/rees46/event_date=.../*.snappy.parquet` and `bronze/criteo/*.snappy.parquet`.
4. Bronze terminal proof: row counts and total records.
5. Iceberg terminal proof: `Iceberg batch layer complete`, SQL output, and `109,950,743` count.
6. Iceberg S3 proof: `batch/iceberg/.../metadata/`, `snap-*.avro`, `v1.metadata.json`, `version-hint.text`.
7. Delta terminal proof: trending products and hourly event volume.
8. Delta S3 proof: `speed/delta/events/_delta_log/` and `speed/delta/trending/_delta_log/`.
9. Hudi S3 proof: `serving/hudi/user_profiles/.hoodie/`.
10. MLflow UI: four runs for CTR, ALS, Product2Vec, and KMeans.
11. Benchmark output: `benchmark/results.json` or chart.
12. FastAPI `/docs` page with serving endpoints.
13. Streamlit dashboard with charts and live event action.
14. Live event proof: `live-events/` before and after dashboard/replay event.
15. Databricks notebook/workspace screenshot for visual credibility.

## Useful commands

```bash
aws s3 ls s3://clickstream-analytics-akash/raw-csv/rees46/
aws s3 ls s3://clickstream-analytics-akash/bronze/rees46/ --recursive | grep -i parquet | head -20
aws s3 ls s3://clickstream-analytics-akash/bronze/criteo/ --recursive | grep -i parquet | head -20
aws s3 ls s3://clickstream-analytics-akash/bronze/ --recursive --summarize | tail -5
aws s3 ls s3://clickstream-analytics-akash/batch/iceberg/ --recursive | grep -E 'metadata|snap|manifest|version' | head -80
aws s3 ls s3://clickstream-analytics-akash/speed/delta/events/ --recursive | grep _delta_log | head -40
aws s3 ls s3://clickstream-analytics-akash/serving/hudi/user_profiles/ --recursive | head -80
aws s3 cp s3://clickstream-analytics-akash/benchmark/results.json -
aws s3 ls s3://clickstream-analytics-akash/live-events/ --recursive | tail
```

## Recommended captions

- Bronze: "114.9M records converted from raw CSV into Snappy Parquet on S3."
- Iceberg: "Batch layer with table metadata, snapshots, manifest files, and time-travel capability."
- Delta: "Speed layer for simulated real-time product trend aggregation."
- Hudi: "Serving layer with mutable user profiles and upsert support."
- MLflow: "Four ML workloads tracked with metrics and artifacts."
- Live simulation: "Real historical events replayed as micro-batches to simulate live clickstream traffic."
