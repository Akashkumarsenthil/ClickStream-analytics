# ClickStream EC2 Pipeline Updates

These files are the EC2/S3 pipeline updates developed during the live AWS run.
They are intended to be copied into your repository and committed.

## What this patch adds

- `pipeline/01_bronze.py` - raw CSV to S3 Bronze Parquet.
- `pipeline/02_iceberg.py` - fixed Iceberg catalog configuration for Spark 3.4.x.
- `pipeline/03_delta.py` - Delta speed layer with trending products and hourly volume.
- `pipeline/04_hudi.py` - Hudi serving layer with mutable user profiles.
- `pipeline/05_ml.py` - MLflow-tracked CTR, ALS, Product2Vec, and KMeans training scaffold.
- `pipeline/06_benchmark.py` - Iceberg vs Delta vs Hudi benchmark writer.
- `scripts/run_ec2_pipeline.sh` - safe pipeline runner with `set -e` and `set -o pipefail`.
- `config/spark-defaults.spark34.conf` - Spark 3.4.1-compatible config template with no secrets.
- `docs/demo_screenshot_checklist.md` - final presentation screenshot checklist.
- `docs/realtime_simulation_explanation.md` - wording to justify streaming simulation.

## Important security note

Do not commit AWS access keys, Kaggle keys, or any secrets. Use `aws configure`, environment variables, or an EC2 IAM role.
The key used during the live run was exposed and should be rotated/deleted.

## Current validated run state

Validated on EC2 with PySpark 3.4.1:

- Bronze completed: 109,950,743 REES46 rows plus 5,000,000 Criteo-schema rows.
- Iceberg completed: REES46 and Criteo Iceberg tables created under `s3://clickstream-analytics-akash/batch/iceberg/`.
- Delta completed: events and trending outputs created under `s3://clickstream-analytics-akash/speed/delta/`.

