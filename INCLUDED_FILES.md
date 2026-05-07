# Final Repository File List

This cleaned package includes only the files that should be committed for the final project repo.

## Included

```text
README.md
requirements.txt
.gitignore
index.html
api/__init__.py
api/main.py
dashboard/app.py
pipeline/01_bronze.py
pipeline/02_iceberg.py
pipeline/03_delta.py
pipeline/04_hudi.py
pipeline/05_ml.py
pipeline/06_benchmark.py
config/spark-defaults.spark34.conf
scripts/run_ec2_pipeline.sh
docs/demo_screenshot_checklist.md
docs/realtime_simulation_explanation.md
docs/images/architecture_diagram.png
docs/images/benchmark_results.png
docs/images/iceberg_query.png
docs/images/delta_query.png
docs/images/hudi_query.png
```

## Notes before commit

- `index.html` contains `API_BASE_URL = "https://YOUR_NGROK_OR_API_URL"`; replace this with the active ngrok/API URL before the demo.
- `dashboard/app.py` reads `API_BASE_URL` from an environment variable and defaults to `http://localhost:8000`.
- `api/main.py` is the cleaned FastAPI serving/live-ingestion app.
- `pipeline/` is the EC2-tested, numbered pipeline structure and should be treated as the source of truth.
