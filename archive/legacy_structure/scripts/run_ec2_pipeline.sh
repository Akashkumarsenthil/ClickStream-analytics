#!/bin/bash
set -e
set -o pipefail

cd /home/ubuntu/pipeline

export PYSPARK_PYTHON=python3

run_stage() {
  local label="$1"
  local script="$2"
  local log_file="$3"
  echo "=== STARTING ${label} ==="
  python3 "${script}" 2>&1 | tee "${log_file}"
  echo "=== ${label} DONE ==="
}

run_stage "BRONZE" "01_bronze.py" "/home/ubuntu/logs_01.txt"
run_stage "ICEBERG" "02_iceberg.py" "/home/ubuntu/logs_02.txt"
run_stage "DELTA" "03_delta.py" "/home/ubuntu/logs_03.txt"
run_stage "HUDI" "04_hudi.py" "/home/ubuntu/logs_04.txt"
run_stage "ML" "05_ml.py" "/home/ubuntu/logs_05.txt"
run_stage "BENCHMARK" "06_benchmark.py" "/home/ubuntu/logs_06.txt"

echo "=== PIPELINE COMPLETE ==="
