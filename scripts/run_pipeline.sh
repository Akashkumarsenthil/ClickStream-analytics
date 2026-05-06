#!/bin/bash
# ============================================================
# Full Pipeline Orchestration Script
# E-Commerce Clickstream Analytics — DATA 228
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# Spark packages
ICEBERG_PKG="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0"
DELTA_PKG="io.delta:delta-spark_2.12:3.1.0"
HUDI_PKG="org.apache.hudi:hudi-spark3.5-bundle_2.12:0.14.1"
ALL_PKGS="$ICEBERG_PKG,$DELTA_PKG,$HUDI_PKG"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

TOTAL_START=$(date +%s)

step() {
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  STEP $1: $2${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
}

log_time() {
    local start=$1
    local end=$(date +%s)
    local elapsed=$((end - start))
    echo -e "${YELLOW}  ⏱  Completed in ${elapsed}s${NC}"
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  E-COMMERCE CLICKSTREAM ANALYTICS — FULL PIPELINE       ║"
echo "║  DATA 228 Spring 2026 | Team 3                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Ingestion (Bronze Layer) ───
step "1/8" "DATA INGESTION → BRONZE LAYER (Parquet)"
STEP_START=$(date +%s)

echo "  [1a] Ingesting Criteo Click Logs..."
spark-submit \
    --master local[*] \
    --driver-memory 8g \
    ingestion/criteo_ingest.py --days 2 2>&1 | tail -5

echo "  [1b] Ingesting REES46 eCommerce data..."
spark-submit \
    --master local[*] \
    --driver-memory 8g \
    ingestion/rees46_ingest.py 2>&1 | tail -5

log_time $STEP_START

# ─── Step 2: Batch Layer (Iceberg) ───
step "2/8" "BATCH LAYER — APACHE ICEBERG"
STEP_START=$(date +%s)

spark-submit \
    --master local[*] \
    --driver-memory 8g \
    --packages "$ICEBERG_PKG" \
    batch_layer/iceberg_batch.py 2>&1 | tail -10

log_time $STEP_START

# ─── Step 3: Speed Layer (Delta) ───
step "3/8" "SPEED LAYER — DELTA LAKE + STRUCTURED STREAMING"
STEP_START=$(date +%s)

spark-submit \
    --master local[*] \
    --driver-memory 8g \
    --packages "$DELTA_PKG" \
    speed_layer/delta_streaming.py 2>&1 | tail -10

log_time $STEP_START

# ─── Step 4: Serving Layer (Hudi) ───
step "4/8" "SERVING LAYER — APACHE HUDI (MoR)"
STEP_START=$(date +%s)

spark-submit \
    --master local[*] \
    --driver-memory 8g \
    --packages "$HUDI_PKG" \
    serving_layer/hudi_serving.py 2>&1 | tail -10

log_time $STEP_START

# ─── Step 5: CTR Model ───
step "5/8" "ML: CTR PREDICTION (LR + GBT)"
STEP_START=$(date +%s)

spark-submit \
    --master local[*] \
    --driver-memory 8g \
    ml/ctr/ctr_model.py 2>&1 | tail -15

log_time $STEP_START

# ─── Step 6: Recommender ───
step "6/8" "ML: ALS COLLABORATIVE FILTERING"
STEP_START=$(date +%s)

spark-submit \
    --master local[*] \
    --driver-memory 8g \
    ml/recommender/als_recommender.py 2>&1 | tail -15

log_time $STEP_START

# ─── Step 7: Product2Vec + Segmentation ───
step "7/8" "ML: PRODUCT2VEC + CUSTOMER SEGMENTATION"
STEP_START=$(date +%s)

echo "  [7a] Training Product2Vec..."
spark-submit \
    --master local[*] \
    --driver-memory 8g \
    ml/session/product2vec.py 2>&1 | tail -10

echo "  [7b] Training K-Means Segmentation..."
spark-submit \
    --master local[*] \
    --driver-memory 8g \
    ml/segmentation/customer_segments.py 2>&1 | tail -10

log_time $STEP_START

# ─── Step 8: Benchmark ───
step "8/8" "TABLE FORMAT BENCHMARK"
STEP_START=$(date +%s)

spark-submit \
    --master local[*] \
    --driver-memory 8g \
    --packages "$ALL_PKGS" \
    benchmark/format_benchmark.py 2>&1 | tail -20

log_time $STEP_START

# ─── Summary ───
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ PIPELINE COMPLETE                                    ║"
echo "║  Total time: ${TOTAL_ELAPSED}s                                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Start the API:                                          ║"
echo "║    uvicorn api.main:app --host 0.0.0.0 --port 8000       ║"
echo "║                                                          ║"
echo "║  Start the Dashboard:                                    ║"
echo "║    streamlit run dashboard/app.py                         ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
