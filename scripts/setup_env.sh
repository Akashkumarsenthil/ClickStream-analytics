#!/bin/bash
# ============================================================
# Environment Setup Script
# E-Commerce Clickstream Analytics — DATA 228
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "  ENVIRONMENT SETUP"
echo "============================================================"
echo ""

# ─── Check Python ───
echo "[CHECK] Python version..."
python3 --version 2>/dev/null || { echo "[ERROR] Python 3 not found"; exit 1; }

# ─── Check Java ───
echo "[CHECK] Java version..."
java -version 2>/dev/null || {
    echo "[WARN] Java not found. Installing OpenJDK 11..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y openjdk-11-jdk
    elif command -v brew &>/dev/null; then
        brew install openjdk@11
    else
        echo "[ERROR] Please install Java 11+ manually"
        exit 1
    fi
}

# ─── Check Spark ───
echo "[CHECK] Apache Spark..."
if command -v spark-submit &>/dev/null; then
    spark-submit --version 2>/dev/null | head -3
    echo "  ✅ Spark found"
else
    echo "[WARN] Apache Spark not found in PATH"
    echo ""
    echo "Install Spark 3.5+:"
    echo "  Option 1: pip install pyspark==3.5.4"
    echo "  Option 2: Download from https://spark.apache.org/downloads.html"
    echo ""
    echo "Setting SPARK_HOME for pip-installed PySpark..."
    export PYSPARK_PYTHON=python3
fi

# ─── Install Python Dependencies ───
echo ""
echo "[SETUP] Installing Python dependencies..."
cd "$BASE_DIR"

pip install --quiet --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install --quiet -r requirements.txt
    echo "  ✅ Dependencies installed"
else
    echo "[WARN] requirements.txt not found"
fi

# ─── Create Directories ───
echo ""
echo "[SETUP] Creating directory structure..."
mkdir -p data/{raw/criteo,raw/rees46,bronze,silver,gold}
mkdir -p data/{iceberg_warehouse,delta_tables,hudi_tables}
mkdir -p ml_artifacts
mkdir -p benchmark_results
mkdir -p mlruns
echo "  ✅ Directories created"

# ─── Verify Setup ───
echo ""
echo "============================================================"
echo "  SETUP VERIFICATION"
echo "============================================================"

echo -n "  Python:     "; python3 --version
echo -n "  Java:       "; java -version 2>&1 | head -1
echo -n "  pip:        "; pip --version | head -1

echo ""
echo "  PySpark:    $(python3 -c 'import pyspark; print(pyspark.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  FastAPI:    $(python3 -c 'import fastapi; print(fastapi.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  Streamlit:  $(python3 -c 'import streamlit; print(streamlit.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  MLflow:     $(python3 -c 'import mlflow; print(mlflow.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  Plotly:     $(python3 -c 'import plotly; print(plotly.__version__)' 2>/dev/null || echo 'Not installed')"

echo ""
echo "============================================================"
echo "  ✅ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. Download data:  ./scripts/download_data.sh"
echo "    2. Run pipeline:   ./scripts/run_pipeline.sh"
echo "============================================================"
