#!/bin/bash
# ============================================================
# Data Download Script
# E-Commerce Clickstream Analytics — DATA 228
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$BASE_DIR/data/raw"

echo "============================================================"
echo "  DATA DOWNLOAD — E-Commerce Clickstream Analytics"
echo "============================================================"
echo ""

# ─── REES46 eCommerce Behavior Data ───
echo "── REES46 eCommerce Behavior Data ──"
echo "Source: Kaggle (mkechinov/ecommerce-behavior-data)"
echo ""

REES46_DIR="$RAW_DIR/rees46"
mkdir -p "$REES46_DIR"

if [ -f "$REES46_DIR/2019-Oct.csv" ] && [ -f "$REES46_DIR/2019-Nov.csv" ]; then
    echo "[SKIP] REES46 data already downloaded"
else
    echo "[INFO] Downloading REES46 data..."
    echo ""
    echo "Option 1: Using Kaggle CLI (recommended)"
    echo "  pip install kaggle"
    echo "  kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store"
    echo "  unzip ecommerce-behavior-data-from-multi-category-store.zip -d $REES46_DIR"
    echo ""
    echo "Option 2: Manual download"
    echo "  1. Go to: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store"
    echo "  2. Download the dataset (requires Kaggle account)"
    echo "  3. Extract to: $REES46_DIR/"
    echo "  4. You should have:"
    echo "     - $REES46_DIR/2019-Oct.csv"
    echo "     - $REES46_DIR/2019-Nov.csv"
    echo ""

    # Try kaggle CLI
    if command -v kaggle &>/dev/null; then
        echo "[INFO] Kaggle CLI found. Downloading..."
        cd "$REES46_DIR"
        kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store
        unzip -o ecommerce-behavior-data-from-multi-category-store.zip
        rm -f ecommerce-behavior-data-from-multi-category-store.zip
        echo "[DONE] REES46 data downloaded"
    else
        echo "[WARN] Kaggle CLI not found. Install with: pip install kaggle"
        echo "[WARN] Then set up API credentials: https://www.kaggle.com/docs/api"
    fi
fi

echo ""

# ─── Criteo 1TB Click Logs ───
echo "── Criteo 1TB Click Logs ──"
echo "Source: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/"
echo ""

CRITEO_DIR="$RAW_DIR/criteo"
mkdir -p "$CRITEO_DIR"

if ls "$CRITEO_DIR"/day_* 1>/dev/null 2>&1; then
    echo "[SKIP] Criteo data already downloaded"
else
    echo "[INFO] Downloading Criteo data (subset)..."
    echo ""
    echo "The full dataset is ~1.3 TB. We will download a subset."
    echo ""
    echo "Option 1: Download specific days (recommended for dev)"
    echo "  Each day is ~30 GB compressed. For initial development,"
    echo "  download 1-2 days:"
    echo ""

    # Download first N days
    NUM_DAYS=${1:-2}  # Default: 2 days
    echo "  Downloading $NUM_DAYS day(s)..."

    for day in $(seq 0 $((NUM_DAYS - 1))); do
        if [ -f "$CRITEO_DIR/day_${day}" ] || [ -f "$CRITEO_DIR/day_${day}.gz" ]; then
            echo "  [SKIP] day_${day} already exists"
        else
            URL="https://storage.googleapis.com/criteo-research-datasets/day_${day}.gz"
            echo "  [DL] Downloading day_${day}..."
            wget -q --show-progress -O "$CRITEO_DIR/day_${day}.gz" "$URL" 2>/dev/null || \
            curl -L -o "$CRITEO_DIR/day_${day}.gz" "$URL" 2>/dev/null || \
            echo "  [WARN] Failed to download day_${day}. Manual download required."

            if [ -f "$CRITEO_DIR/day_${day}.gz" ]; then
                echo "  [INFO] Decompressing day_${day}..."
                gunzip "$CRITEO_DIR/day_${day}.gz" || true
            fi
        fi
    done

    echo ""
    echo "Option 2: Full dataset download"
    echo "  Visit: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/"
    echo "  Register and download all 24 days (day_0 through day_23)"
    echo "  Extract to: $CRITEO_DIR/"
fi

echo ""
echo "============================================================"
echo "  DATA DOWNLOAD SUMMARY"
echo "============================================================"
echo ""
echo "  REES46 directory:  $REES46_DIR"
if [ -f "$REES46_DIR/2019-Oct.csv" ]; then
    echo "    ✅ 2019-Oct.csv  ($(du -h "$REES46_DIR/2019-Oct.csv" | cut -f1))"
else
    echo "    ❌ 2019-Oct.csv  (missing)"
fi
if [ -f "$REES46_DIR/2019-Nov.csv" ]; then
    echo "    ✅ 2019-Nov.csv  ($(du -h "$REES46_DIR/2019-Nov.csv" | cut -f1))"
else
    echo "    ❌ 2019-Nov.csv  (missing)"
fi

echo ""
echo "  Criteo directory:  $CRITEO_DIR"
criteo_count=$(ls "$CRITEO_DIR"/day_* 2>/dev/null | wc -l)
echo "    Files found: $criteo_count day file(s)"

echo ""
echo "============================================================"
echo "  NEXT STEPS"
echo "============================================================"
echo "  1. Verify data files are in place"
echo "  2. Run: ./scripts/setup_env.sh"
echo "  3. Run: ./scripts/run_pipeline.sh"
echo "============================================================"
