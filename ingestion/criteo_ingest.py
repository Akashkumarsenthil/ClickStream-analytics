"""
Criteo 1TB Click Logs — Ingestion to Bronze Layer (Parquet)
===========================================================
Reads raw TSV files from the Criteo dataset (subset of days),
applies schema, handles missing values, and writes to Bronze as Parquet.

Usage:
    spark-submit ingestion/criteo_ingest.py [--days 5]
"""

import argparse
import sys
import os
import time
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, FloatType
)
from pyspark.sql.functions import (
    col, lit, monotonically_increasing_id, input_file_name,
    regexp_extract, when, count, sum as spark_sum, mean
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    spark_config, criteo_config,
    CRITEO_RAW_DIR, CRITEO_BRONZE_DIR
)


def build_criteo_schema() -> StructType:
    """Build schema for Criteo TSV files.
    
    Format: label \t int1 \t int2 ... int13 \t cat1 \t cat2 ... cat26
    """
    fields = [StructField(criteo_config.label_col, IntegerType(), nullable=False)]

    for col_name in criteo_config.int_feature_cols:
        fields.append(StructField(col_name, FloatType(), nullable=True))

    for col_name in criteo_config.cat_feature_cols:
        fields.append(StructField(col_name, StringType(), nullable=True))

    return StructType(fields)


def create_spark_session() -> SparkSession:
    """Create Spark session optimized for Criteo ingestion."""
    builder = SparkSession.builder \
        .appName("CriteoIngestion") \
        .master(spark_config.master)

    for key, value in spark_config.base_conf.items():
        builder = builder.config(key, value)

    # Criteo-specific tuning
    builder = builder \
        .config("spark.sql.files.maxPartitionBytes", "256m") \
        .config("spark.sql.parquet.compression.codec", "zstd")

    return builder.getOrCreate()


def ingest_single_day(spark: SparkSession, day: int) -> DataFrame:
    """Ingest a single day of Criteo data.
    
    Args:
        spark: Active SparkSession
        day: Day number (0-23)
    
    Returns:
        DataFrame with schema-applied Criteo data for that day
    """
    day_path = os.path.join(CRITEO_RAW_DIR, f"day_{day}")

    # Also check for single file format
    if not os.path.exists(day_path):
        day_path = os.path.join(CRITEO_RAW_DIR, f"day_{day}.gz")
    if not os.path.exists(day_path):
        day_path = os.path.join(CRITEO_RAW_DIR, f"day_{day}.tsv")

    print(f"[INFO] Ingesting day {day} from: {day_path}")
    schema = build_criteo_schema()

    df = spark.read.csv(
        day_path,
        sep=criteo_config.delimiter,
        schema=schema,
        header=False,
        mode="DROPMALFORMED"
    )

    # Add metadata columns
    df = df.withColumn("day", lit(day)) \
           .withColumn("row_id", monotonically_increasing_id())

    return df


def compute_ingestion_stats(df: DataFrame) -> dict:
    """Compute summary statistics for ingested data."""
    total_rows = df.count()

    # Label distribution
    label_dist = df.groupBy("label").count().collect()
    label_counts = {row["label"]: row["count"] for row in label_dist}

    # Missing values per integer feature
    int_null_counts = {}
    for col_name in criteo_config.int_feature_cols:
        null_count = df.filter(col(col_name).isNull()).count()
        int_null_counts[col_name] = null_count

    return {
        "total_rows": total_rows,
        "label_distribution": label_counts,
        "ctr": label_counts.get(1, 0) / total_rows if total_rows > 0 else 0,
        "int_feature_nulls": int_null_counts,
    }


def fill_missing_values(df: DataFrame) -> DataFrame:
    """Fill missing values in integer features with column median.
    
    Categorical features: fill with "__MISSING__" placeholder.
    """
    # Fill integer features with 0 (fast approach; median can be done later)
    for col_name in criteo_config.int_feature_cols:
        df = df.withColumn(
            col_name,
            when(col(col_name).isNull(), lit(0.0)).otherwise(col(col_name))
        )

    # Fill categorical features with placeholder
    for col_name in criteo_config.cat_feature_cols:
        df = df.withColumn(
            col_name,
            when(col(col_name).isNull(), lit("__MISSING__")).otherwise(col(col_name))
        )

    return df


def write_bronze(df: DataFrame, partition_col: str = "day"):
    """Write DataFrame to Bronze layer as partitioned Parquet."""
    output_path = CRITEO_BRONZE_DIR
    print(f"[INFO] Writing Bronze Parquet to: {output_path}")

    df.write \
        .mode("overwrite") \
        .partitionBy(partition_col) \
        .parquet(output_path)

    print(f"[INFO] Bronze layer write complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Criteo 1TB Click Logs Ingestion")
    parser.add_argument("--days", type=int, default=criteo_config.num_days_subset,
                        help="Number of days to ingest (default: 5)")
    parser.add_argument("--fill-missing", action="store_true", default=True,
                        help="Fill missing values during ingestion")
    parser.add_argument("--stats", action="store_true", default=True,
                        help="Compute and print ingestion statistics")
    args = parser.parse_args()

    spark = create_spark_session()
    print("=" * 70)
    print("  CRITEO 1TB CLICK LOGS — BRONZE LAYER INGESTION")
    print(f"  Days to ingest: {args.days}")
    print("=" * 70)

    start_time = time.time()

    # Ingest each day and union
    day_dfs = []
    for day in range(args.days):
        try:
            day_df = ingest_single_day(spark, day)
            day_dfs.append(day_df)
            row_count = day_df.count()
            print(f"  Day {day}: {row_count:,} rows ingested")
        except Exception as e:
            print(f"  [WARN] Day {day} not found or failed: {e}")
            continue

    if not day_dfs:
        print("[ERROR] No data files found. Run scripts/download_data.sh first.")
        spark.stop()
        sys.exit(1)

    # Union all days
    combined_df = reduce(DataFrame.unionByName, day_dfs)
    print(f"\n[INFO] Total rows across {len(day_dfs)} days: {combined_df.count():,}")

    # Fill missing values
    if args.fill_missing:
        print("[INFO] Filling missing values...")
        combined_df = fill_missing_values(combined_df)

    # Compute stats
    if args.stats:
        print("\n[INFO] Computing ingestion statistics...")
        stats = compute_ingestion_stats(combined_df)
        print(f"  Total Rows: {stats['total_rows']:,}")
        print(f"  CTR (Click-Through Rate): {stats['ctr']:.4f}")
        print(f"  Label Distribution: {stats['label_distribution']}")
        top_null = sorted(stats['int_feature_nulls'].items(),
                          key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 features by nulls: {top_null}")

    # Write to Bronze
    write_bronze(combined_df)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Criteo ingestion completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
