"""
REES46 eCommerce Behavior Data — Ingestion to Bronze Layer (Parquet)
====================================================================
Reads raw CSV files (Oct/Nov 2019), parses timestamps, cleans data,
and writes to Bronze as Parquet partitioned by event_date.

Usage:
    spark-submit ingestion/rees46_ingest.py
"""

import sys
import os
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, TimestampType
)
from pyspark.sql.functions import (
    col, to_date, to_timestamp, year, month, dayofmonth,
    when, lit, count, countDistinct, round as spark_round,
    regexp_replace, trim
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    spark_config, rees46_config,
    REES46_RAW_DIR, REES46_BRONZE_DIR
)


def build_rees46_schema() -> StructType:
    """Build schema for REES46 CSV files."""
    return StructType([
        StructField("event_time", StringType(), nullable=True),
        StructField("event_type", StringType(), nullable=False),
        StructField("product_id", LongType(), nullable=False),
        StructField("category_id", LongType(), nullable=True),
        StructField("category_code", StringType(), nullable=True),
        StructField("brand", StringType(), nullable=True),
        StructField("price", DoubleType(), nullable=True),
        StructField("user_id", LongType(), nullable=False),
    ])


def create_spark_session() -> SparkSession:
    """Create Spark session for REES46 ingestion."""
    builder = SparkSession.builder \
        .appName("REES46Ingestion") \
        .master(spark_config.master)

    for key, value in spark_config.base_conf.items():
        builder = builder.config(key, value)

    builder = builder \
        .config("spark.sql.parquet.compression.codec", "zstd") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")

    return builder.getOrCreate()


def read_rees46_csv(spark: SparkSession, filename: str) -> DataFrame:
    """Read a single REES46 CSV file with schema enforcement."""
    filepath = os.path.join(REES46_RAW_DIR, filename)
    print(f"[INFO] Reading: {filepath}")

    schema = build_rees46_schema()

    df = spark.read.csv(
        filepath,
        schema=schema,
        header=True,
        mode="DROPMALFORMED"
    )

    return df


def clean_rees46(df: DataFrame) -> DataFrame:
    """Clean and transform REES46 data.
    
    - Parse event_time to proper timestamp
    - Extract date components for partitioning
    - Clean category codes and brands
    - Filter invalid records
    """
    # Parse timestamp: "2019-10-01 00:00:00 UTC" format
    df = df.withColumn(
        "event_time_clean",
        to_timestamp(
            regexp_replace(col("event_time"), " UTC$", ""),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

    # Extract date components
    df = df.withColumn("event_date", to_date(col("event_time_clean"))) \
           .withColumn("event_year", year(col("event_time_clean"))) \
           .withColumn("event_month", month(col("event_time_clean"))) \
           .withColumn("event_day", dayofmonth(col("event_time_clean")))

    # Clean category_code: extract main and sub categories
    df = df.withColumn(
        "main_category",
        when(
            col("category_code").isNotNull(),
            regexp_replace(col("category_code"), "\\..*", "")
        ).otherwise(lit("unknown"))
    )
    df = df.withColumn(
        "sub_category",
        when(
            col("category_code").contains("."),
            regexp_replace(col("category_code"), "^[^.]*\\.", "")
        ).otherwise(lit("unknown"))
    )

    # Clean brand
    df = df.withColumn(
        "brand_clean",
        when(col("brand").isNull(), lit("unknown"))
        .otherwise(trim(col("brand")))
    )

    # Filter invalid records
    df = df.filter(
        (col("event_type").isin(rees46_config.event_types)) &
        (col("product_id").isNotNull()) &
        (col("user_id").isNotNull()) &
        (col("price").isNotNull()) &
        (col("price") > 0) &
        (col("event_time_clean").isNotNull())
    )

    # Select final columns
    df = df.select(
        col("event_time_clean").alias("event_time"),
        "event_date",
        "event_year",
        "event_month",
        "event_day",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "main_category",
        "sub_category",
        col("brand_clean").alias("brand"),
        "price",
        "user_id",
    )

    return df


def compute_rees46_stats(df: DataFrame):
    """Compute and print summary statistics."""
    print("\n" + "=" * 50)
    print("  REES46 INGESTION STATISTICS")
    print("=" * 50)

    total = df.count()
    print(f"  Total events: {total:,}")

    # Event type distribution
    print("\n  Event Type Distribution:")
    event_dist = df.groupBy("event_type").agg(
        count("*").alias("count")
    ).orderBy("event_type").collect()
    for row in event_dist:
        pct = row["count"] / total * 100
        print(f"    {row['event_type']:>10}: {row['count']:>12,} ({pct:.1f}%)")

    # Unique counts
    unique_users = df.select(countDistinct("user_id")).first()[0]
    unique_products = df.select(countDistinct("product_id")).first()[0]
    unique_brands = df.select(countDistinct("brand")).first()[0]
    unique_categories = df.select(countDistinct("main_category")).first()[0]

    print(f"\n  Unique users:      {unique_users:,}")
    print(f"  Unique products:   {unique_products:,}")
    print(f"  Unique brands:     {unique_brands:,}")
    print(f"  Unique categories: {unique_categories:,}")

    # Date range
    date_range = df.agg({"event_date": "min"}).first()[0], \
                 df.agg({"event_date": "max"}).first()[0]
    print(f"\n  Date range: {date_range[0]} to {date_range[1]}")

    # Price stats
    price_stats = df.select(
        spark_round(col("price").cast("double"), 2).alias("price")
    ).summary("mean", "min", "max", "50%").collect()
    for row in price_stats:
        print(f"  Price {row['summary']:>5}: ${row['price']}")

    print("=" * 50)


def write_bronze(df: DataFrame):
    """Write to Bronze layer partitioned by event_date."""
    output_path = REES46_BRONZE_DIR
    print(f"\n[INFO] Writing Bronze Parquet to: {output_path}")

    df.write \
        .mode("overwrite") \
        .partitionBy("event_year", "event_month") \
        .parquet(output_path)

    print(f"[INFO] Bronze layer write complete")


def main():
    spark = create_spark_session()
    print("=" * 70)
    print("  REES46 ECOMMERCE BEHAVIOR DATA — BRONZE LAYER INGESTION")
    print("=" * 70)

    start_time = time.time()

    # Read all CSV files
    dfs = []
    for filename in rees46_config.files:
        try:
            df = read_rees46_csv(spark, filename)
            raw_count = df.count()
            print(f"  {filename}: {raw_count:,} raw rows")
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {filename} not found: {e}")
            continue

    if not dfs:
        print("[ERROR] No REES46 data files found. Run scripts/download_data.sh first.")
        spark.stop()
        sys.exit(1)

    # Union all files
    from functools import reduce
    combined_df = reduce(DataFrame.unionByName, dfs)

    # Clean data
    print("\n[INFO] Cleaning and transforming data...")
    clean_df = clean_rees46(combined_df)

    # Compute stats
    compute_rees46_stats(clean_df)

    # Write Bronze
    write_bronze(clean_df)

    elapsed = time.time() - start_time
    print(f"\n[DONE] REES46 ingestion completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
