"""
Speed Layer — Delta Lake + Spark Structured Streaming
=====================================================
Simulates real-time event streaming from REES46 data using
Spark Structured Streaming with Delta Lake as the sink.
Detects live trending products and real-time category metrics.

Usage:
    spark-submit --packages io.delta:delta-spark_2.12:3.1.0 \
        speed_layer/delta_streaming.py
"""

import sys
import os
import time
import json
import shutil

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg,
    when, lit, window, current_timestamp, expr,
    from_json, to_json, struct, round as spark_round,
    dense_rank, row_number, collect_list, size,
    max as spark_max, min as spark_min
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    DoubleType, TimestampType
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    spark_config, rees46_config,
    REES46_BRONZE_DIR, DELTA_PATH
)

# Delta table paths
DELTA_EVENTS_PATH = os.path.join(DELTA_PATH, "events_stream")
DELTA_TRENDING_PATH = os.path.join(DELTA_PATH, "trending_products")
DELTA_REALTIME_METRICS_PATH = os.path.join(DELTA_PATH, "realtime_metrics")
DELTA_CHECKPOINT_PATH = os.path.join(DELTA_PATH, "_checkpoints")
STREAM_SOURCE_PATH = os.path.join(DELTA_PATH, "_stream_source")


def create_delta_session() -> SparkSession:
    """Create Spark session with Delta Lake extensions."""
    builder = SparkSession.builder \
        .appName("DeltaSpeedLayer") \
        .master(spark_config.master)

    for key, value in spark_config.delta_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def prepare_stream_source(spark: SparkSession):
    """Prepare a file-based streaming source from Bronze data.
    
    Splits Bronze data into small Parquet files to simulate
    micro-batch streaming.
    """
    print("[INFO] Preparing streaming source from Bronze data...")

    # Clean previous stream source
    if os.path.exists(STREAM_SOURCE_PATH):
        shutil.rmtree(STREAM_SOURCE_PATH)
    os.makedirs(STREAM_SOURCE_PATH, exist_ok=True)

    # Read Bronze and repartition into small files
    events_df = spark.read.parquet(REES46_BRONZE_DIR)

    # Take a subset and split into micro-batches (small files)
    # Each file simulates ~1 minute of data
    sample_df = events_df.repartition(100)  # 100 micro-batch files

    sample_df.write \
        .mode("overwrite") \
        .parquet(STREAM_SOURCE_PATH)

    file_count = len([f for f in os.listdir(STREAM_SOURCE_PATH) if f.endswith('.parquet')])
    total_rows = events_df.count()
    print(f"[INFO] Stream source prepared: {file_count} files, {total_rows:,} total events")


def create_delta_tables(spark: SparkSession):
    """Initialize Delta tables for streaming sinks."""
    print("[INFO] Creating Delta Lake tables...")

    # Events stream table
    events_schema = spark.read.parquet(REES46_BRONZE_DIR).schema

    empty_events = spark.createDataFrame([], events_schema)
    empty_events.write \
        .format("delta") \
        .mode("overwrite") \
        .save(DELTA_EVENTS_PATH)

    print(f"[INFO] Delta events table created at: {DELTA_EVENTS_PATH}")


def start_event_stream(spark: SparkSession) -> None:
    """Start structured streaming from file source to Delta Lake.
    
    Simulates real-time ingestion using file-based streaming.
    """
    print("[INFO] Starting event stream (file source → Delta Lake)...")

    # Read schema from Bronze
    schema = spark.read.parquet(REES46_BRONZE_DIR).schema

    # Create streaming source
    stream_df = spark.readStream \
        .schema(schema) \
        .option("maxFilesPerTrigger", 5) \
        .parquet(STREAM_SOURCE_PATH)

    # Write to Delta table
    query = stream_df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", os.path.join(DELTA_CHECKPOINT_PATH, "events")) \
        .start(DELTA_EVENTS_PATH)

    return query


def compute_trending_products(spark: SparkSession) -> None:
    """Compute trending products using streaming aggregation.
    
    Uses sliding window aggregation to detect products with
    rapidly increasing view/cart/purchase counts.
    """
    print("[INFO] Starting trending products stream...")

    schema = spark.read.parquet(REES46_BRONZE_DIR).schema

    stream_df = spark.readStream \
        .schema(schema) \
        .option("maxFilesPerTrigger", 5) \
        .parquet(STREAM_SOURCE_PATH)

    # Windowed aggregation for trending detection
    trending = stream_df \
        .withWatermark("event_time", "1 hour") \
        .groupBy(
            window(col("event_time"), "1 hour", "15 minutes"),
            col("product_id"),
            col("main_category"),
            col("brand"),
        ) \
        .agg(
            count("*").alias("event_count"),
            spark_sum(when(col("event_type") == "view", 1).otherwise(0)).alias("views"),
            spark_sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("carts"),
            spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
            countDistinct("user_id").alias("unique_users"),
            avg("price").alias("avg_price"),
        )

    # Compute trend score: weighted sum of interactions
    trending = trending.withColumn(
        "trend_score",
        spark_round(
            col("views") * 1.0 + col("carts") * 3.0 + col("purchases") * 5.0,
            2
        )
    )

    # Write trending products to Delta
    query = trending.writeStream \
        .format("delta") \
        .outputMode("complete") \
        .option("checkpointLocation", os.path.join(DELTA_CHECKPOINT_PATH, "trending")) \
        .start(DELTA_TRENDING_PATH)

    return query


def compute_realtime_category_metrics(spark: SparkSession) -> None:
    """Compute real-time category-level metrics."""
    print("[INFO] Starting real-time category metrics stream...")

    schema = spark.read.parquet(REES46_BRONZE_DIR).schema

    stream_df = spark.readStream \
        .schema(schema) \
        .option("maxFilesPerTrigger", 5) \
        .parquet(STREAM_SOURCE_PATH)

    category_metrics = stream_df \
        .withWatermark("event_time", "1 hour") \
        .groupBy(
            window(col("event_time"), "30 minutes"),
            col("main_category"),
        ) \
        .agg(
            count("*").alias("total_events"),
            countDistinct("user_id").alias("active_users"),
            countDistinct("product_id").alias("products_interacted"),
            spark_sum(when(col("event_type") == "purchase", col("price")).otherwise(0)).alias("revenue"),
            spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
        )

    query = category_metrics.writeStream \
        .format("delta") \
        .outputMode("complete") \
        .option("checkpointLocation", os.path.join(DELTA_CHECKPOINT_PATH, "category_metrics")) \
        .start(DELTA_REALTIME_METRICS_PATH)

    return query


def run_batch_trending_analysis(spark: SparkSession):
    """Batch analysis on Delta tables after streaming completes.
    
    Reads the Delta trending table and identifies top trending products.
    """
    print("\n" + "=" * 50)
    print("  DELTA LAKE — TRENDING PRODUCT ANALYSIS")
    print("=" * 50)

    try:
        trending_df = spark.read.format("delta").load(DELTA_TRENDING_PATH)

        # Top 20 trending products
        print("\n[QUERY] Top 20 Trending Products:")
        trending_df \
            .orderBy(col("trend_score").desc()) \
            .select(
                "product_id", "brand", "main_category",
                "views", "carts", "purchases",
                "unique_users", "trend_score",
                "window.start", "window.end"
            ) \
            .show(20, truncate=False)

        # Trending by category
        print("\n[QUERY] Trending by Category:")
        trending_df.groupBy("main_category").agg(
            spark_sum("trend_score").alias("total_trend_score"),
            count("*").alias("trending_products"),
            spark_sum("purchases").alias("total_purchases"),
        ).orderBy(col("total_trend_score").desc()).show(15, truncate=False)

    except Exception as e:
        print(f"[WARN] Trending table not yet available: {e}")


def demonstrate_delta_features(spark: SparkSession):
    """Demonstrate Delta Lake features: time-travel, versioning."""
    print("\n" + "=" * 50)
    print("  DELTA LAKE FEATURES DEMONSTRATION")
    print("=" * 50)

    try:
        # Show Delta table history
        print("\n[INFO] Delta Events Table History:")
        spark.sql(f"DESCRIBE HISTORY delta.`{DELTA_EVENTS_PATH}`").show(truncate=False)

        # Table details
        print("\n[INFO] Delta Events Table Details:")
        spark.sql(f"DESCRIBE DETAIL delta.`{DELTA_EVENTS_PATH}`").show(truncate=False)

        # Show version 0 (time travel)
        print("\n[INFO] Reading version 0 of Delta table (time-travel):")
        v0_df = spark.read.format("delta").option("versionAsOf", 0).load(DELTA_EVENTS_PATH)
        print(f"  Version 0 row count: {v0_df.count():,}")

    except Exception as e:
        print(f"[WARN] Delta demo failed: {e}")


def main():
    spark = create_delta_session()
    print("=" * 70)
    print("  SPEED LAYER — DELTA LAKE + STRUCTURED STREAMING")
    print("  Simulated Real-Time Event Processing")
    print("=" * 70)

    start_time = time.time()

    # 1. Prepare streaming source
    prepare_stream_source(spark)

    # 2. Create Delta tables
    create_delta_tables(spark)

    # 3. Start streaming queries
    event_query = start_event_stream(spark)
    trending_query = compute_trending_products(spark)
    category_query = compute_realtime_category_metrics(spark)

    # 4. Let streams process for a bit
    print("\n[INFO] Streaming active — processing micro-batches...")
    print("[INFO] Waiting 60 seconds for streams to process...")

    try:
        # Wait for streams to process all files
        event_query.awaitTermination(timeout=60)
    except Exception:
        pass

    # 5. Stop streams
    print("[INFO] Stopping streams...")
    for query in [event_query, trending_query, category_query]:
        try:
            query.stop()
        except Exception:
            pass

    # 6. Batch analysis on streamed data
    run_batch_trending_analysis(spark)

    # 7. Demonstrate Delta features
    demonstrate_delta_features(spark)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Speed layer completed in {elapsed:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
